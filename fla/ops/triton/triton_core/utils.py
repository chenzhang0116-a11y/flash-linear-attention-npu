import inspect
import itertools
import contextlib
import os
import functools
import warnings
import logging
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple
from packaging import version

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
import triton.runtime.driver as driver

logger = logging.getLogger(__name__)

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.

    If FLA_DISABLE_TENSOR_CACHE environment variable is set to '1', caching is disabled.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        # Skip cache if FLA_DISABLE_TENSOR_CACHE is set
        if FLA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_indices_list( 
    cu_seqlens: list[int],
    chunk_size: int
 ) -> list[int]: 
    indices = []
    
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        length = end - start
        
        if length <= 0:
            continue
            
        num_chunks = (length + chunk_size - 1) // chunk_size
        
        for chunk_id in range(num_chunks):
            indices.append((i))
            indices.append((chunk_id))
            
    return indices


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


if hasattr(triton.language, '_experimental_make_tensor_descriptor'):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, 'make_tensor_descriptor'):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """


    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'


def map_triton_backend_to_torch_device() -> str:
    backend = get_available_device()        # 'cuda' | 'hip' | 'xpu' | 'cpu' | ...
    return {'cuda': 'cuda', 'hip': 'cuda', 'xpu': 'xpu'}.get(backend, backend)


device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_platform = get_available_device()
device_name = map_triton_backend_to_torch_device()

# device_torch_lib must be accessible; for 'cpu' fallback, default to torch.cuda if available
if device == 'cpu':
    # Fallback: try to get a usable device lib, or use a dummy
    try:
        device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda'
    except Exception:
        device = 'cuda'
device_torch_lib = getattr(torch, device, torch)

is_amd = (device_platform == 'hip')
is_nvidia = (device_platform == 'cuda')
is_npu = (device_platform == 'npu')
is_nvidia_hopper = (
            is_nvidia and ('NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9))

is_tf32_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8)
is_tma_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 9) \
                   and os.environ.get('FLA_NO_USE_TMA', '0') != '1' and \
                   (hasattr(triton.language, '_experimental_make_tensor_descriptor') or hasattr(triton.language,
                                                                                                'make_tensor_descriptor'))
is_gather_supported = hasattr(triton.language, 'gather')

# NPU (Ascend) hardware limit: coreDim (product of all grid dimensions) ≤ 65535.
# We use 65534 as the threshold to stay safely under the limit.
MAX_COREDIM = 65534

# Uppercase aliases for backward compatibility with fla.utils import convention.
# KDA and other upstream code imports these as uppercase (e.g., IS_AMD, IS_GATHER_SUPPORTED).
IS_AMD = is_amd
IS_NVIDIA = is_nvidia
IS_NPU = is_npu
IS_NVIDIA_HOPPER = is_nvidia_hopper
IS_TF32_SUPPORTED = is_tf32_supported
IS_GATHER_SUPPORTED = is_gather_supported
IS_TMA_SUPPORTED = is_tma_supported

# autotune cache support
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

if is_nvidia and not is_tf32_supported:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)

if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    if device == 'npu':
        # NPU fallback for older PyTorch versions
        autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
        autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

        def custom_device_ctx(index: int):
            return torch.npu.device(index)
    else:
        assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'
        autocast_custom_fwd = device_torch_lib.amp.custom_fwd
        autocast_custom_bwd = device_torch_lib.amp.custom_bwd

        def custom_device_ctx(index: int):
            return torch.cuda.device(index)


def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.

    Args:
        no_guard_contiguous: If True, skip all contiguous checks. If a list of parameter names, skip contiguous check for those parameters.
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        # Get function signature for parameter name mapping
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Convert no_guard_contiguous to list of parameter names if it's a list
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            # Process args with parameter name mapping
            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    # For *args beyond signature, use position as name
                    param_name = f"__arg_{i}"

                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            # Process kwargs
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    # Handle direct usage without parentheses: @input_guard
    if fn is not None:
        return decorator(fn)

    return decorator


def _cpu_device_warning():
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tl.log(x.to(tl.float32))
    @triton.jit
    def log2(x): return tl.log2(x.to(tl.float32))


# Gather operation: tl.gather wrapper or fallback for platforms that don't support it
if not is_gather_supported:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather


# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def get_autotune_config(
    multibuffer_list: tuple = (False,),
    unit_flag_list: tuple = (False,),
    limit_auto_multi_buffer_only_for_local_buffer_list: tuple = (False,),
    limit_auto_multi_buffer_of_local_buffer_list: tuple = ("no-l0c",),
    set_workspace_multibuffer_list: tuple = (2, 4),
    enable_hivm_auto_cv_balance_list: tuple = (True,),
    tile_mix_vector_loop_num_list: tuple = (2, 4),
    tile_mix_cube_loop_num_list: tuple = (2, 4),
):
    configs = []
    for (
        multibuffer,
        unit_flag,
        limit_auto_multi_buffer_only_for_local_buffer,
        limit_auto_multi_buffer_of_local_buffer,
    ) in itertools.product(
        list(multibuffer_list),
        list(unit_flag_list),
        list(limit_auto_multi_buffer_only_for_local_buffer_list),
        list(limit_auto_multi_buffer_of_local_buffer_list),
    ):

        if limit_auto_multi_buffer_only_for_local_buffer:
            configs.append(
                triton.Config(
                    {},
                    multibuffer=multibuffer,
                    unit_flag=unit_flag,
                    limit_auto_multi_buffer_only_for_local_buffer=limit_auto_multi_buffer_only_for_local_buffer,
                    limit_auto_multi_buffer_of_local_buffer=limit_auto_multi_buffer_of_local_buffer,
                )
            )
        else:
            for (
                set_workspace_multibuffer,
                enable_hivm_auto_cv_balance,
                tile_mix_vector_loop,
                tile_mix_cube_loop,
            ) in itertools.product(
                list(set_workspace_multibuffer_list),
                list(enable_hivm_auto_cv_balance_list),
                list(tile_mix_vector_loop_num_list),
                list(tile_mix_cube_loop_num_list),
            ):
                configs.append(
                    triton.Config(
                        {},
                        multibuffer=multibuffer,
                        unit_flag=unit_flag,
                        limit_auto_multi_buffer_only_for_local_buffer=limit_auto_multi_buffer_only_for_local_buffer,
                        limit_auto_multi_buffer_of_local_buffer=limit_auto_multi_buffer_of_local_buffer,
                        set_workspace_multibuffer=set_workspace_multibuffer,
                        enable_hivm_auto_cv_balance=enable_hivm_auto_cv_balance,
                        tile_mix_vector_loop=tile_mix_vector_loop,
                        tile_mix_cube_loop=tile_mix_cube_loop,
                    )
                )
    return configs


def get_npu_properties():
    return driver.active.utils.get_device_properties(torch.npu.current_device())


# ============================================================================
# Autotune helpers for NPU coreDim filtering
# ============================================================================


def _get_autotune_layer(kernel):
    """Get the Autotuner layer from a Triton kernel.

    When @triton.heuristics wraps @triton.autotune (or @fla_cache_autotune),
    the outermost object is a Heuristics instance which lacks .configs.
    The Autotuner lives one level down at kernel.fn.  When heuristics is
    absent, kernel itself is the Autotuner.
    """
    if hasattr(kernel, 'configs'):
        return kernel
    # Heuristics wraps Autotuner: kernel.fn is the Autotuner
    if hasattr(kernel, 'fn') and hasattr(kernel.fn, 'configs'):
        return kernel.fn
    raise AttributeError(
        f"Cannot find .configs on {type(kernel).__name__} kernel. "
        f"Expected @triton.autotune or @triton.heuristics(@triton.autotune(...)) decorator chain."
    )


def filter_safe_configs(kernel, grid_fn, max_coredim=MAX_COREDIM, runtime_meta=None):
    """
    Filter autotune configs, keeping only those where grid product ≤ max_coredim.
    On non-NPU devices, this is a no-op that returns the original configs.

    Args:
        kernel: A Triton kernel with a .configs attribute (from @triton.autotune),
                or a @triton.heuristics wrapper around such a kernel.
        grid_fn: A callable that takes a dict of meta parameters and returns
                 a tuple of grid dimensions.
        max_coredim: Maximum allowed product of grid dimensions (default: MAX_COREDIM).
        runtime_meta: Optional dict of runtime constexpr values (e.g. {'S': 128})
                      that the grid function may reference via meta[...] but are
                      not part of the autotune config kwargs.  These are merged
                      into meta before calling grid_fn.

    Returns:
        A list of safe autotune configs.
    """
    autotune_layer = _get_autotune_layer(kernel)

    if not is_npu:
        return autotune_layer.configs

    if runtime_meta is None:
        runtime_meta = {}

    safe = []
    for c in autotune_layer.configs:
        meta = {**runtime_meta, **c.kwargs}
        grid_dims = grid_fn(meta)
        coredim = 1
        for d in grid_dims:
            coredim *= d
        if coredim <= max_coredim:
            safe.append(c)

    if not safe:
        # All configs overflow — pick the one with the smallest coreDim and warn.
        all_with_coredim = []
        for c in autotune_layer.configs:
            meta = {**runtime_meta, **c.kwargs}
            grid_dims = grid_fn(meta)
            coredim = 1
            for d in grid_dims:
                coredim *= d
            all_with_coredim.append((coredim, c))
        all_with_coredim.sort(key=lambda x: x[0])
        # Use .fn.name when kernel is a Heuristics wrapper (lacks .name),
        # otherwise kernel.name directly.
        kernel_name = getattr(kernel, 'name', None) or getattr(kernel.fn, 'name', '<unknown>')
        warnings.warn(
            f"All autotune configs for {kernel_name} exceed coreDim limit. "
            f"Using least-overflowing config (coreDim={all_with_coredim[0][0]}). "
            f"This may still crash on NPU.",
            RuntimeWarning,
        )
        safe = [all_with_coredim[0][1]]

    return safe


# ============================================================================
# Softplus implementations (NPU-safe: pure Triton, no NVIDIA PTX)
# ============================================================================


@triton.jit
def softplus(x):
    """Triton softplus: log(1 + exp(x)), with threshold at 20.0 for numerical stability.

    Equivalent to:
        return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)

    NOTE: On NVIDIA GPUs, the PTX inline-assembly version (softplus_nv) is faster,
    but NPU does not support PTX. This pure Triton version is portable across all
    backends including Ascend NPU.
    """
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)


@triton.jit
def softplus2(x):
    """Triton softplus in base-2: log2(1 + exp2(x)), with threshold at 15.0.

    Equivalent to:
        return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)

    NOTE: On NVIDIA GPUs, the PTX inline-assembly version (softplus2_nv) is faster,
    but NPU does not support PTX. This pure Triton version is portable across all
    backends including Ascend NPU.
    """
    return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)