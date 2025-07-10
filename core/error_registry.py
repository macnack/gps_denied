from dataclasses import dataclass
from typing import Callable, Dict, Optional
import torch


@dataclass
class ErrorSpec:
    var_name: str
    init_fn: Callable[[int, torch.device], torch.Tensor]
    get_homography: Callable[[torch.Tensor], torch.Tensor]
    id_vals: Optional[torch.Tensor] = None
    dim: int = 1
    reshape: tuple = ((1, 1),)


_REGISTRY: Dict[str, ErrorSpec] = {}


def register(
    var_name: str,
    init_fn: Callable,
    get_homography: Callable,
    id_vals: Optional[torch.Tensor] = None,
    dim: int = 1,
    reshape: tuple = (1, 1),
):
    """Decorator that attaches metadata and puts the fn in the registry."""

    def _decorator(fn):
        _REGISTRY[fn.__name__] = ErrorSpec(
            var_name, init_fn, get_homography, id_vals, dim, reshape)
        fn.spec = _REGISTRY[fn.__name__]  # keep a back-pointer
        return fn

    return _decorator


def get(name: str) -> ErrorSpec:
    return _REGISTRY[name]
