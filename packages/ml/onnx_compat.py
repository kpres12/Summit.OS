"""
Compatibility shim for skl2onnx on Python 3.14+ with onnx >= 1.19.

Python 3.14 stricter protobuf enforcement rejects numpy integer subclasses
(np.uint32, np.int64, np.bool_, etc.) and plain Python booleans in onnx
AttributeProto integer fields:

  TypeError: Field onnx.AttributeProto.ints: Expected an int, got a boolean.

Root cause: skl2onnx's random_forest / gradient_boosting converters pass
numpy-typed arrays for tree node attributes.

Patching onnx.helper.* alone is insufficient because skl2onnx._container
does `from onnx.helper import make_node, make_attribute` — a local binding
that bypasses module-level patches applied afterward. We therefore:

  1. Patch onnx.helper.make_attribute and make_node so any fresh import gets
     the shim automatically.
  2. Retroactively patch skl2onnx.common._container's local bindings if that
     module was already imported (e.g. when skl2onnx is imported before us).

Import this module BEFORE importing skl2onnx in any training script:
    import onnx_compat  # noqa: F401 — must be first
"""
import sys


def _coerce_ints(value):
    """Coerce bool/numpy-int scalar or iterable values to plain Python int."""
    if isinstance(value, (str, bytes)):
        return value
    # scalar bool
    if isinstance(value, bool):
        return int(value)
    # iterable (list, tuple, ndarray, generator, …)
    if hasattr(value, "__iter__"):
        try:
            coerced = []
            for v in value:
                if isinstance(v, bool):
                    coerced.append(int(v))
                elif type(v) is not int and hasattr(v, "__index__"):
                    # numpy int types: np.uint32, np.int64, np.intp, etc.
                    try:
                        coerced.append(int(v))
                    except Exception:
                        coerced.append(v)
                else:
                    coerced.append(v)
            return coerced
        except Exception:
            return value
    return value


try:
    import onnx.helper as _h

    _orig_make_attribute = _h.make_attribute
    _orig_make_node = _h.make_node

    def _patched_make_attribute(key, value, doc_string=None, attr_type=None):
        value = _coerce_ints(value)
        kwargs = {}
        if doc_string is not None:
            kwargs["doc_string"] = doc_string
        if attr_type is not None:
            kwargs["attr_type"] = attr_type
        return _orig_make_attribute(key, value, **kwargs)

    def _patched_make_node(op_type, inputs, outputs, name=None, doc_string=None,
                           domain=None, overload=None, **kwargs):
        clean_kwargs = {k: _coerce_ints(v) for k, v in kwargs.items()}
        extra = {}
        if name is not None:
            extra["name"] = name
        if doc_string is not None:
            extra["doc_string"] = doc_string
        if domain is not None:
            extra["domain"] = domain
        if overload is not None:
            extra["overload"] = overload
        return _orig_make_node(op_type, inputs, outputs, **extra, **clean_kwargs)

    # Patch the onnx.helper module so fresh imports get shims
    _h.make_attribute = _patched_make_attribute
    _h.make_node = _patched_make_node

    # Retroactively patch skl2onnx._container if it's already been imported
    # (it uses `from onnx.helper import make_node, make_attribute` — local bindings)
    _skl_container_name = "skl2onnx.common._container"
    if _skl_container_name in sys.modules:
        _c = sys.modules[_skl_container_name]
        if getattr(_c, "make_attribute", None) is _orig_make_attribute:
            _c.make_attribute = _patched_make_attribute
        if getattr(_c, "make_node", None) is _orig_make_node:
            _c.make_node = _patched_make_node

except ImportError:
    pass  # onnx not installed yet
