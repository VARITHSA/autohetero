"""Built-in workloads (DESIGN.md §10).

autohetero.ops.<name> is dispatched like an @auto function once autohetero.enable() is on, or when called
inside an @auto function; otherwise it runs where its input is. RAW holds the plain implementations, which
calibration and bench/ time directly.
"""
import functools

from . import analytics, image, models, science

RAW = {f.__name__: f for f in (
    image.blur, image.resize, image.sobel, models.resnet18, models.mobilenetv2,
    analytics.filter, analytics.sort, analytics.groupby, science.matmul, science.fft,
)}
SPLITTABLE = {"blur", "resize", "sobel", "resnet18", "mobilenetv2", "filter", "matmul", "fft"}  # along dim 0 of arg 0


def _dispatched(name):
    raw = RAW[name]

    @functools.wraps(raw)
    def op(*args, **kwargs):
        from .. import dispatch  # imported per call: dispatch imports calibration, which imports this package

        return dispatch.builtin(name, raw, args, kwargs)

    return op


blur, resize, sobel = _dispatched("blur"), _dispatched("resize"), _dispatched("sobel")
resnet18, mobilenetv2 = _dispatched("resnet18"), _dispatched("mobilenetv2")
filter, sort, groupby = _dispatched("filter"), _dispatched("sort"), _dispatched("groupby")
matmul, fft = _dispatched("matmul"), _dispatched("fft")
