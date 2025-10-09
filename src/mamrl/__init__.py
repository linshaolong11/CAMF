from importlib.metadata import version

from mamrl.classifier import MAMRLClassifier
from mamrl.debug_versions import display_debug_info
from mamrl.regressor import MAMRLRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "MAMRLClassifier",
    "MAMRLRegressor",
    "__version__",
    "display_debug_info",
]
