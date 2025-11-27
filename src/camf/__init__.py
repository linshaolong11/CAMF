from importlib.metadata import version

from camf.classifier import CAMFClassifier
from camf.debug_versions import display_debug_info
from camf.regressor import CAMFRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "CAMFClassifier",
    "CAMFRegressor",
    "__version__",
    "display_debug_info",
]
