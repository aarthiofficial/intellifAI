# coding: utf-8
# pylint: disable=wrong-import-position

"""
InsightFace: A Face Analysis Toolkit.
"""

from __future__ import absolute_import

__version__ = '0.7.3'

# Dependency check
try:
    import onnxruntime
except ImportError as e:
    raise ImportError(
        "Required dependency 'onnxruntime' is not installed. "
        "Please install it using `pip install onnxruntime`."
    ) from e

# Local submodule imports
from . import model_zoo
from . import utils
from . import app
from . import data

# Optional: expose high-level APIs here
from .app import FaceAnalysis
from .model_zoo import get_model
