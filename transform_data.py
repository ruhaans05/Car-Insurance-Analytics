import importlib.util
import os

# Load the implementation from src/transform_data.py to keep source files in `src/`
_path = os.path.join(os.path.dirname(__file__), "src", "transform_data.py")
spec = importlib.util.spec_from_file_location("_src_transform_data", _path)
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)

# Re-export the public symbols used by tests
pca_feature_importance = _module.pca_feature_importance
EXCLUDE_FROM_PCA = getattr(_module, "EXCLUDE_FROM_PCA", None)

__all__ = ["pca_feature_importance", "EXCLUDE_FROM_PCA"]
