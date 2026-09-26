"""Pytest configuration and backward-compatibility patches."""
from sklearn.impute import SimpleImputer

# Ensure pickled models from scikit-learn 1.7.x execute seamlessly on newer sklearn releases
_orig_transform = SimpleImputer.transform


def _compat_transform(self, X):
    if not hasattr(self, "_fill_dtype"):
        self._fill_dtype = self.statistics_.dtype if hasattr(self, "statistics_") else None
    return _orig_transform(self, X)


SimpleImputer.transform = _compat_transform
