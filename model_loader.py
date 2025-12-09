import os
import joblib
from typing import List, Any, Dict
import threading
from column_selector import ColumnSelector

# Path to file (assumed in same folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ids_pipeline.pkl")

# Internal cache and lock to ensure single-load
_model_cache = {
    "loaded": False,
    "package": None
}
_cache_lock = threading.Lock()


def _load_model_file() -> Any:
    """
    Load ids_pipeline.pkl using joblib and return the loaded object.
    This function does not modify the cache.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    return loaded


def _normalize_package(loaded_obj) -> Dict[str, Any]:
    package = {}

    if isinstance(loaded_obj, dict):
        package["pipeline"] = (
            loaded_obj.get("pipeline")
            or loaded_obj.get("model")
            or loaded_obj.get("pipeline_obj")
        )

        package["required_raw_features"] = loaded_obj.get("required_raw_features")

        package["label_mapping"] = loaded_obj.get("label_mapping")

        # NEW: keep metadata
        package["model_metadata"] = loaded_obj.get("model_metadata")

    else:
        package["pipeline"] = loaded_obj
        package["required_raw_features"] = None
        package["label_mapping"] = None
        package["model_metadata"] = None

    if package["pipeline"] is None:
        raise ValueError("Loaded model missing 'pipeline'.")

    return package


def _ensure_loaded():
    """
    Thread-safe load into _model_cache if not already loaded.
    """
    if _model_cache["loaded"]:
        return

    with _cache_lock:
        if _model_cache["loaded"]:
            return

        loaded_obj = _load_model_file()
        package = _normalize_package(loaded_obj)

        # Post-process label_mapping: ensure it's a dict with both id->name and name->id
        label_map = package.get("label_mapping")
        if label_map is not None:
            # Accept either {0: 'BENIGN', 1: 'DDoS'} or {'BENIGN':0,...}
            # Normalize to two dicts:
            id_to_name = {}
            name_to_id = {}
            for k, v in label_map.items():
                # keys may be strings of numbers; attempt int
                try:
                    kid = int(k)
                    id_to_name[kid] = v
                    name_to_id[str(v)] = kid
                except Exception:
                    # key is probably a name mapping to id
                    name_to_id[str(k)] = int(v)
                    id_to_name[int(v)] = str(k)
            package["_label_id_to_name"] = id_to_name
            package["_label_name_to_id"] = name_to_id
        else:
            package["_label_id_to_name"] = None
            package["_label_name_to_id"] = None

        _model_cache["package"] = package
        _model_cache["loaded"] = True


def get_pipeline():
    """
    Returns the pipeline package or pipeline object.
    Prefer returning the normalized package (dict) which contains pipeline and metadata.
    """
    _ensure_loaded()
    return _model_cache["package"]


def get_required_features():
    _ensure_loaded()
    package = _model_cache["package"]

    features = package.get("required_raw_features")

    # FIX: Check for None instead of falsy
    if features is None:
        pipeline = package.get("pipeline")
        try:
            if hasattr(pipeline, "named_steps") and "selector" in pipeline.named_steps:
                sel = pipeline.named_steps["selector"]
                if hasattr(sel, "required_columns"):
                    features = list(sel.required_columns)
        except Exception:
            features = None

    if features is None:
        features = []

    return features


def decode_label(prediction):
    """
    Decode predicted class id or name into readable label name.
    Accepts:
        - integer class id (0, 1, ...)
        - string class name (already)
    Returns:
        readable label string
    """
    _ensure_loaded()
    package = _model_cache["package"]

    # If prediction is int-like
    try:
        # If it's numeric (or numeric-string) try to interpret as int
        if isinstance(prediction, (int,)) or (isinstance(prediction, str) and prediction.isdigit()):
            pid = int(prediction)
            id_to_name = package.get("_label_id_to_name")
            if id_to_name and pid in id_to_name:
                return id_to_name[pid]
            # fallback: return str(pid)
            return str(pid)
    except Exception:
        pass

    # If prediction is already a string (likely the class name)
    if isinstance(prediction, str):
        # If mapping exists and prediction is a key in name->id, return the string as-is
        return prediction

    # Last resort: convert to string
    return str(prediction)
