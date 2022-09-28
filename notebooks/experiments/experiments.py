import pathlib
import semantic_version

def resource_dir(exp_id : str) -> pathlib.Path:
    """Returns the resource directory for the given experiment.
    
    The directory will be created if it doesn't exist.
    
    Args:
        exp_id: an experiment identifier, like "1.1.2".
    """
    exp_ver = semantic_version.Version(exp_id)
    exp_dir = pathlib.Path(f"../resources/experiments/exp_{exp_ver.major}_{exp_ver.minor}_{exp_ver.patch}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir