def version():
    from automl import version_

    return version_


 
def __getattr__(name):
    if name in ("__version__", "version_"):
        return version()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
