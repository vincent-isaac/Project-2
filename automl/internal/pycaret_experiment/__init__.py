class TimeSeriesExperiment:
    def __init__(self) -> None:
        msg = (
            "\nTimeSeriesExperiment class has been deprecated. "
            "Please import the following instead.\n"
            ">>> from automl.time_series import TSForecastingExperiment"
        )
        raise DeprecationWarning(msg)
