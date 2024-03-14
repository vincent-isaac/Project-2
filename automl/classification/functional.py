import logging
import os
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

import pandas as pd
from joblib.memory import Memory

from automl.classification.oop import ClassificationExperiment
from automl.internal.parallel.parallel_backend import ParallelBackend
from automl.loggers.base_logger import BaseLogger
from automl.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE, TARGET_LIKE
from automl.utils.generic import check_if_global_is_not_none

_EXPERIMENT_CLASS = ClassificationExperiment
_CURRENT_EXPERIMENT: Optional[ClassificationExperiment] = None
_CURRENT_EXPERIMENT_EXCEPTION = (
    "_CURRENT_EXPERIMENT global variable is not set. Please run setup() first."
)
_CURRENT_EXPERIMENT_DECORATOR_DICT = {
    "_CURRENT_EXPERIMENT": _CURRENT_EXPERIMENT_EXCEPTION
}


def setup(
    data: Optional[DATAFRAME_LIKE] = None,
    data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
    target: TARGET_LIKE = -1,
    index: Union[bool, int, str, SEQUENCE_LIKE] = True,
    train_size: float = 0.7,
    test_data: Optional[DATAFRAME_LIKE] = None,
    ordinal_features: Optional[Dict[str, list]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    keep_features: Optional[List[str]] = None,
    preprocess: bool = True,
    create_date_columns: List[str] = ["day", "month", "year"],
    imputation_type: Optional[str] = "simple",
    numeric_imputation: Union[int, float, str] = "mean",
    categorical_imputation: str = "mode",
    iterative_imputation_iters: int = 5,
    numeric_iterative_imputer: Union[str, Any] = "lightgbm",
    categorical_iterative_imputer: Union[str, Any] = "lightgbm",
    text_features_method: str = "tf-idf",
    max_encoding_ohe: int = 25,
    encoding_method: Optional[Any] = None,
    rare_to_value: Optional[float] = None,
    rare_value: str = "rare",
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    low_variance_threshold: Optional[float] = None,
    group_features: Optional[dict] = None,
    drop_groups: bool = False,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    bin_numeric_features: Optional[List[str]] = None,
    remove_outliers: bool = False,
    outliers_method: str = "iforest",
    outliers_threshold: float = 0.05,
    fix_imbalance: bool = False,
    fix_imbalance_method: Union[str, Any] = "SMOTE",
    transformation: bool = False,
    transformation_method: str = "yeo-johnson",
    normalize: bool = False,
    normalize_method: str = "zscore",
    pca: bool = False,
    pca_method: str = "linear",
    pca_components: Optional[Union[int, float, str]] = None,
    feature_selection: bool = False,
    feature_selection_method: str = "classic",
    feature_selection_estimator: Union[str, Any] = "lightgbm",
    n_features_to_select: Union[int, float] = 0.2,
    custom_pipeline: Optional[Any] = None,
    custom_pipeline_position: int = -1,
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = True,
    fold_strategy: Union[str, Any] = "stratifiedkfold",
    fold: int = 10,
    fold_shuffle: bool = False,
    fold_groups: Optional[Union[str, pd.DataFrame]] = None,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,
    html: bool = True,
    session_id: Optional[int] = None,
    system_log: Union[bool, str, logging.Logger] = True,
    log_experiment: Union[bool, str, BaseLogger, List[Union[str, BaseLogger]]] = False,
    experiment_name: Optional[str] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    verbose: bool = True,
    memory: Union[bool, str, Memory] = True,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
):

    exp = _EXPERIMENT_CLASS()
    set_current_experiment(exp)
    return exp.setup(
        data=data,
        data_func=data_func,
        target=target,
        index=index,
        train_size=train_size,
        test_data=test_data,
        ordinal_features=ordinal_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        text_features=text_features,
        ignore_features=ignore_features,
        keep_features=keep_features,
        preprocess=preprocess,
        create_date_columns=create_date_columns,
        imputation_type=imputation_type,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        iterative_imputation_iters=iterative_imputation_iters,
        numeric_iterative_imputer=numeric_iterative_imputer,
        categorical_iterative_imputer=categorical_iterative_imputer,
        text_features_method=text_features_method,
        max_encoding_ohe=max_encoding_ohe,
        encoding_method=encoding_method,
        rare_to_value=rare_to_value,
        rare_value=rare_value,
        polynomial_features=polynomial_features,
        polynomial_degree=polynomial_degree,
        low_variance_threshold=low_variance_threshold,
        group_features=group_features,
        drop_groups=drop_groups,
        remove_multicollinearity=remove_multicollinearity,
        multicollinearity_threshold=multicollinearity_threshold,
        bin_numeric_features=bin_numeric_features,
        remove_outliers=remove_outliers,
        outliers_method=outliers_method,
        outliers_threshold=outliers_threshold,
        fix_imbalance=fix_imbalance,
        fix_imbalance_method=fix_imbalance_method,
        transformation=transformation,
        transformation_method=transformation_method,
        normalize=normalize,
        normalize_method=normalize_method,
        pca=pca,
        pca_method=pca_method,
        pca_components=pca_components,
        feature_selection=feature_selection,
        feature_selection_method=feature_selection_method,
        feature_selection_estimator=feature_selection_estimator,
        n_features_to_select=n_features_to_select,
        custom_pipeline=custom_pipeline,
        custom_pipeline_position=custom_pipeline_position,
        data_split_shuffle=data_split_shuffle,
        data_split_stratify=data_split_stratify,
        fold_strategy=fold_strategy,
        fold=fold,
        fold_shuffle=fold_shuffle,
        fold_groups=fold_groups,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
        html=html,
        session_id=session_id,
        system_log=system_log,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
        experiment_custom_tags=experiment_custom_tags,
        log_plots=log_plots,
        log_profile=log_profile,
        log_data=log_data,
        verbose=verbose,
        memory=memory,
        profile=profile,
        profile_kwargs=profile_kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def compare_models(
    include: Optional[List[Union[str, Any]]] = None,
    exclude: Optional[List[str]] = None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "Accuracy",
    n_select: int = 1,
    budget_time: Optional[float] = None,
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    engine: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    parallel: Optional[ParallelBackend] = None,
) -> Union[Any, List[Any]]:
    

    return _CURRENT_EXPERIMENT.compare_models(
        include=include,
        exclude=exclude,
        fold=fold,
        round=round,
        cross_validation=cross_validation,
        sort=sort,
        n_select=n_select,
        budget_time=budget_time,
        turbo=turbo,
        errors=errors,
        fit_kwargs=fit_kwargs,
        groups=groups,
        experiment_custom_tags=experiment_custom_tags,
        probability_threshold=probability_threshold,
        engine=engine,
        verbose=verbose,
        parallel=parallel,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_allowed_engines(estimator: str) -> Optional[str]:
    
    return _CURRENT_EXPERIMENT.get_allowed_engines(estimator=estimator)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_engine(estimator: str) -> Optional[str]:

    return _CURRENT_EXPERIMENT.get_engine(estimator=estimator)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_model(
    estimator: Union[str, Any],
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    engine: Optional[str] = None,
    verbose: bool = True,
    return_train_score: bool = False,
    **kwargs,
) -> Any:

    return _CURRENT_EXPERIMENT.create_model(
        estimator=estimator,
        fold=fold,
        round=round,
        cross_validation=cross_validation,
        fit_kwargs=fit_kwargs,
        groups=groups,
        probability_threshold=probability_threshold,
        experiment_custom_tags=experiment_custom_tags,
        engine=engine,
        verbose=verbose,
        return_train_score=return_train_score,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def tune_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    n_iter: int = 10,
    custom_grid: Optional[Union[Dict[str, list], Any]] = None,
    optimize: str = "Accuracy",
    custom_scorer=None,
    search_library: str = "scikit-learn",
    search_algorithm: Optional[str] = None,
    early_stopping: Any = False,
    early_stopping_max_iters: int = 10,
    choose_better: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    return_tuner: bool = False,
    verbose: bool = True,
    tuner_verbose: Union[int, bool] = True,
    return_train_score: bool = False,
    **kwargs,
) -> Any:
    

    return _CURRENT_EXPERIMENT.tune_model(
        estimator=estimator,
        fold=fold,
        round=round,
        n_iter=n_iter,
        custom_grid=custom_grid,
        optimize=optimize,
        custom_scorer=custom_scorer,
        search_library=search_library,
        search_algorithm=search_algorithm,
        early_stopping=early_stopping,
        early_stopping_max_iters=early_stopping_max_iters,
        choose_better=choose_better,
        fit_kwargs=fit_kwargs,
        groups=groups,
        return_tuner=return_tuner,
        verbose=verbose,
        tuner_verbose=tuner_verbose,
        return_train_score=return_train_score,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def ensemble_model(
    estimator,
    method: str = "Bagging",
    fold: Optional[Union[int, Any]] = None,
    n_estimators: int = 10,
    round: int = 4,
    choose_better: bool = False,
    optimize: str = "Accuracy",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    verbose: bool = True,
    return_train_score: bool = False,
) -> Any:
   

    return _CURRENT_EXPERIMENT.ensemble_model(
        estimator=estimator,
        method=method,
        fold=fold,
        n_estimators=n_estimators,
        round=round,
        choose_better=choose_better,
        optimize=optimize,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
        probability_threshold=probability_threshold,
        return_train_score=return_train_score,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def blend_models(
    estimator_list: list,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    choose_better: bool = False,
    optimize: str = "Accuracy",
    method: str = "auto",
    weights: Optional[List[float]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    verbose: bool = True,
    return_train_score: bool = False,
) -> Any:
    
    return _CURRENT_EXPERIMENT.blend_models(
        estimator_list=estimator_list,
        fold=fold,
        round=round,
        choose_better=choose_better,
        optimize=optimize,
        method=method,
        weights=weights,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
        probability_threshold=probability_threshold,
        return_train_score=return_train_score,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def stack_models(
    estimator_list: list,
    meta_model=None,
    meta_model_fold: Optional[Union[int, Any]] = 5,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    method: str = "auto",
    restack: bool = False,
    choose_better: bool = False,
    optimize: str = "Accuracy",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    verbose: bool = True,
    return_train_score: bool = False,
) -> Any:
    

    return _CURRENT_EXPERIMENT.stack_models(
        estimator_list=estimator_list,
        meta_model=meta_model,
        meta_model_fold=meta_model_fold,
        fold=fold,
        round=round,
        method=method,
        restack=restack,
        choose_better=choose_better,
        optimize=optimize,
        fit_kwargs=fit_kwargs,
        groups=groups,
        probability_threshold=probability_threshold,
        verbose=verbose,
        return_train_score=return_train_score,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def plot_model(
    estimator,
    plot: str = "auc",
    scale: float = 1,
    save: bool = False,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    plot_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display_format: Optional[str] = None,
) -> Optional[str]:
   

    return _CURRENT_EXPERIMENT.plot_model(
        estimator=estimator,
        plot=plot,
        scale=scale,
        save=save,
        fold=fold,
        fit_kwargs=fit_kwargs,
        plot_kwargs=plot_kwargs,
        groups=groups,
        verbose=verbose,
        display_format=display_format,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def evaluate_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    plot_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
):
   

    return _CURRENT_EXPERIMENT.evaluate_model(
        estimator=estimator,
        fold=fold,
        fit_kwargs=fit_kwargs,
        plot_kwargs=plot_kwargs,
        groups=groups,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def interpret_model(
    estimator,
    plot: str = "summary",
    feature: Optional[str] = None,
    observation: Optional[int] = None,
    use_train_data: bool = False,
    X_new_sample: Optional[pd.DataFrame] = None,
    y_new_sample: Optional[pd.DataFrame] = None,  # add for pfi explainer
    save: Union[str, bool] = False,
    **kwargs,
):
   

    return _CURRENT_EXPERIMENT.interpret_model(
        estimator=estimator,
        plot=plot,
        feature=feature,
        observation=observation,
        use_train_data=use_train_data,
        X_new_sample=X_new_sample,
        y_new_sample=y_new_sample,
        save=save,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def calibrate_model(
    estimator,
    method: str = "sigmoid",
    calibrate_fold: Optional[Union[int, Any]] = 5,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    return_train_score: bool = False,
) -> Any:
    

    return _CURRENT_EXPERIMENT.calibrate_model(
        estimator=estimator,
        method=method,
        calibrate_fold=calibrate_fold,
        fold=fold,
        round=round,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
        return_train_score=return_train_score,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def optimize_threshold(
    estimator,
    optimize: str = "Accuracy",
    return_data: bool = False,
    plot_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **shgo_kwargs,
):
    

    return _CURRENT_EXPERIMENT.optimize_threshold(
        estimator=estimator,
        optimize=optimize,
        return_data=return_data,
        plot_kwargs=plot_kwargs,
        verbose=verbose,
        **shgo_kwargs,
    )


# not using check_if_global_is_not_none on purpose
def predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    probability_threshold: Optional[float] = None,
    encoded_labels: bool = False,
    raw_score: bool = False,
    round: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
   
    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.predict_model(
        estimator=estimator,
        data=data,
        probability_threshold=probability_threshold,
        encoded_labels=encoded_labels,
        raw_score=raw_score,
        round=round,
        verbose=verbose,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def finalize_model(
    estimator,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    model_only: bool = False,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
) -> Any:
   

    return _CURRENT_EXPERIMENT.finalize_model(
        estimator=estimator,
        fit_kwargs=fit_kwargs,
        groups=groups,
        model_only=model_only,
        experiment_custom_tags=experiment_custom_tags,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def deploy_model(
    model,
    model_name: str,
    authentication: dict,
    platform: str = "aws",
):

    return _CURRENT_EXPERIMENT.deploy_model(
        model=model,
        model_name=model_name,
        authentication=authentication,
        platform=platform,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def save_model(
    model, model_name: str, model_only: bool = False, verbose: bool = True, **kwargs
):

    return _CURRENT_EXPERIMENT.save_model(
        model=model,
        model_name=model_name,
        model_only=model_only,
        verbose=verbose,
        **kwargs,
    )


# not using check_if_global_is_not_none on purpose
def load_model(
    model_name: str,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):
    

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.load_model(
        model_name=model_name,
        platform=platform,
        authentication=authentication,
        verbose=verbose,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def automl(
    optimize: str = "Accuracy",
    use_holdout: bool = False,
    turbo: bool = True,
    return_train_score: bool = False,
) -> Any:
    return _CURRENT_EXPERIMENT.automl(
        optimize=optimize,
        use_holdout=use_holdout,
        turbo=turbo,
        return_train_score=return_train_score,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def pull(pop: bool = False) -> pd.DataFrame:
    return _CURRENT_EXPERIMENT.pull(pop=pop)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def models(
    type: Optional[str] = None,
    internal: bool = False,
    raise_errors: bool = True,
) -> pd.DataFrame:
    return _CURRENT_EXPERIMENT.models(
        type=type, internal=internal, raise_errors=raise_errors
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_metrics(
    reset: bool = False,
    include_custom: bool = True,
    raise_errors: bool = True,
) -> pd.DataFrame:
   

    return _CURRENT_EXPERIMENT.get_metrics(
        reset=reset,
        include_custom=include_custom,
        raise_errors=raise_errors,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def add_metric(
    id: str,
    name: str,
    score_func: type,
    target: str = "pred",
    greater_is_better: bool = True,
    multiclass: bool = True,
    **kwargs,
) -> pd.Series:
   

    return _CURRENT_EXPERIMENT.add_metric(
        id=id,
        name=name,
        score_func=score_func,
        target=target,
        greater_is_better=greater_is_better,
        multiclass=multiclass,
        **kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def remove_metric(name_or_id: str):
    
    return _CURRENT_EXPERIMENT.remove_metric(name_or_id=name_or_id)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_logs(experiment_name: Optional[str] = None, save: bool = False) -> pd.DataFrame:
   

    return _CURRENT_EXPERIMENT.get_logs(experiment_name=experiment_name, save=save)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_config(variable: Optional[str] = None):

    return _CURRENT_EXPERIMENT.get_config(variable=variable)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def set_config(variable: str, value):

    return _CURRENT_EXPERIMENT.set_config(variable=variable, value=value)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def save_experiment(
    path_or_file: Union[str, os.PathLike, BinaryIO], **cloudpickle_kwargs
) -> None:

    return _CURRENT_EXPERIMENT.save_experiment(
        path_or_file=path_or_file, **cloudpickle_kwargs
    )


def load_experiment(
    path_or_file: Union[str, os.PathLike, BinaryIO],
    data: Optional[DATAFRAME_LIKE] = None,
    data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
    test_data: Optional[DATAFRAME_LIKE] = None,
    preprocess_data: bool = True,
    **cloudpickle_kwargs,
) -> ClassificationExperiment:
    
    exp = _EXPERIMENT_CLASS.load_experiment(
        path_or_file=path_or_file,
        data=data,
        data_func=data_func,
        test_data=test_data,
        preprocess_data=preprocess_data,
        **cloudpickle_kwargs,
    )
    set_current_experiment(exp)
    return exp


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_leaderboard(
    finalize_models: bool = False,
    model_only: bool = False,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _CURRENT_EXPERIMENT.get_leaderboard(
        finalize_models=finalize_models,
        model_only=model_only,
        fit_kwargs=fit_kwargs,
        groups=groups,
        verbose=verbose,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def dashboard(
    estimator,
    display_format: str = "dash",
    dashboard_kwargs: Optional[Dict[str, Any]] = None,
    run_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):

    return _CURRENT_EXPERIMENT.dashboard(
        estimator, display_format, dashboard_kwargs, run_kwargs, **kwargs
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def convert_model(estimator, language: str = "python") -> str:
   
    return _CURRENT_EXPERIMENT.convert_model(estimator, language)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def check_fairness(estimator, sensitive_features: list, plot_kwargs: dict = {}):
    
    return _CURRENT_EXPERIMENT.check_fairness(
        estimator=estimator,
        sensitive_features=sensitive_features,
        plot_kwargs=plot_kwargs,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_api(
    estimator, api_name: str, host: str = "127.0.0.1", port: int = 8000
) -> None:
    
    return _CURRENT_EXPERIMENT.create_api(
        estimator=estimator, api_name=api_name, host=host, port=port
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_docker(
    api_name: str, base_image: str = "python:3.8-slim", expose_port: int = 8000
) -> None:
    
    return _CURRENT_EXPERIMENT.create_docker(
        api_name=api_name, base_image=base_image, expose_port=expose_port
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def create_app(estimator, app_kwargs: Optional[dict] = None) -> None:
    
    return _CURRENT_EXPERIMENT.create_app(estimator=estimator, app_kwargs=app_kwargs)


def check_drift(
    reference_data: Optional[pd.DataFrame] = None,
    current_data: Optional[pd.DataFrame] = None,
    target: Optional[str] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    filename: Optional[str] = None,
) -> str:
    
    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.check_drift(
        reference_data=reference_data,
        current_data=current_data,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        filename=filename,
    )


def set_current_experiment(experiment: ClassificationExperiment) -> None:
    
    global _CURRENT_EXPERIMENT

    if not isinstance(experiment, ClassificationExperiment):
        raise TypeError(
            f"experiment must be a automl ClassificationExperiment object, got {type(experiment)}."
        )
    _CURRENT_EXPERIMENT = experiment


def get_current_experiment() -> ClassificationExperiment:
    
    return _CURRENT_EXPERIMENT
