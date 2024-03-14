import datetime
import gc
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import numpy as np  
import pandas as pd
import plotly.express as px
import sklearn
from joblib.memory import Memory
from scipy.optimize import shgo

from automl.containers.metrics.classification import get_all_metric_containers
from automl.containers.models.classification import (
    ALL_ALLOWED_ENGINES,
    get_all_model_containers,
    get_container_default_engines,
)
from automl.internal.display import CommonDisplay
from automl.internal.logging import get_logger
from automl.internal.meta_estimators import (
    CustomProbabilityThresholdClassifier,
    get_estimator_from_meta_estimator,
)
from automl.internal.parallel.parallel_backend import ParallelBackend
from automl.internal.pipeline import Pipeline as InternalPipeline
from automl.internal.preprocess.preprocessor import Preprocessor
from automl.internal.pycaret_experiment.non_ts_supervised_experiment import (
    _NonTSSupervisedExperiment,
)
from automl.internal.validation import is_sklearn_cv_generator
from automl.loggers.base_logger import BaseLogger
from automl.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE, TARGET_LIKE
from automl.utils.generic import (
    MLUsecase,
    get_classification_task,
    get_label_encoder,
    highlight_setup,
)

LOGGER = get_logger()


class ClassificationExperiment(_NonTSSupervisedExperiment, Preprocessor):
    _create_app_predict_kwargs = {"raw_score": True}

    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLASSIFICATION
        self.exp_name_log = "clf-default-name"
        self._variable_keys = self._variable_keys.union(
            {"fix_imbalance", "is_multiclass"}
        )
        self._available_plots = {
            "pipeline": "Pipeline Plot",
            "parameter": "Hyperparameters",
            "auc": "AUC",
            "confusion_matrix": "Confusion Matrix",
            "threshold": "Threshold",
            "pr": "Precision Recall",
            "error": "Prediction Error",
            "class_report": "Class Report",
            "rfe": "Feature Selection",
            "learning": "Learning Curve",
            "manifold": "Manifold Learning",
            "calibration": "Calibration Curve",
            "vc": "Validation Curve",
            "dimension": "Dimensions",
            "feature": "Feature Importance",
            "feature_all": "Feature Importance (All)",
            "boundary": "Decision Boundary",
            "lift": "Lift Chart",
            "gain": "Gain Chart",
            "tree": "Decision Tree",
            "ks": "KS Statistic Plot",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = get_all_model_containers(self, raise_errors=raise_errors)
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return get_all_metric_containers(self.variables, raise_errors=raise_errors)

    @property
    def is_multiclass(self) -> bool:
        """
        Method to check if the problem is multiclass.
        """
        # Cache the result to avoid calculating it every time
        if hasattr(self, "_is_multiclass"):
            return self._is_multiclass
        if getattr(self, "y", None) is None:
            return False
        try:
            self._is_multiclass = self.y.value_counts().count() > 2
        except Exception:
            self._is_multiclass = False
        return self._is_multiclass

    def _get_default_plots_to_log(self) -> List[str]:
        return ["auc", "confusion_matrix", "feature"]

    def setup(
        self,
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
        numeric_imputation: str = "mean",
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
        log_experiment: Union[
            bool, str, BaseLogger, List[Union[str, BaseLogger]]
        ] = False,
        experiment_name: Optional[str] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        engine: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        memory: Union[bool, str, Memory] = True,
        profile: bool = False,
        profile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        

        self._register_setup_params(dict(locals()))

        if (data is None and data_func is None) or (
            data is not None and data_func is not None
        ):
            raise ValueError("One and only one of data and data_func must be set")

        # No extra code above this line
        # Setup initialization ===================================== >>

        runtime_start = time.time()

        # Configuration
        sklearn.set_config(print_changed_only=False)

        self.all_allowed_engines = ALL_ALLOWED_ENGINES

        # Define parameter attrs
        self.fold_shuffle_param = fold_shuffle
        self.fold_groups_param = fold_groups

        self._initialize_setup(
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            memory=memory,
            verbose=verbose,
        )

        # Prepare experiment specific params ======================= >>

        self.log_plots_param = log_plots
        if self.log_plots_param is True:
            self.log_plots_param = self._get_default_plots_to_log()
        elif isinstance(self.log_plots_param, list):
            for i in self.log_plots_param:
                if i not in self._available_plots:
                    raise ValueError(
                        f"Invalid value for log_plots '{i}'. Possible values "
                        f"are: {', '.join(self._available_plots.keys())}."
                    )

        # Set up data ============================================== >>
        if data_func is not None:
            data = data_func()

        self.data = self._prepare_dataset(data, target)
        self.target_param = self.data.columns[-1]
        self.index = index
        self.data_split_stratify = data_split_stratify
        self.data_split_shuffle = data_split_shuffle

        self._prepare_folds(
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
            data_split_shuffle=data_split_shuffle,
        )

        self._prepare_train_test(
            train_size=train_size,
            test_data=test_data,
            data_split_stratify=data_split_stratify,
            data_split_shuffle=data_split_shuffle,
        )

        self._prepare_column_types(
            ordinal_features=ordinal_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
        )

        self._set_exp_model_engines(
            container_default_engines=get_container_default_engines(),
            engine=engine,
        )

        # Preprocessing ============================================ >>

        # Initialize empty pipeline
        self.pipeline = InternalPipeline(
            steps=[("placeholder", None)],
            memory=self.memory,
        )

        if preprocess:
            self.logger.info("Preparing preprocessing pipeline...")

            # Encode the target column
            y_unique = self.y.unique()
            if sorted(list(y_unique)) != list(range(len(y_unique))):
                self._encode_target_column()

            # Convert date feature to numerical values
            if self._fxs["Date"]:
                self._date_feature_engineering(create_date_columns)

            # Impute missing values
            if imputation_type == "simple":
                self._simple_imputation(numeric_imputation, categorical_imputation)
            elif imputation_type == "iterative":
                self._iterative_imputation(
                    iterative_imputation_iters=iterative_imputation_iters,
                    numeric_iterative_imputer=numeric_iterative_imputer,
                    categorical_iterative_imputer=categorical_iterative_imputer,
                )
            elif imputation_type is not None:
                raise ValueError(
                    "Invalid value for the imputation_type parameter, got "
                    f"{imputation_type}. Possible values are: simple, iterative."
                )

            # Convert text features to meaningful vectors
            if self._fxs["Text"]:
                self._text_embedding(text_features_method)

            # Encode non-numerical features
            if self._fxs["Ordinal"] or self._fxs["Categorical"]:
                self._encoding(
                    max_encoding_ohe=max_encoding_ohe,
                    encoding_method=encoding_method,
                    rare_to_value=rare_to_value,
                    rare_value=rare_value,
                )

            # Create polynomial features from the existing ones
            if polynomial_features:
                self._polynomial_features(polynomial_degree)

            # Drop features with too low variance
            if low_variance_threshold is not None:
                self._low_variance(low_variance_threshold)

            # Get statistical properties of a group of features
            if group_features:
                self._group_features(group_features, drop_groups)

            # Drop features that are collinear with other features
            if remove_multicollinearity:
                self._remove_multicollinearity(multicollinearity_threshold)

            # Bin numerical features to 5 clusters
            if bin_numeric_features:
                self._bin_numerical_features(bin_numeric_features)

            # Remove outliers from the dataset
            if remove_outliers:
                self._remove_outliers(outliers_method, outliers_threshold)

            # Balance the classes in the target column
            if fix_imbalance:
                self._balance(fix_imbalance_method, session_id)

            # Power transform the data to be more Gaussian-like
            if transformation:
                self._transformation(transformation_method)

            # Scale the features
            if normalize:
                self._normalization(normalize_method)

            # Apply Principal Component Analysis
            if pca:
                self._pca(pca_method, pca_components)

            # Select relevant features
            if feature_selection:
                self._feature_selection(
                    feature_selection_method=feature_selection_method,
                    feature_selection_estimator=feature_selection_estimator,
                    n_features_to_select=n_features_to_select,
                )

        # Add custom transformers to the pipeline
        if custom_pipeline:
            self._add_custom_pipeline(custom_pipeline, custom_pipeline_position)

        # Remove weird characters from column names
        # This has to be done right before the estimator, as modifying column
        # names early messes self._fxs up
        if any(re.search("[^A-Za-z0-9_]", col) for col in self.dataset):
            self._clean_column_names()

        # Remove placeholder step
        if ("placeholder", None) in self.pipeline.steps and len(self.pipeline) > 1:
            self.pipeline.steps.remove(("placeholder", None))

        self.pipeline.fit(self.X_train, self.y_train)

        self.logger.info("Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self.pipeline}")

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")

        container = []
        container.append(["Session id", self.seed])
        container.append(["Target", self.target_param])
        container.append(["Target type", get_classification_task(self.y)])
        le = get_label_encoder(self.pipeline)
        if le:
            mapping = {str(v): i for i, v in enumerate(le.classes_)}
            container.append(
                ["Target mapping", ", ".join([f"{k}: {v}" for k, v in mapping.items()])]
            )
        container.append(["Original data shape", self.data.shape])
        container.append(["Transformed data shape", self.dataset_transformed.shape])
        container.append(["Transformed train set shape", self.train_transformed.shape])
        container.append(["Transformed test set shape", self.test_transformed.shape])
        for fx, cols in self._fxs.items():
            if len(cols) > 0:
                container.append([f"{fx} features", len(cols)])
        if self.data.isna().sum().sum():
            n_nans = 100 * self.data.isna().any(axis=1).sum() / len(self.data)
            container.append(["Rows with missing values", f"{round(n_nans, 1)}%"])
        if preprocess:
            container.append(["Preprocess", preprocess])
            container.append(["Imputation type", imputation_type])
            if imputation_type == "simple":
                container.append(["Numeric imputation", numeric_imputation])
                container.append(["Categorical imputation", categorical_imputation])
            elif imputation_type == "iterative":
                if isinstance(numeric_iterative_imputer, str):
                    num_imputer = numeric_iterative_imputer
                else:
                    num_imputer = numeric_iterative_imputer.__class__.__name__

                if isinstance(categorical_iterative_imputer, str):
                    cat_imputer = categorical_iterative_imputer
                else:
                    cat_imputer = categorical_iterative_imputer.__class__.__name__

                container.append(
                    ["Iterative imputation iterations", iterative_imputation_iters]
                )
                container.append(["Numeric iterative imputer", num_imputer])
                container.append(["Categorical iterative imputer", cat_imputer])
            if self._fxs["Text"]:
                container.append(
                    ["Text features embedding method", text_features_method]
                )
            if self._fxs["Categorical"]:
                container.append(["Maximum one-hot encoding", max_encoding_ohe])
                container.append(["Encoding method", encoding_method])
            if polynomial_features:
                container.append(["Polynomial features", polynomial_features])
                container.append(["Polynomial degree", polynomial_degree])
            if low_variance_threshold is not None:
                container.append(["Low variance threshold", low_variance_threshold])
            if remove_multicollinearity:
                container.append(["Remove multicollinearity", remove_multicollinearity])
                container.append(
                    ["Multicollinearity threshold", multicollinearity_threshold]
                )
            if remove_outliers:
                container.append(["Remove outliers", remove_outliers])
                container.append(["Outliers threshold", outliers_threshold])
            if fix_imbalance:
                container.append(["Fix imbalance", fix_imbalance])
                container.append(["Fix imbalance method", fix_imbalance_method])
            if transformation:
                container.append(["Transformation", transformation])
                container.append(["Transformation method", transformation_method])
            if normalize:
                container.append(["Normalize", normalize])
                container.append(["Normalize method", normalize_method])
            if pca:
                container.append(["PCA", pca])
                container.append(["PCA method", pca_method])
                container.append(["PCA components", pca_components])
            if feature_selection:
                container.append(["Feature selection", feature_selection])
                container.append(["Feature selection method", feature_selection_method])
                container.append(
                    ["Feature selection estimator", feature_selection_estimator]
                )
                container.append(["Number of features selected", n_features_to_select])
            if custom_pipeline:
                container.append(["Custom pipeline", "Yes"])
            container.append(["Fold Generator", self.fold_generator.__class__.__name__])
            container.append(["Fold Number", fold])
            container.append(["CPU Jobs", self.n_jobs_param])
            container.append(["Use GPU", self.gpu_param])
            container.append(["Log Experiment", self.logging_param])
            container.append(["Experiment Name", self.exp_name_log])
            container.append(["USI", self.USI])

        self._display_container = [
            pd.DataFrame(container, columns=["Description", "Value"])
        ]
        self.logger.info(f"Setup _display_container: {self._display_container[0]}")
        display = CommonDisplay(
            verbose=self.verbose,
            html_param=self.html_param,
        )
        if self.verbose:
            pd.set_option("display.max_rows", 100)
            display.display(self._display_container[0].style.apply(highlight_setup))
            pd.reset_option("display.max_rows")  # Reset option

        # Wrap-up ================================================== >>

        # Create a profile report
        self._profile(profile, profile_kwargs)

        # Define models and metrics
        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        runtime = np.array(time.time() - runtime_start).round(2)
        self._set_up_logging(
            runtime,
            log_data,
            log_profile,
            experiment_custom_tags=experiment_custom_tags,
        )

        self._setup_ran = True
        self.logger.info(f"setup() successfully completed in {runtime}s...............")

        return self

    def compare_models(
        self,
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
        
        caller_params = dict(locals())

        # No extra code above this line

        if engine is not None:
            # Save current engines, then set to user specified options
            initial_model_engines = self.exp_model_engines.copy()
            for estimator, eng in engine.items():
                self._set_engine(estimator=estimator, engine=eng, severity="error")

        try:
            return_values = super().compare_models(
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
                verbose=verbose,
                probability_threshold=probability_threshold,
                parallel=parallel,
                caller_params=caller_params,
            )
        finally:
            if engine is not None:
                # Reset the models back to the default engines
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engine=initial_model_engines,
                )

        return return_values

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        engine: Optional[str] = None,
        verbose: bool = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:
        
        if engine is not None:
            # Save current engines, then set to user specified options
            initial_default_model_engines = self.exp_model_engines.copy()
            self._set_engine(estimator=estimator, engine=engine, severity="error")

        try:
            return_values = super().create_model(
                estimator=estimator,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                fit_kwargs=fit_kwargs,
                groups=groups,
                verbose=verbose,
                experiment_custom_tags=experiment_custom_tags,
                probability_threshold=probability_threshold,
                return_train_score=return_train_score,
                **kwargs,
            )
        finally:
            if engine is not None:
                # Reset the models back to the default engines
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engine=initial_default_model_engines,
                )

        return return_values

    def tune_model(
        self,
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
        

        return super().tune_model(
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

    def ensemble_model(
        self,
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
        

        return super().ensemble_model(
            estimator=estimator,
            method=method,
            fold=fold,
            n_estimators=n_estimators,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            probability_threshold=probability_threshold,
            verbose=verbose,
            return_train_score=return_train_score,
        )

    def blend_models(
        self,
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
        

        return super().blend_models(
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

    def stack_models(
        self,
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
        
        return super().stack_models(
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
            verbose=verbose,
            probability_threshold=probability_threshold,
            return_train_score=return_train_score,
        )

    def plot_model(
        self,
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
        

        return super().plot_model(
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

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
    ):
        

        return super().evaluate_model(
            estimator=estimator,
            fold=fold,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
        )

    def interpret_model(
        self,
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
        

        return super().interpret_model(
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

    def calibrate_model(
        self,
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
        
        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing calibrate_model()")
        self.logger.info(f"calibrate_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        
        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        self.logger.info("Preloading libraries")

        # pre-load libraries

        self.logger.info("Preparing display monitor")

        progress_args = {"max": 2 + 4}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        np.random.seed(self.seed)

        probability_threshold = None
        if isinstance(estimator, CustomProbabilityThresholdClassifier):
            probability_threshold = estimator.probability_threshold
            estimator = get_estimator_from_meta_estimator(estimator)

        self.logger.info("Getting model name")

        full_name = self._get_model_name(estimator)

        self.logger.info(f"Base model : {full_name}")

        display.update_monitor(2, full_name)

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Selecting Estimator")

        """
        MONITOR UPDATE ENDS
        """

        # calibrating estimator

        self.logger.info("Importing untrained CalibratedClassifierCV")

        calibrated_model_definition = self._all_models_internal["CalibratedCV"]
        model = calibrated_model_definition.class_def(
            estimator=estimator,
            method=method,
            cv=calibrate_fold,
            **calibrated_model_definition.args,
        )

        display.move_progress()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        model, model_fit_time = self._create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            groups=groups,
            probability_threshold=probability_threshold,
            return_train_score=return_train_score,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        model_results = model_results.round(round)

        display.move_progress()

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:
            avgs_dict_log = {
                k: v
                for k, v in model_results.loc[
                    self._get_return_train_score_indices_for_logging(
                        return_train_score=return_train_score
                    )
                ].items()
            }
            self.logging_param.log_model_comparison(
                model_results, f"calibrate_models_{self._get_model_name(model)}"
            )

            self._log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="calibrate_models",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                display=display,
            )

        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "calibrate_model() successfully completed......................................"
        )

        gc.collect()
        return model

    def optimize_threshold(
        self,
        estimator,
        optimize: str = "Accuracy",
        return_data: bool = False,
        plot_kwargs: Optional[dict] = None,
        verbose: bool = True,
        **shgo_kwargs,
    ):
        

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing optimize_threshold()")
        self.logger.info(f"optimize_threshold({function_params_str})")

        self.logger.info("Importing libraries")

        # import libraries

        np.random.seed(self.seed)

        """
        ERROR HANDLING STARTS HERE
        """

        self.logger.info("Checking exceptions")

        # exception 1 for multi-class
        if self.is_multiclass:
            raise TypeError(
                "optimize_threshold() cannot be used when target is multi-class."
            )

        # check predict_proba value
        if type(estimator) is not list:
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(
                    "Estimator doesn't support predict_proba function and cannot be used in optimize_threshold()."
                )

        if "func" in shgo_kwargs or "bounds" in shgo_kwargs or "args" in shgo_kwargs:
            raise ValueError("shgo_kwargs cannot contain 'func', 'bounds' or 'args'.")

        shgo_kwargs.setdefault("sampling_method", "sobol")
        shgo_kwargs.setdefault("options", {})
        shgo_kwargs.setdefault("minimizer_kwargs", {})
        shgo_kwargs["minimizer_kwargs"].setdefault("options", {})
        shgo_kwargs["minimizer_kwargs"]["options"].setdefault("ftol", 1e-3)
        shgo_kwargs.setdefault("n", 8)
        shgo_kwargs["options"].setdefault("maxiter", 4)
        shgo_kwargs["options"].setdefault("f_tol", 1e-3)

        # checking optimize parameter
        optimize = self._get_metric_by_name_or_id(optimize)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )
        direction = -1 if optimize.greater_is_better else 1
        optimize = optimize.display_name

        """
        ERROR HANDLING ENDS HERE
        """

        self.logger.info("defining variables")
        # get estimator name
        model_name = self._get_model_name(estimator)

        # defines empty list
        results_df = []

        self.logger.info("starting optimization")

        def objective(x, *args):
            probability_threshold = x[0]
            model = self._create_model(
                estimator,
                verbose=False,
                system=False,
                probability_threshold=probability_threshold,
            )
            model_results = (
                self.pull(pop=True)
                .reset_index()
                .drop(columns=["Split"], errors="ignore")
                .set_index(["Fold"])
                .loc[
                    [
                        self._get_return_train_score_indices_for_logging(
                            return_train_score=False
                        )
                    ]
                ]
            )
            model_results["probability_threshold"] = probability_threshold
            model_results["model"] = model[0]
            results_df.append(model_results)
            value = model_results[optimize].values[0]
            msg = f"Threshold: {probability_threshold}. {optimize}: {value}"
            if verbose:
                print(msg)
            self.logger.info(msg)
            return value * direction

        # This is necessary to make sure the sampler has a
        # deterministic seed.
        class FixedRandom(np.random.RandomState):
            def __init__(self_, seed=None) -> None:  # noqa
                super().__init__(self.seed)

        with patch("numpy.random.RandomState", FixedRandom):
            result = shgo(objective, ((0, 1),), **shgo_kwargs)

        message = (
            "optimization loop finished successfully. "
            f"Best threshold: {result.x[0]} with {optimize}={result.fun*direction}"
        )
        if verbose:
            print(message)
        self.logger.info(message)

        results_concat = pd.concat(results_df, axis=0)
        results_concat = results_concat.sort_values("probability_threshold")
        results_concat_melted = results_concat.drop("model", axis=1).melt(
            id_vars=["probability_threshold"],
            value_vars=list(results_concat.columns[:-1]),
        )
        best_model_by_metric = results_concat[
            results_concat["probability_threshold"] == result.x[0]
        ]["model"].iloc[0]
        assert isinstance(best_model_by_metric, CustomProbabilityThresholdClassifier)
        assert best_model_by_metric.probability_threshold == result.x[0]

        self.logger.info("plotting optimization threshold using plotly")

        title = f"{model_name} Probability Threshold Optimization (default = 0.5)"
        plot_kwargs = plot_kwargs or {}
        fig = px.line(
            results_concat_melted,
            x="probability_threshold",
            y="value",
            title=title,
            color="variable",
            **plot_kwargs,
        )
        fig.show()

        self.logger.info("returning model with best metric")
        if return_data:
            self.logger.info("also returning data as return_data = True")
            self.logger.info(
                "optimize_threshold() successfully completed......................................"
            )
            return (results_concat_melted, best_model_by_metric)
        else:
            self.logger.info(
                "optimize_threshold() successfully completed......................................"
            )
            return best_model_by_metric

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:
        

        return super().predict_model(
            estimator=estimator,
            data=data,
            probability_threshold=probability_threshold,
            encoded_labels=encoded_labels,
            raw_score=raw_score,
            round=round,
            verbose=verbose,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = False,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
    ) -> Any:
        

        return super().finalize_model(
            estimator=estimator,
            fit_kwargs=fit_kwargs,
            groups=groups,
            model_only=model_only,
            experiment_custom_tags=experiment_custom_tags,
        )

    def deploy_model(
        self,
        model,
        model_name: str,
        authentication: dict,
        platform: str = "aws",
    ):
        

        return super().deploy_model(
            model=model,
            model_name=model_name,
            authentication=authentication,
            platform=platform,
        )

    def save_model(
        self,
        model,
        model_name: str,
        model_only: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        

        return super().save_model(
            model=model,
            model_name=model_name,
            model_only=model_only,
            verbose=verbose,
            **kwargs,
        )

    def load_model(
        self,
        model_name: str,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):
        

        return super().load_model(
            model_name=model_name,
            platform=platform,
            authentication=authentication,
            verbose=verbose,
        )

    def automl(
        self,
        optimize: str = "Accuracy",
        use_holdout: bool = False,
        turbo: bool = True,
        return_train_score: bool = False,
    ) -> Any:
       
        return super().automl(
            optimize=optimize,
            use_holdout=use_holdout,
            turbo=turbo,
            return_train_score=return_train_score,
        )

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:
        
        return super().models(type=type, internal=internal, raise_errors=raise_errors)

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:
        

        return super().get_metrics(
            reset=reset,
            include_custom=include_custom,
            raise_errors=raise_errors,
        )

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        target: str = "pred",
        greater_is_better: bool = True,
        multiclass: bool = True,
        **kwargs,
    ) -> pd.Series:
       

        return super().add_metric(
            id=id,
            name=name,
            score_func=score_func,
            target=target,
            greater_is_better=greater_is_better,
            multiclass=multiclass,
            **kwargs,
        )

    def remove_metric(self, name_or_id: str):
        

        return super().remove_metric(name_or_id=name_or_id)

    def get_logs(
        self, experiment_name: Optional[str] = None, save: bool = False
    ) -> pd.DataFrame:
       
        return super().get_logs(experiment_name=experiment_name, save=save)

    def dashboard(
        self,
        estimator,
        display_format: str = "dash",
        dashboard_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        
        # soft dependencies check
        super().dashboard(
            estimator, display_format, dashboard_kwargs, run_kwargs, **kwargs
        )

        dashboard_kwargs = dashboard_kwargs or {}
        run_kwargs = run_kwargs or {}

        from explainerdashboard import ClassifierExplainer, ExplainerDashboard

        le = get_label_encoder(self.pipeline)
        if le:
            labels_ = list(le.classes_)
        else:
            labels_ = None

        # Replaceing chars which dash doesnt accept for column name `.` , `{`, `}`
        X_test_df = self.X_test_transformed.copy()
        X_test_df.columns = [
            col.replace(".", "__").replace("{", "__").replace("}", "__")
            for col in X_test_df.columns
        ]
        explainer = ClassifierExplainer(
            estimator, X_test_df, self.y_test_transformed, labels=labels_, **kwargs
        )
        return ExplainerDashboard(
            explainer, mode=display_format, **dashboard_kwargs
        ).run(**run_kwargs)
