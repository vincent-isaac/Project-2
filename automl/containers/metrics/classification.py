from typing import Any, Dict, Optional, Union

from sklearn import metrics
from sklearn.metrics._scorer import _BaseScorer

import automl.containers.base_container
import automl.internal.metrics
from automl.containers.metrics.base_metric import MetricContainer


class ClassificationMetricContainer(MetricContainer):
     

    def __init__(
        self,
        id: str,
        name: str,
        score_func: type,
        scorer: Optional[Union[str, _BaseScorer]] = None,
        target: str = "pred",
        args: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
        greater_is_better: bool = True,
        is_multiclass: bool = True,
        is_custom: bool = False,
    ) -> None:
        allowed_targets = ["pred", "pred_proba", "threshold"]
        if target not in allowed_targets:
            raise ValueError(f"Target must be one of {', '.join(allowed_targets)}.")

        if not args:
            args = {}
        if not isinstance(args, dict):
            raise TypeError("args needs to be a dictionary.")

        scorer = (
            scorer
            if scorer
            else automl.internal.metrics.make_scorer_with_error_score(
                score_func,
                needs_proba=target == "pred_proba",
                needs_threshold=target == "threshold",
                greater_is_better=greater_is_better,
                error_score=0.0,
                **args,
            )
        )

        super().__init__(
            id=id,
            name=name,
            score_func=score_func,
            scorer=scorer,
            args=args,
            display_name=display_name,
            greater_is_better=greater_is_better,
            is_custom=is_custom,
        )

        self.target = target
        self.is_multiclass = is_multiclass

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = {
            "ID": self.id,
            "Name": self.name,
            "Display Name": self.display_name,
            "Score Function": self.score_func,
            "Scorer": self.scorer,
            "Target": self.target,
            "Args": self.args,
            "Greater is Better": self.greater_is_better,
            "Multiclass": self.is_multiclass,
            "Custom": self.is_custom,
        }

        return d


class AccuracyMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="acc",
            name="Accuracy",
            score_func=metrics.accuracy_score,
            scorer="accuracy",
        )


class ROCAUCMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        args = {"average": "weighted", "multi_class": "ovr"}
        score_func = automl.internal.metrics.BinaryMulticlassScoreFunc(
            metrics.roc_auc_score,
            kwargs_if_binary={"average": "macro", "multi_class": "raise"},
        )
        super().__init__(
            id="auc",
            name="AUC",
            score_func=score_func,
            scorer=automl.internal.metrics.make_scorer_with_error_score(
                score_func, needs_proba=True, error_score=0.0, **args
            ),
            target="pred_proba",
            args=args,
        )


class RecallMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        args = {"average": "weighted"}
        score_func = automl.internal.metrics.BinaryMulticlassScoreFunc(
            automl.internal.metrics.EncodedDecodedLabelsScoreFunc(
                metrics.recall_score,
                automl.internal.metrics.get_pos_label(globals_dict),
            ),
            kwargs_if_binary={"average": "binary"},
        )
        super().__init__(
            id="recall",
            name="Recall",
            score_func=score_func,
            scorer=metrics.make_scorer(
                score_func,
                **args,
            ),
            args=args,
        )


class PrecisionMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        args = {"average": "weighted"}
        score_func = automl.internal.metrics.BinaryMulticlassScoreFunc(
            automl.internal.metrics.EncodedDecodedLabelsScoreFunc(
                metrics.precision_score,
                automl.internal.metrics.get_pos_label(globals_dict),
            ),
            kwargs_if_binary={"average": "binary"},
        )
        super().__init__(
            id="precision",
            name="Precision",
            display_name="Prec.",
            score_func=score_func,
            scorer=metrics.make_scorer(
                score_func,
                **args,
            ),
            args=args,
        )


class F1MetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        args = {"average": "weighted"}
        score_func = automl.internal.metrics.BinaryMulticlassScoreFunc(
            automl.internal.metrics.EncodedDecodedLabelsScoreFunc(
                metrics.f1_score, automl.internal.metrics.get_pos_label(globals_dict)
            ),
            kwargs_if_binary={"average": "binary"},
        )
        super().__init__(
            id="f1",
            name="F1",
            score_func=score_func,
            scorer=metrics.make_scorer(
                score_func,
                **args,
            ),
            args=args,
        )


class KappaMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="kappa",
            name="Kappa",
            score_func=metrics.cohen_kappa_score,
            scorer=metrics.make_scorer(metrics.cohen_kappa_score),
        )


class MCCMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="mcc",
            name="MCC",
            score_func=metrics.matthews_corrcoef,
            scorer=metrics.make_scorer(metrics.matthews_corrcoef),
        )


def get_all_metric_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, ClassificationMetricContainer]:
    return automl.containers.base_container.get_all_containers(
        globals(), globals_dict, ClassificationMetricContainer, raise_errors
    )
