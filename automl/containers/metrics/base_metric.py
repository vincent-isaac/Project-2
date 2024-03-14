

from typing import Any, Dict, Optional, Union

from sklearn.metrics import make_scorer  # type: ignore
from sklearn.metrics._scorer import _BaseScorer  # type: ignore

from automl.containers.base_container import BaseContainer


class MetricContainer(BaseContainer):
    

    def __init__(
        self,
        id: str,
        name: str,
        score_func: type,
        scorer: Optional[Union[str, _BaseScorer]] = None,
        args: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
        greater_is_better: bool = True,
        is_custom: bool = False,
    ) -> None:
        super().__init__(id=id, name=name, class_def=score_func, args=args)
        self.scorer = scorer if scorer else make_scorer(score_func, **args)
        self.display_name = display_name if display_name else name
        self.greater_is_better = greater_is_better
        self.is_custom = is_custom

    @property
    def score_func(self):
        return self.class_def

    @score_func.setter
    def score_func(self, value):
        self.class_def = value

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        
        d = {
            "ID": self.id,
            "Name": self.name,
            "Display Name": self.display_name,
            "Score Function": self.score_func,
            "Scorer": self.scorer,
            "Args": self.args,
            "Greater is Better": self.greater_is_better,
            "Custom": self.is_custom,
        }

        return d
