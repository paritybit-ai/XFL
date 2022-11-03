import pytest

from common.xoperator import get_operator
from algorithm.framework.vertical.xgboost.trainer import VerticalXgboostTrainer


@pytest.mark.parametrize("name, role", [
        ("vertical_xgboost", "trainer"), ("vertical_xgboost", "client"), 
        ("mixed_xgboost", "label_trainer"), ("vertical_abc", "assist_trainer")
 ])
def test_get_operator(name, role):
    if role == "client" or name in ["mixed_xgboost", "vertical_abc"]:
        with pytest.raises(ValueError):
            get_operator(name, role)
    else:
        assert get_operator(name, role).__name__ == VerticalXgboostTrainer.__name__

