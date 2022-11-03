import pytest

from common.xregister import xregister, XRegister
from algorithm.framework.vertical.xgboost.trainer import VerticalXgboostTrainer


class Abc():
    pass


class TestXRegister():
    @pytest.mark.parametrize("target", [
        (Abc), ("abc"), ("Abc"), (Abc), ("CDE")
    ])
    def test_register(self, target):
        if target == "abc":
            xregister.register(target)(lambda x: x+2)
            assert 'abc' in xregister.__dict__
        elif target == "CDE":
            with pytest.raises(TypeError):
                xregister.register(target)("CDE")
        else:
            xregister.register(target)
            assert 'Abc' in xregister.__dict__
            
    @pytest.mark.parametrize("name", ["Abc", "XYZ"])
    def test_call(self, name):
        if name == "Abc":
            assert xregister(name).__name__ == Abc.__name__
        else:
            with pytest.raises(KeyError):
                xregister(name)
         
    @pytest.mark.parametrize("name", ["Abc", "XYZ"])     
    def test_unregister(self, name):
        xregister.unregister(name)
        assert "Abc" not in xregister.__dict__
        
    def test_registered_object(self):
        res = xregister.registered_object
        assert xregister.__dict__ == res

    def test_get_class_name(self):
        name = XRegister.get_class_name()
        assert name == "XRegister"
