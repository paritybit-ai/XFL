import json

from .qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional
from .x_types import String, Bool, Integer, Float, Any, All
from .checker import check, Checked


def get_matched_config(config, rule):
    # Only Any, All is supported
    r = check(config, rule)

    def get_matched(checked):
        if isinstance(checked, Checked):
            if isinstance(checked.value, dict):
                tmp = {}
                for k, v in checked.value.items():
                    if hasattr(k, 'is_match'):
                        if k.is_match:
                            tmp.update({k.value: get_matched(v)})
                    else:
                        tmp.update({k: get_matched(v)})
                return tmp
            else:
                if checked.is_match:
                    return checked.value
                else:
                    return None
        else:
            return checked
    
    return get_matched(r)
