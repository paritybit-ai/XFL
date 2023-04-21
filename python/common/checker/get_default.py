import copy

from .qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional
from .x_types import String, Bool, Integer, Float, Any, All


def get_default(descriptor):
    if isinstance(descriptor, dict):
        if "__rule__" not in descriptor:
            descriptor_copy = copy.deepcopy(descriptor)
            descriptor_copy["__rule__"] = [Required(*list(descriptor.keys()))]
            return get_default(descriptor_copy)
        else:
            if not isinstance(descriptor.get("__rule__"), list):
                descriptor_copy = copy.deepcopy(descriptor)
                descriptor_copy["__rule__"] = [descriptor["__rule__"]]
                return get_default(descriptor_copy)
            else:
                descriptor_copy = copy.deepcopy(descriptor)
                is_continue = True
                for i, item in enumerate(descriptor["__rule__"]):
                    if isinstance(item, Optional):
                        if item.default is not None:
                            descriptor_copy["__rule__"][i] = item.default
                            is_continue = False
                
                if not is_continue:
                    return get_default(descriptor_copy)
                else:
                
                    res = {}
                    for item in descriptor["__rule__"]:
                        if isinstance(item, OneOf):
                            key = get_default(item.default)
                            res[key] = get_default(descriptor[key])
                        elif isinstance(item, SomeOf):
                            if isinstance(item.default, (list, tuple)):
                                for k in item.default:
                                    key = get_default(k)
                                    res[key] = get_default(descriptor[key])
                            else:
                                key = get_default(item.default)
                                res[key] = get_default(descriptor[key])
                        elif isinstance(item, Required):
                            for k in item.default:
                                key = get_default(k)
                                res[key] = get_default(descriptor[key])
                        elif isinstance(item, Optional):
                            if item.default is None:
                                pass
                            else:
                                raise ValueError("Code is not well developed.")
                            # else:
                            #     res[get_default(item.default)] = descriptor[item.default]
                            #     # res[item.default] = get_default(descriptor[item.default])
                                
                            #     if isinstance(item.default, (str, int, float)):
                            #         res[item.default] = get_default(descriptor[item.default])
                            #     elif isinstance(item, (String, Integer, Float)):
                            #         res[get_default(item.default)] = get_default(descriptor[item.default])
                            #     elif isinstance(item.default, (bool, Bool)):
                            #         raise ValueError("Rule is not set correctly.")
                            #     else:
                            #         raise ValueError("Code is not well developed.")
                                    
                                
                                # for k in item.default:
                                #     key = get_default(k)
                                #     res[key] = get_default(descriptor[key])
                        elif isinstance(item, RepeatableSomeOf):
                            raise ValueError("Rule is not set correctly.")
                        elif isinstance(item, (String, Bool, Integer, Float)):
                            key = get_default(item.default)
                            key2 = None
                            for k in descriptor.keys():
                                if k.__hash__() == item.__class__().__hash__():
                                    key2 = k
                            
                            res[key] = get_default(descriptor[key2])
                        elif isinstance(item, Any):
                            pass
                        else:
                            res[item] = get_default(descriptor[item])
                    return res
    elif isinstance(descriptor, list):
        if len(descriptor) == 0:
            return []
        elif len(descriptor) == 1:
            if isinstance(descriptor[0], OneOf):
                return [get_default(descriptor[0].default)]
            elif isinstance(descriptor[0], (SomeOf, RepeatableSomeOf)):
                if descriptor[0].default is None:
                    return []
                else:
                    return [get_default(v) for v in descriptor[0].default]
            elif isinstance(descriptor[0], Required):
                raise ValueError("Rule is not set correctly.")
            elif isinstance(descriptor[0], Optional):
                if descriptor[0].default is None:
                    return []
                else:
                    return get_default([descriptor[0].default])
            elif isinstance(descriptor[0], (String, Bool, Integer, Float)):
                return [descriptor[0].default]
            elif isinstance(descriptor[0], Any):
                return []
            else:
                return [get_default(v) for v in descriptor]
        else:
            return [get_default(v) for v in descriptor]
    elif isinstance(descriptor, (OneOf, Optional)):
        return get_default(descriptor.default)
    elif isinstance(descriptor, (SomeOf, RepeatableSomeOf, Required)):
        raise ValueError("Rule is not set correctly.")
    elif isinstance(descriptor, (String, Bool, Integer, Float)):
        return descriptor.default
    elif isinstance(descriptor, (Any, All)):
        return None
    else:
        return descriptor
    
    