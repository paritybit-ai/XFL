
import inspect
import math

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import sklearn.metrics as sklearn_metrics

import algorithm.core.metrics as custom_metrics
from algorithm.core.metrics import metric_dict
# from common.checker.qualifiers import (OneOf, Optional, RepeatableSomeOf,
#                                        Required, SomeOf)
# from common.checker.x_types import All, Any, Bool, Float, Integer, String


def gen_torch_optim_dict(out_path: str):
    methods = [getattr(optim, name) for name in dir(optim) if isinstance(getattr(optim, name), type) and name not in ['Optimizer']]
    blank = ''

    with open(out_path, 'w') as f:
        f.write('from common.checker.x_types import String, Bool, Integer, Float, Any, All\n')
        f.write('from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional\n')
        
        f.write('\n\n')
        f.write('optimizer = {')
        
        blank += '    '

        for i, method in enumerate(methods):
            # print(method.__name__)
            # print(inspect.getfullargspec(method))
            # print(inspect.signature(method).parameters)
            mark0 = ',' if i > 0 else ''
            f.write(f'{mark0}\n' + blank + f'"{method.__name__}": ' + '{')
            blank += '    '
            params = list(inspect.signature(method).parameters.values())
            
            required_params = []
            whole_params = []
            
            for j, param in enumerate(params):
                name = param.name
                default = param.default
                
                mark1 = ',' if j > 1 else ''
                
                # Don't support params
                if name == 'params':
                    continue
                
                # No default lr value for SGD
                # if name == 'lr' and not isinstance(name, (int, float)):
                #     default = 0.001
                    
                if isinstance(default, bool):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Bool({default})')
                elif isinstance(default, int):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Integer({default})')
                elif isinstance(default, float):
                    default = None if math.isnan(default) else default
                    f.write(f'{mark1}\n' + blank + f'"{name}": Float({default})')
                elif isinstance(default, str):
                    f.write(f'{mark1}\n' + blank + f'"{name}": String("{default}")')
                elif isinstance(default, (list, tuple)):
                    f.write(f'{mark1}\n' + blank + f'"{name}": [')
                    for k, item in enumerate(default):
                        mark2 = ',' if k != 0 else ''
                        if isinstance(item, bool):
                            v = f'Bool({item})'
                        elif isinstance(item, int):
                            v = f'Integer({item})'
                        elif isinstance(item, float):
                            item = None if math.isnan(item) else item
                            v = f'Float({item})'
                        elif isinstance(item, str):
                            v = f'String("{item}")'
                        else:
                            v = f'Any({item})'
                        
                        f.write(f'{mark2}\n' + blank + '        ' + v)
                    f.write(f'{mark1}\n' + blank + '    ' + ']')
                elif default is None:
                    f.write(f'{mark1}\n' + blank + f'"{name}": All(None)')
                else:
                    f.write(f'{mark1}\n' + blank + f'"{name}": ' + 'All("No default value")')
                    required_params.append(name)
                    print(f"{name}, {default}")
                    pass
                whole_params.append(name)
            
            if len(whole_params) != 0:
                mark2 = ',' if len(whole_params) > 0 else ''
                
                f.write(f'{mark2}\n' + blank + '"__rule__": [')
                
                if len(required_params) > 0:
                    f.write("Required(")
                    for j, name in enumerate(required_params):
                        mark3 = ', ' if j > 0 else ''
                        f.write(f'{mark3}"{name}"')
                    f.write(")")
                
                optional_params = list(set(whole_params) - set(required_params))
                    
                for j, name in enumerate(optional_params):
                    mark3 = ', ' if len(required_params) > 0 or j > 0 else ''
                    f.write(f'{mark3}Optional("{name}")')
                    
                f.write(']')
                    
                blank = blank[:-4]
                f.write('\n' + blank + '}')
            
        f.write('\n}\n')
        

def gen_torch_lr_scheduler_dict(out_path: str):
    methods = [getattr(lr_scheduler, name) for name in dir(lr_scheduler) if isinstance(getattr(lr_scheduler, name), type) and '_' not in name and name not in ['Optimizer', 'ChainedScheduler', 'Counter']]
    blank = ''

    with open(out_path, 'w') as f:
        f.write('from common.checker.x_types import String, Bool, Integer, Float, Any, All\n')
        f.write('from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional\n')
        
        f.write('\n\n')
        f.write('lr_scheduler = {')
        
        blank += '    '

        for i, method in enumerate(methods):
            # print(method.__name__)
            # print(inspect.getfullargspec(method))
            # print(inspect.signature(method).parameters)
            mark0 = ',' if i > 0 else ''
            f.write(f'{mark0}\n' + blank + f'"{method.__name__}": ' + '{')
            blank += '    '
            params = list(inspect.signature(method).parameters.values())
            
            required_params = []
            whole_params = []
            
            for j, param in enumerate(params):
                name = param.name
                default = param.default
                
                mark1 = ',' if j > 1 else ''
                
                # Don't support optimizer
                if name == 'optimizer':
                    continue
                    
                if isinstance(default, bool):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Bool({default})')
                elif isinstance(default, int):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Integer({default})')
                elif isinstance(default, float):
                    default = None if math.isnan(default) else default
                    f.write(f'{mark1}\n' + blank + f'"{name}": Float({default})')
                elif isinstance(default, str):
                    f.write(f'{mark1}\n' + blank + f'"{name}": String("{default}")')
                elif isinstance(default, (list, tuple)):
                    f.write(f'{mark1}\n' + blank + f'"{name}": [')
                    for k, item in enumerate(default):
                        mark2 = ',' if k != 0 else ''
                        if isinstance(item, bool):
                            v = f'Bool({item})'
                        elif isinstance(item, int):
                            v = f'Integer({item})'
                        elif isinstance(item, float):
                            item = None if math.isnan(item) else item
                            v = f'Float({item})'
                        elif isinstance(item, str):
                            v = f'String("{item}")'
                        else:
                            v = f'Any({item})'
                        
                        f.write(f'{mark2}\n' + blank + '        ' + v)
                    f.write(f'{mark1}\n' + blank + '    ' + ']')
                elif default is None:
                    f.write(f'{mark1}\n' + blank + f'"{name}": All(None)')
                else:
                    f.write(f'{mark1}\n' + blank + f'"{name}": ' + 'All("No default value")')
                    required_params.append(name)
                    print(f"{name}, {default}")
                    pass
                whole_params.append(name)
            
            if len(whole_params) != 0:
                mark2 = ',' if len(whole_params) > 0 else ''
                
                f.write(f'{mark2}\n' + blank + '"__rule__": [')
                
                if len(required_params) > 0:
                    f.write("Required(")
                    for j, name in enumerate(required_params):
                        mark3 = ', ' if j > 0 else ''
                        f.write(f'{mark3}"{name}"')
                    f.write(")")
                
                optional_params = list(set(whole_params) - set(required_params))
                    
                for j, name in enumerate(optional_params):
                    mark3 = ', ' if len(required_params) > 0 or j > 0 else ''
                    f.write(f'{mark3}Optional("{name}")')
                    
                f.write(']')
                
            blank = blank[:-4]
            f.write('\n' + blank + '}')
            
        f.write('\n}\n')
          

def gen_torch_lossfunc_dict(out_path: str):
    methods = [getattr(nn, name) for name in dir(nn) if isinstance(getattr(nn, name), type) and 'Loss' in name]
    blank = ''

    with open(out_path, 'w') as f:
        f.write('from common.checker.x_types import String, Bool, Integer, Float, Any, All\n')
        f.write('from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional\n')
        
        f.write('\n\n')
        f.write('lossfunc = {')
        
        blank += '    '

        for i, method in enumerate(methods):
            # print(method.__name__)
            # print(inspect.getfullargspec(method))
            # print(inspect.signature(method).parameters)
            mark0 = ',' if i > 0 else ''
            f.write(f'{mark0}\n' + blank + f'"{method.__name__}": ' + '{')
            blank += '    '
            params = list(inspect.signature(method).parameters.values())
            
            required_params = []
            whole_params = []
            
            for j, param in enumerate(params):
                name = param.name
                default = param.default
                
                mark1 = ',' if j > 0 else ''
                
                # Don't support params
                # if name == 'optimizer':
                #     continue
                    
                if isinstance(default, bool):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Bool({default})')
                elif isinstance(default, int):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Integer({default})')
                elif isinstance(default, float):
                    default = None if math.isnan(default) else default
                    f.write(f'{mark1}\n' + blank + f'"{name}": Float({default})')
                elif isinstance(default, str):
                    f.write(f'{mark1}\n' + blank + f'"{name}": String("{default}")')
                elif isinstance(default, (list, tuple)):
                    f.write(f'{mark1}\n' + blank + f'"{name}": [')
                    for k, item in enumerate(default):
                        mark2 = ',' if k != 0 else ''
                        if isinstance(item, bool):
                            v = f'Bool({item})'
                        elif isinstance(item, int):
                            v = f'Integer({item})'
                        elif isinstance(item, float):
                            item = None if math.isnan(item) else item
                            v = f'Float({item})'
                        elif isinstance(item, str):
                            v = f'String("{item}")'
                        else:
                            v = f'Any({item})'
                        
                        f.write(f'{mark2}\n' + blank + '        ' + v)
                    f.write(f'{mark1}\n' + blank + '    ' + ']')
                elif default is None:
                    f.write(f'{mark1}\n' + blank + f'"{name}": All(None)')
                else:
                    f.write(f'{mark1}\n' + blank + f'"{name}": ' + 'All("No default value")')
                    required_params.append(name)
                    print(f"{name}, {default}")
                    pass
                whole_params.append(name)
            
            if len(whole_params) != 0:
                mark2 = ',' if len(whole_params) > 0 else ''
                
                f.write(f'{mark2}\n' + blank + '"__rule__": [')
                
                if len(required_params) > 0:
                    f.write("Required(")
                    for j, name in enumerate(required_params):
                        mark3 = ', ' if j > 0 else ''
                        f.write(f'{mark3}"{name}"')
                    f.write(")")
                
                optional_params = list(set(whole_params) - set(required_params))
                    
                for j, name in enumerate(optional_params):
                    mark3 = ', ' if len(required_params) > 0 or j > 0 else ''
                    f.write(f'{mark3}Optional("{name}")')
                    
                f.write(']')
                
            blank = blank[:-4]
            f.write('\n' + blank + '}')
            
        f.write('\n}\n')
        

def gen_metric_dict(out_path: str):
    candidate_methods_name = dir(sklearn_metrics)   # [getattr(sklearn_metrics, name) for name in dir(sklearn_metrics)]
    
    valid_combination = [('y_true', 'y_pred'), ('X', 'Y'), ('y_true', 'y_score'), ('X', 'labels'), ('labels_true', 'labels_pred'), ('x', 'y'), ('y_true', 'y_prob'), ('X', 'labels'), ('a', 'b')]
    
    methods = []
    for name in candidate_methods_name:
        method = getattr(sklearn_metrics, name)
        if inspect.isfunction(method):
            params = list(inspect.signature(method).parameters.keys())
            if len(params) >= 2:
                if (params[0], params[1]) in valid_combination:
                    methods.append(name)
                    # print(params, name)
    
    methods = [getattr(sklearn_metrics, name) for name in methods]
    
    custom_methods = []
    for name in dir(custom_metrics):
        method = getattr(custom_metrics, name)
        if inspect.isfunction(method):
            if name not in ["get_metric"]:
                custom_methods.append(name)
                
    custom_methods = [getattr(custom_metrics, name) for name in custom_methods]
    
    names_map = {v: k for k, v in metric_dict.items()}
                
    # print(list(set(dir(sklearn_metrics)) - set(methods)))
    # print("####")
    # for name in list(set(dir(sklearn_metrics)) - set(methods)):
    #     method = getattr(sklearn_metrics, name)
    #     if inspect.isfunction(method):
    #         print(list(inspect.signature(method).parameters.keys()), name)
    blank = ''

    with open(out_path, 'w') as f:
        f.write('from common.checker.x_types import String, Bool, Integer, Float, Any, All\n')
        f.write('from common.checker.qualifiers import OneOf, SomeOf, RepeatableSomeOf, Required, Optional\n')
        
        f.write('\n\n')
        f.write('metrics = {')
        
        blank += '    '

        for i, method in enumerate(methods + custom_methods):
            # print(method.__name__)
            # print(inspect.getfullargspec(method))
            # print(inspect.signature(method).parameters)
            mark0 = ',' if i > 0 else ''
            if method.__name__ in names_map:
                f.write(f'{mark0}\n' + blank + f'"{names_map[method.__name__]}": ' + '{')
            else:
                f.write(f'{mark0}\n' + blank + f'"{method.__name__}": ' + '{')
            blank += '    '
            params = list(inspect.signature(method).parameters.values())[2:]
            
            required_params = []
            whole_params = []
            
            is_first = True
            for j, param in enumerate(params):
                name = param.name
                default = param.default

                if name == 'kwds':
                    continue
                
                mark1 = ',' if is_first is False else ''
                
                is_first = False
                
                # Don't support params
                # if name == 'optimizer':
                #     continue
                
                if isinstance(default, bool):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Bool({default})')
                elif isinstance(default, int):
                    f.write(f'{mark1}\n' + blank + f'"{name}": Integer({default})')
                elif isinstance(default, float):
                    default = None if math.isnan(default) else default
                    f.write(f'{mark1}\n' + blank + f'"{name}": Float({default})')
                elif isinstance(default, str):
                    f.write(f'{mark1}\n' + blank + f'"{name}": String("{default}")')
                elif isinstance(default, (list, tuple)):
                    f.write(f'{mark1}\n' + blank + f'"{name}": [')
                    for k, item in enumerate(default):
                        mark2 = ',' if k != 0 else ''
                        if isinstance(item, bool):
                            v = f'Bool({item})'
                        elif isinstance(item, int):
                            item = None if math.isnan(item) else item
                            v = f'Integer({item})'
                        elif isinstance(item, float):
                            v = f'Float({item})'
                        elif isinstance(item, str):
                            v = f'String("{item}")'
                        else:
                            v = f'Any({item})'
                        
                        f.write(f'{mark2}\n' + blank + '        ' + v)
                    f.write(f'{mark1}\n' + blank + '    ' + ']')
                elif default is None:
                    f.write(f'{mark1}\n' + blank + f'"{name}": All(None)')
                else:
                    f.write(f'{mark1}\n' + blank + f'"{name}": ' + 'All("No default value")')
                    required_params.append(name)
                    print(f"{name}, {default}")
                    pass
                whole_params.append(name)
                
            if len(whole_params) != 0:
                mark2 = ',' if len(whole_params) > 0 else ''
                
                f.write(f'{mark2}\n' + blank + '"__rule__": [')
                
                if len(required_params) > 0:
                    f.write("Required(")
                    for j, name in enumerate(required_params):
                        mark3 = ', ' if j > 0 else ''
                        f.write(f'{mark3}"{name}"')
                    f.write(")")
                
                optional_params = list(set(whole_params) - set(required_params))
                    
                for j, name in enumerate(optional_params):
                    mark3 = ', ' if len(required_params) > 0 or j > 0 else ''
                    f.write(f'{mark3}Optional("{name}")')
                    
                f.write(']')
                
            blank = blank[:-4]
            f.write('\n' + blank + '}')
            
        f.write('\n}\n')
          

if __name__ == "__main__":
    from pathlib import Path
    out_path = Path(__file__).parent / 'optimizer.py'
    gen_torch_optim_dict(out_path)
    
    out_path = Path(__file__).parent / 'lr_scheduler.py'
    gen_torch_lr_scheduler_dict(out_path)
    
    out_path = Path(__file__).parent / 'lossfunc.py'
    gen_torch_lossfunc_dict(out_path)
    
    out_path = Path(__file__).parent / 'metrics.py'
    gen_metric_dict(out_path)

        




