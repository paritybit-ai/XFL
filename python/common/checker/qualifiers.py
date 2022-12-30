from .base import Base


class OfBase(Base):
    def __init__(self, *args, default=None):
        super().__init__()
        self.candidates = args
        self.default = None

    def set_default(self, value):
        '''设置default值'''
        self.default = value
        return self


class OneOf(OfBase):
    '''
    只选择一个
    '''

    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        if self.default is None:
            self.default = self.candidates[0]

    def set_default_index(self, idx):
        '''设置default的index'''
        self.default = self.candidates[idx]
        return self


class SomeOf(OfBase):
    '''
    不放回的选多个(>=1, <=候选数)
    '''

    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        if self.default is None:
            self.default = [self.candidates[0]]

    def set_default_indices(self, *idx):
        '''设置default的index，可以有多个'''
        self.default = [self.candidates[i] for i in idx]
        return self


# repeatable, many
class RepeatableSomeOf(OfBase):
    '''
    有放回的选多个
    '''

    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)
        if self.default is None:
            self.default = [self.candidates[0]]

    def set_default_indices(self, *idx):
        '''设置default的index，可以有多个'''
        self.default = [self.candidates[i] for i in idx]
        return self


class Required(OfBase):
    '''
    必须都存在
    '''

    def __init__(self, *args):
        super().__init__(*args)
        self.default = args


class Optional(OfBase):
    '''
    在dict的__rule__中表示该key可不存在，在list中表示list可为空，其他地方表示该值可为None。
    注：Optional只能接受一个参数。
    '''

    def __init__(self, *args, default=None):
        super().__init__(*args, default=default)

    def set_default_not_none(self):
        '''设置default为非None值'''

        self.default = self.candidates[0]
        return self
