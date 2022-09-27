# Copyright 2022 The XFL Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union


class Hooker(object):
    def __init__(self):
        self.hooks = {}
        self.context = {}

    def register_hook(self, place: str, rank: int, func: object, desc: str = ''):
        """ register hook.

        Args:
            place (str): hook place.
            rank (int): execute rank with the same hook name.
            func (object): function to register.
            desc (str, optional): description of the function to register. Defaults to ''.

        Raises:
            ValueError: when rank of the hook place has been registered.
        """
        if place not in self.hooks:
            self.hooks[place] = {}

        if rank in self.hooks[place]:
            raise ValueError(
                f"Rank {rank} of hook place {place} has already been registered.")

        self.hooks[place][rank] = {}
        self.hooks[place][rank]['func'] = func
        self.hooks[place][rank]['desc'] = desc

    def execute_hook_at(self, place: str) -> int:
        """ execute functions registered by the hook place.

        Args:
            place (str): hook place.

        Returns:
            int: 1 represent needs break after the hook execution, else 0.
        """
        hooks = self.hooks.get(place, {})

        for rank in sorted(hooks):
            if hooks[rank]['func'](self.context) == 1:
                return 1
        return 0

    def declare_hooks(self, place: Union[str, list[str]]):
        """ declare hooks.

        Args:
            place (Union[str, list[str]]): hooks place
        """
        if isinstance(place, list):
            for place in place:
                if str(place) not in self.hooks:
                    self.hooks[str(place)] = {}
        else:
            if str(place) not in self.hooks:
                self.hooks[str(place)] = {}

    def declare_context(self, place: Union[str, list[str]]):
        """ declare context place.

        Args:
            place (Union[str, list[str]]): place of context to declare.
        """
        if isinstance(place, list):
            for place in place:
                if str(place) not in self.context:
                    self.context[str(place)] = None
        else:
            if str(place) not in self.context:
                self.context[str(place)] = None
