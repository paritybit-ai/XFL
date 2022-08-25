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


class FeatureImportance(object):
    def __init__(self, importance_gain=0, importance_split=0, main_type="split"):
        self.legal_type = ["split", "gain"]
        assert main_type in self.legal_type, "illegal importance type {}".format(main_type)
        self.importance_gain = importance_gain
        self.importance_split = importance_split
        self.main_type = main_type

    def get(self):
        if self.main_type == "split":
            return self.importance_split
        elif self.main_type == "gain":
            return self.importance_gain

    def add_gain(self, val):
        self.importance_gain += val

    def add_split(self, val):
        self.importance_split += val

    def __eq__(self, other):
        if self.main_type == "split":
            return self.importance_split == other.importance_split
        elif self.main_type == "gain":
            return self.importance_gain == other.importance_gain

    def __lt__(self, other):
        if self.main_type == "split":
            return self.importance_split < other.importance_split
        elif self.main_type == "gain":
            return self.importance_gain < other.importance_gain

    def __repr__(self):
        if self.main_type == "gain":
            return "importance: {}".format(self.importance_gain)
        elif self.main_type == "split":
            return "importance: {}".format(self.importance_split)

    def __add__(self, other):
        new_importance = FeatureImportance(main_type=self.main_type,
                                           importance_gain=self.importance_gain + other.importance_gain,
                                           importance_split=self.importance_split + other.importance_split)
        return new_importance
