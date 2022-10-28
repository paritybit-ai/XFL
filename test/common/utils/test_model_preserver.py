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


import pytest

from common.utils.model_preserver import ModelPreserver, os, torch


class TestModelPreserver():

    @pytest.mark.parametrize('final,model_path', [(True, "test_save_dir/test.model.pth"), (False, "test_save_dir/test.model_epoch_10.pth")])
    def  test_save(self, mocker, final, model_path):
        mocker.patch("os.makedirs")
        mocker.patch("torch.save")
        ModelPreserver.save("test_save_dir","test.model.pth", {}, epoch=10, final=final, suggest_threshold=0.1)
        os.makedirs.assert_called_once_with("test_save_dir")
        torch.save.assert_called_once_with({"state_dict":{},"suggest_threshold":0.1}, model_path)




    def test_load(self, mocker):
        mocker.patch("torch.load")
        ModelPreserver.load("test_path")
        torch.load.assert_called_once_with("test_path")

