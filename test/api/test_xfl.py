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


import argparse
import sys

import client
import scheduler_run
import trainer_run
import xfl
from common.utils.logger import logger


class mocker_version_info():
    major = 3
    minor = 7



def test_add_args(mocker):
    spy_add_mutually_exclusive_group = mocker.spy(argparse.ArgumentParser,'add_mutually_exclusive_group')
    mocker.spy(argparse.ArgumentParser,'add_argument')
    parser = xfl.add_args(argparse.ArgumentParser(description="XFL - BaseBit Federated Learning"))
    spy_add_mutually_exclusive_group.assert_called_once()
    assert parser.parse_args(['-s']) == argparse.Namespace(scheduler=True, assist_trainer=False, trainer='trainer', client=None, bar=False, config_path='/opt/config')
    assert parser.parse_args(['-a']) == argparse.Namespace(scheduler=False, assist_trainer=True, trainer='trainer', client=None, bar=False, config_path='/opt/config')
    assert parser.parse_args(['-t']) == argparse.Namespace(scheduler=False, assist_trainer=False, trainer='trainer', client=None, bar=False, config_path='/opt/config')
    assert parser.parse_args(['-c', 'start']) == argparse.Namespace(scheduler=False, assist_trainer=False, trainer='trainer', client='start', bar=False, config_path='/opt/config')
    


def test_main(mocker):
    mocker.patch('argparse.ArgumentParser.parse_args',return_value=argparse.Namespace(
        scheduler=True, assist_trainer=False, trainer='trainer', client=None, bar=False, config_path='/opt/config'
    ))
    mocker.patch('scheduler_run.main')
    xfl.main()
    scheduler_run.main.assert_called_once_with("/opt/config", False)

    mocker.patch('argparse.ArgumentParser.parse_args',return_value=argparse.Namespace(
        scheduler=False, assist_trainer=False, trainer='trainer', client='start', bar=False, config_path='/opt/config'
    ))
    mocker.patch('client.main')
    xfl.main()
    client.main.assert_called_once_with("start", "/opt/config")

    mocker.patch('argparse.ArgumentParser.parse_args',return_value=argparse.Namespace(
        scheduler=False, assist_trainer=True, trainer='trainer', client=None, bar=False, config_path='/opt/config'
    ))
    mocker.patch('trainer_run.main')
    xfl.main()
    trainer_run.main.assert_called_once_with("assist_trainer", "assist_trainer", config_path='/opt/config')

    mocker.patch('argparse.ArgumentParser.parse_args',return_value=argparse.Namespace(
        scheduler=False, assist_trainer=False, trainer='trainer', client=None, bar=False, config_path='/opt/config'
    ))
    mocker.patch('trainer_run.main')
    xfl.main()
    trainer_run.main.assert_called_once_with("trainer", "trainer", config_path='/opt/config')




def test_check_version(mocker):
    import collections
    mock_version_info = collections.namedtuple('mock_version_info', ['major', 'minor'])
    mocker.patch.object(sys, 'version_info', mock_version_info(3,7))
    mocker.patch.object(logger, 'error')
    mocker.patch('sys.exit')
    xfl.check_version()
    logger.error.assert_called_once_with("Python Version is not: 3.9" )
    sys.exit.assert_called_once_with(-1)
