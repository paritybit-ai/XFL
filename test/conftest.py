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


import time

import pytest

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


@pytest.fixture(scope='session', autouse=True)
def timer_session_scope():
	start = time.time()
	print('\nstart: {}'.format(time.strftime(DATE_FORMAT, time.localtime(start))))
	yield
	finished = time.time()
	print('finished: {}'.format(time.strftime(DATE_FORMAT, time.localtime(finished))))
	print('Total time cost: {:.3f}s'.format(finished - start))


@pytest.fixture(scope='function', autouse=True)
def timer_function_scope():
	start = time.time()
	yield
	print('Time cost: {:.3f}s'.format(time.time() - start))

