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


import os
import shutil

import numpy as np
import pandas as pd
import pytest

from common.evaluation.metrics import DecisionTable, ThresholdCutter


@pytest.fixture()
def env():
	if not os.path.exists("/opt/checkpoints/unit_test"):
		os.mkdir("/opt/checkpoints/unit_test")
	yield
	if os.path.exists("/opt/checkpoints/unit_test"):
		shutil.rmtree("/opt/checkpoints/unit_test")


class TestDecisionTable:
	@pytest.mark.parametrize('bins', [1, 2, 5, 10, 50])
	def test_fit(self, bins, env):
		file_path = "/opt/checkpoints/decision_table.csv"
		dt = DecisionTable({"bin_number": bins})
		y_true = np.array([1] * 50000 + [0] * 50000)
		np.random.shuffle(y_true)
		y_pred = np.random.normal(0, 1, size=100000)
		dt.fit(y_true, y_pred)
		assert len(dt.stats) == bins
		dt.save(file_path)
		# read_from_file
		df = pd.read_csv(file_path)
		assert len(df) == bins
		assert df.iloc[-1]["累计总样本数"] == 100000
		assert df.iloc[-1]["累计负样本数"] == 50000
		assert df.iloc[-1]["累计负样本/负样本总数"] == "100.00%"
		assert df.iloc[-1]["累计正样本/正样本总数"] == "100.00%"
		assert df.iloc[-1]["累计负样本/累计总样本"] == "50.00%"

	@pytest.mark.parametrize("method", ["equal_frequency", "equal_width", 'other'])
	def test_difference_method(self, method, env):
		file_path = "/opt/checkpoints/decision_table.csv"
		dt = DecisionTable({"method": method})
		y_true = np.array([1] * 50000 + [0] * 50000)
		np.random.shuffle(y_true)
		y_pred = np.random.normal(0, 1, size=100000)
		if method not in ["equal_frequency", "equal_width"]:
			with pytest.raises(NotImplementedError) as e:
				dt.fit(y_true, y_pred)
		else:
			dt.fit(y_true, y_pred)
			dt.save(file_path)
			# read_from_file
			df = pd.read_csv(file_path)
			if method == "equal_frequency":
				assert (df["样本数"] == 100000 / dt.bin_number).all()

	def test_threshold_cutter(self):
		y = [1] * 10 + [0] * 40 + [1] * 40 + [0] * 10
		p = np.arange(0.5, 1, 0.005)
		tc = ThresholdCutter()
		tc.cut_by_index(y, p)
		np.testing.assert_almost_equal(tc.bst_score, 0.6)
		np.testing.assert_almost_equal(tc.bst_threshold, 0.75)
		tc2 = ThresholdCutter()
		tc2.cut_by_value(y, p)
		np.testing.assert_almost_equal(tc2.bst_score, 0.6)
		np.testing.assert_almost_equal(tc2.bst_threshold, 0.7475)

