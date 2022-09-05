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
	@pytest.mark.parametrize('bins', [-1, 0, 1, 2, 5, 10, 50])
	def test_bins(self, bins, env):
		io_file_path = "/opt/checkpoints/decision_table_{}.csv".format(bins)
		config = {"bin_number": bins}

		if bins <= 1:
			with pytest.raises(ValueError) as e:
				dt = DecisionTable(config)
		else:
			dt = DecisionTable(config)
			# prepare a random test data
			y_true = np.array([1] * 50000 + [0] * 50000)
			y_pred = np.random.normal(0, 1, size=100000)
			np.random.shuffle(y_true)

			dt.fit(y_true, y_pred)
			assert len(dt.stats) == bins
			dt.save(io_file_path)
			# read_from_file
			df = pd.read_csv(io_file_path)
			assert len(df) == bins
			assert (df["样本数"] == (100000 / bins)).all()
			assert df.iloc[-1]["累计总样本数"] == 100000
			assert df.iloc[-1]["累计负样本数"] == 50000
			assert df.iloc[-1]["累计负样本/负样本总数"] == "100.00%"
			assert df.iloc[-1]["累计正样本/正样本总数"] == "100.00%"
			assert df.iloc[-1]["累计负样本/累计总样本"] == "50.00%"

	@pytest.mark.parametrize("method", ["equal_frequency", "equal_width", 'other'])
	def test_methods(self, method, env):
		io_file_path = "/opt/checkpoints/decision_table_{}.csv".format(method)
		config = {"method": method}
		if method not in ("equal_frequency", "equal_width"):
			with pytest.raises(NotImplementedError) as e:
				dt = DecisionTable(config)
		else:
			dt = DecisionTable(config)

			# prepare a random test data
			y_true = np.array([1] * 50000 + [0] * 50000)
			y_pred = np.random.normal(0, 1, size=100000)
			np.random.shuffle(y_true)

			dt.fit(y_true, y_pred)
			dt.save(io_file_path)
			# read_from_file
			df = pd.read_csv(io_file_path)
			if method == "equal_frequency":
				assert (df["样本数"] == 100000 / dt.bin_number).all()
			elif method == "equal_width":
				max_value, min_value = y_pred.max(), y_pred.min()
				interval = (max_value - min_value) / dt.bin_number
				left = float(df["区间"].iloc[0].strip("(]").split(', ')[0])
				right = float(df["区间"].iloc[0].strip("(]").split(', ')[1])
				assert left <= min_value
				np.testing.assert_almost_equal(right, min_value + interval, decimal=2)
			else:
				raise NotImplementedError("test failed.")

	def test_threshold_cutter_by_value(self):
		io_file_path = "/opt/checkpoints/ks_plot.csv"

		y = [1] * 100 + [0] * 400 + [1] * 400 + [0] * 100
		p = np.arange(0.5, 1, 0.0005)

		tc = ThresholdCutter(io_file_path)
		tc.cut_by_value(y, p)
		np.testing.assert_almost_equal(tc.bst_score, 0.6, decimal=3)
		np.testing.assert_almost_equal(tc.bst_threshold, 0.75, decimal=3)
		tc.save()

		df = pd.read_csv(io_file_path)
		assert (df["ks"] >= 0).all()
		assert (df["ks"] <= 1.0).all()

	def test_threshold_cutter_by_index(self):
		y = [1] * 100 + [0] * 400 + [1] * 400 + [0] * 100
		p = np.arange(0.5, 1, 0.0005)

		tc = ThresholdCutter()
		tc.cut_by_index(y, p)
		np.testing.assert_almost_equal(tc.bst_score, 0.6, decimal=3)
		np.testing.assert_almost_equal(tc.bst_threshold, 0.75, decimal=3)

