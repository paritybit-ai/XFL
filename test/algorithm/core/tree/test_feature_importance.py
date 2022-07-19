from algorithm.core.tree.feature_importance import FeatureImportance


class TestFeatureImportance(object):
	def test_split_importance(self):
		fi1 = FeatureImportance(0.1, main_type='gain')
		fi2 = FeatureImportance(0.2, main_type='gain')
		fi3 = FeatureImportance(0, 1, main_type='split')
		fi4 = FeatureImportance(0, 2, main_type='split')
		assert fi1.get() == 0.1
		assert fi3.get() == 1
		assert fi1 == fi1
		assert fi1 < fi2
		assert fi3 < fi4
		assert fi3 == fi3
		# test add
		fi5 = fi3 + fi4
		assert fi5.importance_split == 3
		fi3.add_split(1)
		assert fi3.get() == 2
		fi3.add_gain(0.1)
		assert fi3.importance_gain == 0.1
		assert "{}".format(fi3) == "importance: 2"
		assert "{}".format(fi1) == "importance: 0.1"

		fi_list = [fi1, fi2]
		fi_list = sorted(fi_list, reverse=True)
		assert fi_list[0] == fi2
		fi_list = [fi3, fi4]
		fi_list = sorted(fi_list)
		assert fi_list[0] == fi3
