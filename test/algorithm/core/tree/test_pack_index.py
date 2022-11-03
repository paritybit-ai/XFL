
from algorithm.core.tree.pack_index import pack_index, unpack_index


def test_pack_index():
    a = [3, 6, 1]
    res = pack_index(a)
    res = unpack_index(res)
    assert res == sorted(a)