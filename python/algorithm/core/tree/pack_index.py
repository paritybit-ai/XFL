import numpy as np


def pack_index(index_list: list[int]) -> np.ndarray:
    bit_array = [0] * (max(index_list) + 1)
    for i in index_list:
        bit_array[i] = 1
    res = np.packbits(bit_array)
    return res


def unpack_index(packed_index: np.ndarray) -> list[int]:
    bit_array = np.unpackbits(packed_index)
    res = list(np.where(bit_array == 1)[0])
    return res
