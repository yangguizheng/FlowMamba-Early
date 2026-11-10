"""Byte masks and the parsing of packet strings; no GPU needed for the mask tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import masks  # noqa: E402


def test_masks():
    masks.test()


def test_packet_strings_are_cut_and_padded():
    from run_flowmamba_early import string_list_to_arr
    out = string_list_to_arr(["1 2 3", "", " ".join(["7"] * 400)], num_string=4, string_len=5)
    assert out == [1, 2, 3, 0, 0] + [0] * 5 + [7] * 5 + [0] * 5
