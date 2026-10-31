# BSD-3-Clause attribution:
#
# This module is copied verbatim (docstrings, math and all) from
# torchtune 0.6.1, file ``torchtune/modules/position_embeddings.py``,
# Meta Platforms, Inc. / PyTorch:
#   https://github.com/pytorch/torchtune/blob/v0.6.1/torchtune/modules/position_embeddings.py
#
# It exists because importing the real torchtune package requires torchao
# (and friends), while neucodec only ever uses this single class.

from ._rope import RotaryPositionalEmbeddings

__all__ = ["RotaryPositionalEmbeddings"]
