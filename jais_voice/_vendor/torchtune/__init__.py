# Minimal torchtune stub (BSD-3-Clause, see attribution in modules/_rope.py).
#
# neucodec imports ``from torchtune.modules import RotaryPositionalEmbeddings``
# but the real torchtune package hard-imports torchao (plus torchdata/datasets
# via its package init). Installing the real stack is fragile on Colab.
# This vendored copy provides ONLY the class neucodec needs, with identical
# math (source copied from torchtune 0.6.1) so numerics match exactly.
#
# jais_voice.hub prepends this directory to sys.path before importing
# neucodec, so ``import torchtune`` resolves here.

__version__ = "0.6.1-stub"
