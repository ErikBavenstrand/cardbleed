"""Extend the borders of card scans outward for printing.

Continues the existing border pattern (holo speckle, solid colors, ...)
uniformly on all four edges without ever degrading the original image data:

  PNG  -> PNG   original pixels bit-identical (lossless re-serialize)
  WebP -> WebP  written lossless; decoded original pixels preserved exactly
  JPEG -> JPEG  DCT-domain surgery: original coefficient blocks are copied
                bit-exact into a larger grid; only new border blocks are
                encoded (with the original's own quantization tables)

Python API (stable surface):

    from cardbleed import Params, extend_image

    result = extend_image(arr, (16, 16, 16, 16), Params(),
                          np.random.default_rng(0), overwrite=True, notes=[])
"""

from ._version import __version__
from .cli import cli
from .errors import FileError
from .synthesis import Params, extend_image

__all__ = ["FileError", "Params", "__version__", "cli", "extend_image"]
