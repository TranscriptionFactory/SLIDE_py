"""
DSDP5 SDP Solver - Python interface.

Source: https://github.com/sburer/scikit-dsdp

Note: The C extension (.so file) is platform-specific and may need
      to be recompiled for different Python versions/platforms.
      See setup.py for build instructions.

License: GPL
"""

from .dsdp5 import dsdp

__all__ = ["dsdp"]
