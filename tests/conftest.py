"""
Global test configuration for schollab-pipeline.

Sets up sys.path and mocks environment-specific packages (caiman, wx,
tqdm, psutil) that are not available in the base Python environment.
Packages that *may* be installed (torch, tifffile, h5py) are mocked
only when absent, so tests that require the real implementation can
skip via the HAS_* flags.
"""
import sys
import os
from unittest.mock import MagicMock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _subdir in ('caiman', 'fast', 'tools'):
    _p = os.path.join(REPO_ROOT, _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _real(name: str) -> bool:
    """True if *name* is genuinely installed (not already a MagicMock)."""
    if name in sys.modules:
        return not isinstance(sys.modules[name], MagicMock)
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_TORCH    = _real('torch')
HAS_TIFFFILE = _real('tifffile')
HAS_H5PY     = _real('h5py')
HAS_NUMPY    = _real('numpy')

# Always mock — environment-specific, never present in base Python
_ALWAYS_MOCK = [
    'caiman', 'caiman.motion_correction', 'caiman.source_extraction',
    'caiman.source_extraction.cnmf', 'caiman.source_extraction.cnmf.cnmf',
    'caiman.source_extraction.cnmf.params', 'caiman.utils',
    'caiman.utils.utils', 'caiman.summary_images',
    'cv2',
    'wx', 'wx.lib', 'wx.lib.agw', 'wx.lib.agw.multidirdialog',
    'tqdm',
    'psutil',
    'scipy', 'scipy.ndimage', 'scipy.ndimage.filters',
    'skimage', 'skimage.io',
]
for _m in _ALWAYS_MOCK:
    sys.modules.setdefault(_m, MagicMock())

# Mock optional deps only when absent so modules can always be imported
if not HAS_TORCH:
    for _m in [
        'torch', 'torch.nn', 'torch.nn.functional', 'torch.autograd',
        'torch.cuda', 'torch.backends', 'torch.backends.cudnn',
        'torch.utils', 'torch.utils.data',
    ]:
        sys.modules.setdefault(_m, MagicMock())

if not HAS_TIFFFILE:
    sys.modules.setdefault('tifffile', MagicMock())

if not HAS_H5PY:
    sys.modules.setdefault('h5py', MagicMock())
