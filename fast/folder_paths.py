"""
Filesystem paths for one FAST session folder.

Separated from denoising.py so tests and tooling can import without torch/h5py.
"""
import os
from dataclasses import dataclass


@dataclass
class FolderPaths:
	"""
	All filesystem paths for a single data folder.

	Permanent outputs (checkpoint, inference.h5, sentinel) live on root.
	Intermediates (registered/, training/, result/) live under root/scratch/
	and are removed after each FAST run.
	"""
	root:         str
	scratch:      str
	h5:           str
	registered:   str
	training:     str
	result:       str
	checkpoint:   str
	inference_h5: str
	sentinel:     str

	@staticmethod
	def from_root(root: str) -> 'FolderPaths':
		"""Build FolderPaths; scratch is always {session}/scratch/."""
		root = root.rstrip('/')
		scratch = os.path.join(root, 'scratch')
		return FolderPaths(
			root         = root,
			scratch      = scratch,
			h5           = os.path.join(root, 'registered.h5'),
			registered   = os.path.join(scratch, 'registered'),
			training     = os.path.join(scratch, 'training'),
			result       = os.path.join(scratch, 'result'),
			checkpoint   = os.path.join(root, 'checkpoint'),
			inference_h5 = os.path.join(root, 'inference.h5'),
			sentinel     = os.path.join(root, '_fast_complete'),
		)
