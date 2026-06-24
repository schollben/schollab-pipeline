"""
Filesystem paths for one FAST session folder.

Separated from denoising.py so tests and tooling can import without torch/h5py.
"""
import os
import shutil
from dataclasses import dataclass


@dataclass
class FolderPaths:
	"""
	All filesystem paths for a single data folder.

	Permanent outputs (checkpoint, inference.h5, sentinel) live on root.
	Intermediates (registered/, training/, result/) live under root/scratch/
	and are removed at the start and end of each FAST run.
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


def remove_checkpoint_for_rerun(checkpoint_dir: str, skip_training: bool) -> bool:
	"""
	Delete checkpoint/ when starting a FAST re-run (skip_training is false).

	Returns True if the directory was removed. Kept in this module so unit tests
	do not need to import torch via denoising.py.
	"""
	if skip_training or not os.path.isdir(checkpoint_dir):
		return False
	shutil.rmtree(checkpoint_dir)
	return True


def remove_scratch_for_rerun(scratch_dir: str) -> bool:
	"""
	Delete scratch/ when starting a FAST re-run.

	Why: stale intermediates from a crashed or partial run must not leak into
	the new attempt. Always removed on re-run (no skip_training guard).
	"""
	if not os.path.isdir(scratch_dir):
		return False
	shutil.rmtree(scratch_dir)
	return True
