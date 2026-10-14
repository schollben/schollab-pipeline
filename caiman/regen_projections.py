"""
Rebuild std_projection.tif / avg_projection.tif from an existing registered.h5.

Why: sessions motion-corrected before projections existed have registered.h5
but no STD/AVG TIFFs. Re-running MC just for those is hours of wasted I/O;
this reads registered.h5 once and writes both files next to it.

Usage (caiman env, from repo root):
	python caiman/regen_projections.py /mnt/nas/.../expt1 [more folders...]
	python caiman/regen_projections.py --force /mnt/nas/.../expt1
"""
import argparse
import os
import sys

import h5py

from std_projection import (
	AVG_TIFF_NAME,
	STD_TIFF_NAME,
	projection_frames,
	write_avg_projection_tiff,
	write_std_projection_tiff,
)

REGISTERED_H5 = 'registered.h5'


def regen_folder(folder, force=False):
	"""Return True when both TIFFs exist afterwards (written or already there)."""
	h5_path = os.path.join(folder, REGISTERED_H5)
	if not os.path.isfile(h5_path):
		print(f"ERROR: {h5_path} not found — run motion correction first.")
		return False
	have = all(os.path.isfile(os.path.join(folder, n))
		for n in (STD_TIFF_NAME, AVG_TIFF_NAME))
	if have and not force:
		print(f"SKIP: {folder} already has both projection TIFFs (use --force).")
		return True
	print(f"Reading {h5_path}")
	with h5py.File(h5_path, 'r') as f:
		mov = f['mov']
		print(f"  registered.h5 mov: {mov.shape} {mov.dtype}")
		std_frames, avg_frames = projection_frames(mov, mov.shape[0])
	# Writers return None (and print why) when there is no full window.
	ok_std = write_std_projection_tiff(folder, std_frames) is not None
	ok_avg = write_avg_projection_tiff(folder, avg_frames) is not None
	return ok_std and ok_avg


def main(argv=None):
	p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	p.add_argument('folders', nargs='+', help='session folder(s) containing registered.h5')
	p.add_argument('--force', action='store_true', help='overwrite existing projection TIFFs')
	args = p.parse_args(argv)
	failed = [d for d in args.folders if not regen_folder(d, args.force)]
	if failed:
		print(f"FAILED ({len(failed)}): " + ', '.join(failed))
		return 1
	return 0


if __name__ == '__main__':
	sys.exit(main())
