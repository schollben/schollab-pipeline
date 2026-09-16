"""
Standard-deviation projection TIFF after motion correction.

Why: operators want a compact movie of activity (STD over time) without
opening the full registered stack. Each output frame is the STD of 1000
motion-corrected frames. This file is a CaImAn artifact, not acquisition data.

Never call this unless motion correction actually ran — TIFs→H5-only and
skip_caiman must not produce std_projection.tif.
"""
import os
import numpy as np

# Window matches the existing registered.h5 rewrite chunk so STD can be
# computed from frames already in RAM (no second pass over the mmap).
STD_WINDOW = 1000
STD_TIFF_NAME = 'std_projection.tif'


def is_pipeline_artifact_tif(path_or_name):
	"""
	True for CaImAn preview / STD TIFFs that must not be treated as source data.

	Why: std_projection.tif and 01_rigid.tif would otherwise be picked up by
	TIFs→H5 and by the 'has source TIFFs?' clean-safety check.
	"""
	name = os.path.basename(path_or_name)
	if name == STD_TIFF_NAME:
		return True
	return name.endswith('_rigid.tif') or name.endswith('_nonrigid.tif')


def std_of_window(chunk):
	"""STD projection of one (T, Y, X) window → (Y, X) float32."""
	return np.std(np.asarray(chunk, dtype=np.float64), axis=0).astype(np.float32)


def std_projection_stack(movie, window=STD_WINDOW):
	"""
	Stack of STD projections, one per complete window along axis 0.

	Remainder frames (n % window) are dropped from this TIFF only so every
	STD page is exactly `window` images. Those leftover frames are still
	motion-corrected and stored in registered.h5 — this helper never truncates
	the movie.
	"""
	movie = np.asarray(movie)
	n_windows = movie.shape[0] // window
	if n_windows == 0:
		return np.empty((0,) + movie.shape[1:], dtype=np.float32)
	out = np.empty((n_windows,) + movie.shape[1:], dtype=np.float32)
	for i in range(n_windows):
		out[i] = std_of_window(movie[i * window:(i + 1) * window])
	return out


def write_std_projection_tiff(parent_dir, std_frames, tiff_name=STD_TIFF_NAME):
	"""
	Write one TIFF stack next to other session artifacts (registered.h5, …).

	Returns the output path, or None when there are no complete windows.
	"""
	if std_frames is None or len(std_frames) == 0:
		print(
			f"  STD projection TIFF skipped — fewer than {STD_WINDOW} "
			"motion-corrected frames."
		)
		return None
	import tifffile
	from tiff_compat import tiff_writer_append
	path = os.path.join(parent_dir, tiff_name)
	print(
		f"  Writing STD projection TIFF: {path} "
		f"({len(std_frames)} frames, window={STD_WINDOW})"
	)
	with tifffile.TiffWriter(path, bigtiff=False, imagej=False) as tif:
		for frame in std_frames:
			tiff_writer_append(
				tif, np.asarray(frame, dtype=np.float32), contiguous=False
			)
	print(f"  Wrote STD projection TIFF: {path}")
	return path
