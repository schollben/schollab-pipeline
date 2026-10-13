"""
STD / average projection TIFFs after motion correction.

Why: operators want compact movies of activity (STD) and mean intensity
without opening the full registered stack. Each output frame is one
1000-frame window. These files are CaImAn artifacts, not acquisition data.

Never call this unless motion correction actually ran — TIFs→H5-only and
skip_caiman must not produce std_projection.tif or avg_projection.tif.
"""
import os
import numpy as np

# Window matches the existing registered.h5 rewrite chunk so projections can
# be computed from frames already in RAM (no second pass over the mmap).
STD_WINDOW = 1000
STD_TIFF_NAME = 'std_projection.tif'
AVG_TIFF_NAME = 'avg_projection.tif'


def is_pipeline_artifact_tif(path_or_name):
	"""
	True for CaImAn preview / projection TIFFs that must not be source data.

	Why: std/avg_projection.tif and 01_rigid.tif would otherwise be picked up
	by TIFs→H5 and by the 'has source TIFFs?' clean-safety check.
	"""
	name = os.path.basename(path_or_name)
	if name in (STD_TIFF_NAME, AVG_TIFF_NAME):
		return True
	return name.endswith('_rigid.tif') or name.endswith('_nonrigid.tif')


def std_of_window(chunk):
	"""STD projection of one (T, Y, X) window → (Y, X) float32."""
	return np.std(np.asarray(chunk, dtype=np.float64), axis=0).astype(np.float32)


def avg_of_window(chunk):
	"""Mean projection of one (T, Y, X) window → (Y, X) float32."""
	return np.mean(np.asarray(chunk, dtype=np.float64), axis=0).astype(np.float32)


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
	return _write_projection_tiff(parent_dir, std_frames, tiff_name, 'STD')


def write_avg_projection_tiff(parent_dir, avg_frames, tiff_name=AVG_TIFF_NAME):
	return _write_projection_tiff(parent_dir, avg_frames, tiff_name, 'AVG')


def _write_projection_tiff(parent_dir, frames, tiff_name, label):
	"""
	Write one TIFF stack next to other session artifacts (registered.h5, …).

	Returns the output path, or None when there are no complete windows.
	"""
	if frames is None or len(frames) == 0:
		print(
			f"  {label} projection TIFF skipped — fewer than {STD_WINDOW} "
			"motion-corrected frames."
		)
		return None
	import tifffile
	path = os.path.join(parent_dir, tiff_name)
	print(
		f"  Writing {label} projection TIFF: {path} "
		f"({len(frames)} frames, window={STD_WINDOW})"
	)
	_imwrite_plain_stack(tifffile, path, np.asarray(frames, dtype=np.float32))
	print(f"  Wrote {label} projection TIFF: {path}")
	return path


def _imwrite_plain_stack(tifffile, path, stack):
	# ome=False: this is a preview stack, not a Bruker series. OME-XML in
	# ImageDescription makes Fiji File>Open run ImageJ's TIFF reader and
	# Bio-Formats on the same pixels, so the stack appears twice.
	kwargs = dict(photometric='minisblack', imagej=False, metadata=None)
	try:
		tifffile.imwrite(path, stack, ome=False, **kwargs)
	except TypeError:
		tifffile.imwrite(path, stack, **kwargs)
