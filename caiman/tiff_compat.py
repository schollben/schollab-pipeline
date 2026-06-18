"""
tifffile.TiffWriter compatibility.

Older conda releases expose only save(); current tifffile uses write().
CaImAn envs may pin the legacy API, so pipeline code must support both.

Why this module exists: avoid 'TiffWriter' object has no attribute 'write' at runtime.

Chunk stack helpers below are mirrored in fast/utils/h5_utils.py (separate conda env).
"""

import os


def chunk_index_pad_width(num_chunks):
	"""
	Digit width for chunk suffixes so lexicographic sort matches frame order.

	Why: registered_100 sorted before registered_11 with fixed :02d padding (100+ chunks).
	"""
	return max(2, len(str(num_chunks)))


def format_chunk_tif_name(base_name, chunk_number, num_chunks):
	"""Build a chunk TIFF name with enough zero-padding for num_chunks stacks."""
	width = chunk_index_pad_width(num_chunks)
	return f"{base_name}_{chunk_number:0{width}d}.tif"


def chunk_index_from_basename(name):
	"""
	Parse trailing _NNN chunk index from a TIFF basename, or None if not a chunk stack.

	Why: timestamp-prefixed inference outputs (202506181430_registered_100.tif) and
	ScanImage names (file_00001_ch2.tif) must not be mis-parsed as chunk stacks.
	"""
	stem = name
	for ext in ('.tiff', '.tif'):
		if stem.lower().endswith(ext):
			stem = stem[:-len(ext)]
			break
	if '_' not in stem:
		return None
	suffix = stem.rsplit('_', 1)[-1]
	if suffix.isdigit():
		return int(suffix)
	return None


def sort_tif_stack_paths(paths):
	"""
	Sort H5-export or inference TIFF stacks by trailing chunk index.

	Falls back to lexicographic sort when paths are not all chunk stacks.
	"""
	indexed = [(chunk_index_from_basename(os.path.basename(p)), p) for p in paths]
	if indexed and all(idx is not None for idx, _ in indexed):
		return [p for _, p in sorted(indexed, key=lambda item: item[0])]
	return sorted(paths)


def tiff_writer_append(writer, frame, contiguous=False):
	"""
	Append one 2D plane to an open TiffWriter.

	Parameters:
		writer: context-managed tifffile.TiffWriter instance
		frame: 2D array (single TIFF page)
		contiguous: if True, pass contiguous=True when the API accepts it
	"""
	if hasattr(writer, 'write'):
		if contiguous:
			writer.write(frame, contiguous=True)
		else:
			writer.write(frame)
		return
	# Legacy tifffile: save() instead of write()
	if contiguous:
		try:
			writer.save(frame, contiguous=True)
		except TypeError:
			writer.save(frame)
	else:
		writer.save(frame)
