"""
tifffile.TiffWriter compatibility.

Older conda releases expose only save(); current tifffile uses write().
CaImAn envs may pin the legacy API, so pipeline code must support both.

Why this module exists: avoid 'TiffWriter' object has no attribute 'write' at runtime.
"""


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
