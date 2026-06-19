import h5py
import tifffile
import numpy as np
import os
from glob import glob

# Keep chunk helpers aligned with caiman/tiff_compat.py (FAST env cannot import caiman/).
def chunk_index_pad_width(num_chunks):
    """Digit width for chunk suffixes so lex sort matches frame order (>=100 chunks)."""
    return max(2, len(str(num_chunks)))


def format_chunk_tif_name(base_name, chunk_number, num_chunks):
    """Build a chunk TIFF name with enough zero-padding for num_chunks stacks."""
    width = chunk_index_pad_width(num_chunks)
    return f"{base_name}_{chunk_number:0{width}d}.tif"


def chunk_index_from_basename(name):
    """
    Parse trailing _NNN chunk index from a TIFF basename, or None if not a chunk stack.
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
    """Sort chunk TIFF stacks by trailing index; lex sort fallback for other names."""
    indexed = [(chunk_index_from_basename(os.path.basename(p)), p) for p in paths]
    if indexed and all(idx is not None for idx, _ in indexed):
        return [p for _, p in sorted(indexed, key=lambda item: item[0])]
    return sorted(paths)


# Keep aligned with caiman/tiff_compat.py (FAST env may use older tifffile).
def _tiff_writer_append(writer, frame, contiguous=False):
    if hasattr(writer, 'write'):
        if contiguous:
            writer.write(frame, contiguous=True)
        else:
            writer.write(frame)
        return
    if contiguous:
        try:
            writer.save(frame, contiguous=True)
        except TypeError:
            writer.save(frame)
    else:
        writer.save(frame)


def h5_to_tiff(h5_path, max_frames=None, chunk_size=5000, output_dir=None):
    """
    Convert H5 file to TIFF stacks in chunks.

    Parameters:
        h5_path (str): Path to the H5 file
        max_frames (int): Maximum number of frames to save (default: all frames)
        chunk_size (int): Number of frames per TIFF stack (default: 5000)
        output_dir (str): Directory to save TIFFs (default: same dir as h5 file)
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return

    base_dir = os.path.dirname(h5_path)
    base_name = os.path.basename(h5_path)
    base_name = base_name.replace('.h5', '').replace('.hdf5', '')

    save_dir = output_dir if output_dir else base_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Converting {h5_path} to TIFF stacks (chunks of {chunk_size} frames)")

    with h5py.File(h5_path, 'r') as datafile:
        if 'mov' in datafile.keys():
            dataset = datafile['mov']
        else:
            dataset = datafile[list(datafile.keys())[0]]

        numframes = dataset.shape[0]
        print(f"Found {numframes} frames in H5 file")

        frames_to_save = min(max_frames, numframes) if max_frames else numframes
        num_chunks = int(np.ceil(frames_to_save / chunk_size))
        print(f"Will create {num_chunks} TIFF stack(s)")

        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, frames_to_save)
            chunk_frames = end_frame - start_frame

            chunk_output = os.path.join(
                save_dir, format_chunk_tif_name(base_name, chunk_idx + 1, num_chunks)
            )
            print(f"\nChunk {chunk_idx+1}/{num_chunks}: Saving frames {start_frame} to {end_frame-1} to {chunk_output}")

            with tifffile.TiffWriter(chunk_output, bigtiff=False, imagej=True) as tif:
                for i in range(start_frame, end_frame):
                    if (i - start_frame) % 1000 == 0:
                        print(f"  Processing frame {i - start_frame}/{chunk_frames}...")
                    curfr = dataset[i, :, :].astype(np.int16)
                    _tiff_writer_append(tif, curfr, contiguous=True)

            print(f"  Successfully saved {chunk_frames} frames to {chunk_output}")

        print(f"\nConversion complete! Created {num_chunks} TIFF stack(s)\n")
