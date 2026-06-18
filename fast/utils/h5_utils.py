import h5py
import tifffile
import numpy as np
import os
import re
from glob import glob
from tqdm import tqdm

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


def _scanimage_tiffs_one_cycle(paths, cycle_re):
    """Same logic as caiman/tif_to_h5._scanimage_tiffs_one_cycle."""
    from collections import defaultdict

    groups = defaultdict(list)
    for f in paths:
        m = cycle_re.match(os.path.basename(f))
        if m:
            groups[m.group(1)].append(f)
    if not groups:
        return sorted(paths)
    if len(groups) == 1:
        return sorted(next(iter(groups.values())))
    best = max(groups.keys(), key=lambda k: (len(groups[k]), k))
    return sorted(groups[best])


def _acquisition_tif_paths(sorted_paths):
    """
    Same rules as caiman/tif_to_h5._acquisition_tif_paths (FAST env cannot import caiman/).
    """
    out = []
    for f in sorted_paths:
        b = os.path.basename(f)
        if b.endswith('_rigid.tif') or b.endswith('_nonrigid.tif'):
            continue
        out.append(f)
    assert len(out) > 0, (
        "No acquisition TIFFs left after excluding *_rigid.tif / *_nonrigid.tif — "
        "folder may contain only CaImAn sample exports."
    )
    tseries = [f for f in out if os.path.basename(f).startswith('TSeries_')]
    file_pref = [f for f in out if os.path.basename(f).startswith('file_')]
    if len(tseries) > 0:
        return _scanimage_tiffs_one_cycle(tseries, re.compile(r'^TSeries_(\d+)_'))
    if len(file_pref) > 0:
        return _scanimage_tiffs_one_cycle(file_pref, re.compile(r'^file_(\d+)_'))
    return out


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


def tif_stacks_to_h5(tif_dir, h5_savename, h5_key='mov', delete_tiffs=False, frame_offset=False, offset=30):
    """
    Convert .tif stacks to a monolithic .h5 file.

    Parameters:
        tif_dir (str): Directory containing .tif files to convert.
        h5_savename (str): Output .h5 file path.
        h5_key (str): h5 key to save data under. CaImAn assumes 'mov'.
        delete_tiffs (bool): Whether to remove .tif files during conversion.
        frame_offset (bool): Whether to add flipped frame offsets at beginning and end.
        offset (int): Number of frames to add at beginning and end if frame_offset is True.
    """
    tif_fnames = sort_tif_stack_paths(glob(os.path.join(tif_dir, "*.tif")))
    tif_fnames = _acquisition_tif_paths(tif_fnames)
    # Re-sort chunk stacks after acquisition filter (100+ chunks need numeric order).
    tif_fnames = sort_tif_stack_paths(tif_fnames)
    assert len(tif_fnames) > 0, f"No .tif files found in {tif_dir}"

    first_tif = tifffile.imread(tif_fnames[0])
    if len(first_tif.shape) < 3:
        stack_depth = 1
        stack_width, stack_height = first_tif.shape
    else:
        stack_depth, stack_width, stack_height = first_tif.shape

    # All multi-frame stacks should be same size except for the last.
    if stack_depth > 1:
        for i in range(1, len(tif_fnames) - 1):
            tif_stack_handle = tifffile.TiffFile(tif_fnames[i])
            this_stack_depth = len(tif_stack_handle.pages)
            assert this_stack_depth == stack_depth, \
                f"Stack sizes inconsistent: expected {stack_depth} frames " \
                f"but got {this_stack_depth} for file {tif_fnames[i]}"

    if frame_offset:
        first_frames = np.zeros((offset, stack_width, stack_height))
        last_frames = np.zeros((offset, stack_width, stack_height))

        if stack_depth == 1:
            assert len(tif_fnames) > 2 * offset, \
                f"Frame offset is True for {tif_dir}, and offset is {offset}. " \
                f"Insufficient frames ({len(tif_fnames)})"
            last_fnames = tif_fnames[(-1 * offset):]
            for i in range(offset):
                first_frames[i, :, :] = tifffile.imread(tif_fnames[i], is_ome=False)
                last_frames[i, :, :] = tifffile.imread(last_fnames[i], is_ome=False)
            last_stack_length = 1
        else:
            first_stack = tifffile.imread(tif_fnames[0])
            last_stack = tifffile.imread(tif_fnames[-1])
            first_frames[:, :, :] = first_stack[0:offset, :, :]
            if last_stack.shape[0] < offset:
                second_last_stack = tifffile.imread(tif_fnames[-2])
                both_last = np.concatenate((second_last_stack, last_stack), axis=0)
                last_frames = both_last[(-1 * offset):, :, :]
            else:
                last_frames[:, :, :] = last_stack[(-1 * offset):, :, :]
            last_stack_length = last_stack.shape[0]
            del first_stack
            del last_stack
    else:
        last_stack = tifffile.imread(tif_fnames[-1])
        if len(last_stack.shape) < 3:
            last_stack_length = 1
        else:
            last_stack_length = last_stack.shape[0]

    # Calculate total output frames
    if frame_offset:
        out_data_frames = (offset * 2) + (stack_depth * (len(tif_fnames) - 1)) + last_stack_length
    else:
        out_data_frames = (stack_depth * (len(tif_fnames) - 1)) + last_stack_length

    f_out = h5py.File(h5_savename, 'w')
    f_out.create_dataset(h5_key, (out_data_frames, stack_width, stack_height))
    write_start_ind, write_end_ind = (0, 0)

    if frame_offset:
        f_out[h5_key][0:offset, :, :] = np.flip(first_frames, axis=0)
        write_end_ind += offset

    for i in tqdm(range(len(tif_fnames) - 1), desc="Writing all but last stack...", ncols=75):
        this_stack_data = tifffile.imread(tif_fnames[i], is_ome=False)
        write_start_ind = write_end_ind
        write_end_ind = write_start_ind + stack_depth
        f_out[h5_key][write_start_ind:write_end_ind, :, :] = this_stack_data

    # Write the last stack
    last_stack = tifffile.imread(tif_fnames[-1], is_ome=False)
    write_start_ind = write_end_ind

    print('Writing last stack...')
    if frame_offset:
        f_out[h5_key][write_start_ind:-offset, :, :] = last_stack
        f_out[h5_key][-offset:, :, :] = np.flip(last_frames, axis=0)
    else:
        f_out[h5_key][write_start_ind:write_start_ind + last_stack_length, :, :] = last_stack

    f_out.close()
    print(f"Saved: {h5_savename}")
