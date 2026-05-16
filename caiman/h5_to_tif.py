import h5py
import tifffile
import numpy as np
import os
import glob
import sys


def h5_to_tiff(h5_path, max_frames=None, chunk_size=5000, output_dir=None):
    """
    Convert H5 file to TIFF stacks in chunks using the same method as image_registration.py

    Parameters:
        h5_path (str): Path to the H5 file
        max_frames (int): Maximum number of frames to save (default: all frames)
        chunk_size (int): Number of frames per TIFF stack (default: 5000)
        output_dir (str): Directory to save TIFFs (default: same dir as h5 file)
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return

    # Get base name and directory
    base_dir = os.path.dirname(h5_path)
    base_name = os.path.basename(h5_path)
    base_name = base_name.replace('.h5', '').replace('.hdf5', '')

    save_dir = output_dir if output_dir else base_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Converting {h5_path} to TIFF stacks (chunks of {chunk_size} frames)")

    # Open H5 file
    with h5py.File(h5_path, 'r') as datafile:
        # Get the dataset (usually 'mov' or first key)
        if 'mov' in datafile.keys():
            dataset = datafile['mov']
        else:
            dataset = datafile[list(datafile.keys())[0]]

        numframes = dataset.shape[0]
        print(f"Found {numframes} frames in H5 file")

        # Limit frames if specified
        frames_to_save = min(max_frames, numframes) if max_frames else numframes

        # Calculate number of chunks
        num_chunks = int(np.ceil(frames_to_save / chunk_size))
        print(f"Will create {num_chunks} TIFF stack(s)")

        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, frames_to_save)
            chunk_frames = end_frame - start_frame

            # Create output filename with chunk number
            chunk_output = os.path.join(save_dir, f"{base_name}_{chunk_idx+1:02d}.tif")

            print(f"\nChunk {chunk_idx+1}/{num_chunks}: Saving frames {start_frame} to {end_frame-1} to {chunk_output}")

            # Save chunk as TIFF using TiffWriter (same as image_registration.py lines 80-84)
            with tifffile.TiffWriter(chunk_output, bigtiff=False, imagej=True) as tif:
                for i in range(start_frame, end_frame):
                    if (i - start_frame) % 1000 == 0:
                        print(f"  Processing frame {i - start_frame}/{chunk_frames}...")
                    curfr = dataset[i, :, :].astype(np.int16)
                    tif.write(curfr, contiguous=True)

            print(f"  Successfully saved {chunk_frames} frames to {chunk_output}")

        print(f"\nConversion complete! Created {num_chunks} TIFF stack(s)\n\n")


def convert_directory(directory):
    """
    Convert all H5 files in a directory to TIFF stacks

    Parameters:
        directory (str): Directory containing H5 files
    """
    h5_files = glob.glob(os.path.join(directory, "*.h5"))
    h5_files += glob.glob(os.path.join(directory, "*.hdf5"))

    if len(h5_files) == 0:
        print(f"No H5 files found in {directory}")
        return

    print(f"Found {len(h5_files)} H5 file(s) to convert")

    for h5_path in h5_files:
        h5_to_tiff(h5_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert single file:    python h5_to_tif.py <h5_file> [max_frames] [chunk_size]")
        print("  Convert directory:      python h5_to_tif.py <directory>")
        print("\nExamples:")
        print("  python h5_to_tif.py registered.h5")
        print("  python h5_to_tif.py registered.h5 20000")
        print("  python h5_to_tif.py registered.h5 20000 5000")
        print("  python h5_to_tif.py /path/to/directory")
        print("\nOutput:")
        print("  Creates numbered TIFF stacks: registered_01.tif, registered_02.tif, etc.")
        print("  Default chunk size: 8,000 frames per stack")
        sys.exit(1)

    input_path = sys.argv[1]

    # Check if input is a directory or file
    if os.path.isdir(input_path):
        convert_directory(input_path)
    elif os.path.isfile(input_path):
        max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
        chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        h5_to_tiff(input_path, max_frames, chunk_size)
    else:
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
