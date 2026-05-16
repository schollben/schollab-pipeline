from glob import glob
import os
import tifffile
import numpy as np
import h5py
from tqdm import tqdm

def tif_stacks_to_h5(tif_dir, h5_savename, h5_key='mov', delete_tiffs=False, frame_offset=False, offset=30, channel='Ch2'):
    '''
    Convert .tif stacks from BRUKER/SCANIMAGE to monolithic .h5 files.
    If frame_offset, frames from the start and end of the series are appended to either
    end of the resulting .h5 and flipped. This is performed when data is meant to be
    given to DeepInterpolation.

        Parameters:
            tif_dir (str): String for path of directory to convert tifs in.
            h5_savename (str): Name of conversion .h5 file to save.
            h5_key (str): h5 key to save data under. CaImAn assumes 'mov'.
            delete_tiffs (bool): Whether to remove .tif files during conversion
            frame_offset (bool): Whether to add frame offsets at beginning and end
            offset (int): how many frames to add at beginning and end if frame_offset is True
            channel (str or None): Channel to select, e.g. 'Ch2'. Filters TIFFs whose
                filename contains this string. Pass None to include all channels.
        Returns:
            None - Writes .h5 file to disk at 'h5_savename' containing calcium movie data.
    '''
    all_tifs = sorted(glob(os.path.join(tif_dir, "*.tif")))
    if channel is not None:
        has_channel_token = any('Ch' in os.path.basename(f) for f in all_tifs)
        if has_channel_token:
            tif_fnames = [f for f in all_tifs if f'_{channel}_' in os.path.basename(f)]
        else:
            tif_fnames = all_tifs
    else:
        tif_fnames = all_tifs
    assert len(tif_fnames) > 0, f"No TIF files found in {tif_dir}" + (f" for channel '{channel}'" if channel else "") + "."
    print(f"\nFound {len(tif_fnames)} TIFF stacks to write ({channel or 'all channels'}):")
    for f in tif_fnames:
        print(f"  {os.path.basename(f)}")
    print()
    # Don't know how to get width, height without loading into memory.
    #first_tif_handle = tifffile.TiffFile(tif_fnames[0])
    #stack_depth = len(first_tif_handle.pages)
    with tifffile.TiffFile(tif_fnames[0]) as tif:
        stack_depth = len(tif.pages)
        page = tif.pages[0]
        stack_width, stack_height = page.shape[:2]


    # All multi-frame .tif stacks should be same size except for the last.
    # Quick check w/out loading into RAM.
    if stack_depth > 1:
        for i in range(1, len(tif_fnames) - 1):
            tif_stack_handle = tifffile.TiffFile(tif_fnames[i])
            this_stack_depth = len(tif_stack_handle.pages)
            assert this_stack_depth == stack_depth, \
                f"Stack sizes inconsistent : expected {stack_depth} frames \
                    but got {this_stack_depth} for file {tif_fnames[i]}"
    
    # Adding offsets, flipped to front and back of data.
    if frame_offset: 
        first_frames = np.zeros((offset, stack_width, stack_height))
        last_frames  = np.zeros((offset, stack_width, stack_height))
        
        if stack_depth == 1: # .ome.tifs
            # Check for sufficient frames to prevent overlap of frames
            assert len(tif_fnames) > 2 * offset, f"Frame offset is True for {tif_dir}, and offset is {offset}. Insufficient frames ({len(tif_fnames)})"
            last_fnames  = tif_fnames[(-1*offset):]
            for i in range(offset):
                first_frames[i,:,:] = tifffile.imread(tif_fnames[i], is_ome=False)
                last_frames[i,:,:]  = tifffile.imread(last_fnames[i], is_ome=False)
            last_stack_length = 1
        
        else: # multi-page .tifs
            first_stack = tifffile.imread(tif_fnames[0])
            last_stack  = tifffile.imread(tif_fnames[-1])
            
            first_frames[:,:,:] = first_stack[0:offset, :, :]

            if last_stack.shape[0] < offset:
                #raise IndexError(f'Frame flip offset of {offset} exceeds \
                #                 last stack len {last_stack.shape[0]}. Write handling for this.')
                second_last_stack = tifffile.imread(tif_fnames[-2])
                both_last = np.concatenate((second_last_stack, last_stack), axis=0)
                last_frames = both_last[(-1 * offset), :, :]
            else:    
                last_frames[:,:,:] = last_stack[(-1 * offset):, : ,:]
            last_stack_length = last_stack.shape[0]
            del first_stack
            del last_stack
    # Just to guarantee last_stack_length
    else:
        last_stack = tifffile.imread(tif_fnames[-1])
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
        print(f"  Writing [{i+1}/{len(tif_fnames)}]: {os.path.basename(tif_fnames[i])}")
        this_stack_data = tifffile.imread(tif_fnames[i], is_ome=False)
        write_start_ind = write_end_ind
        write_end_ind = write_start_ind + stack_depth
        f_out[h5_key][write_start_ind:write_end_ind, :, :] = this_stack_data

    # Write the last stack
    last_stack = tifffile.imread(tif_fnames[-1], is_ome=False)
    write_start_ind = write_end_ind

    print(f"  Writing [{len(tif_fnames)}/{len(tif_fnames)}]: {os.path.basename(tif_fnames[-1])} (last stack)")
    if frame_offset:
        f_out[h5_key][write_start_ind:-offset, :, :] = last_stack
        f_out[h5_key][-offset:, :, :] = np.flip(last_frames, axis=0)
    else:
        f_out[h5_key][write_start_ind:write_start_ind + last_stack_length, :, :] = last_stack

    f_out.close()