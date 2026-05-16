import datetime
import glob
import os
import logging

import numpy as np
import torch
from models.Unet_Lite import Unet_Lite
from tqdm import tqdm
from skimage import io


def _load_single_tif(filepath):
    """Load a single TIF stack and return as a float tensor."""
    image = io.imread(filepath)
    return torch.from_numpy(image / 1.0).float()


def _prepare_test_input(image_tensor):
    """Prepare a single test image: pad with flipped tail frames, return (input, original_t)."""
    t = image_tensor.shape[0]
    # Temporal augmentation: append flipped last 100 frames
    padded = torch.cat((image_tensor, image_tensor.flip(0)[:100, :, :]), dim=0)
    input_tensor = padded.unsqueeze(0).unsqueeze(0)  # (1, 1, T+pad, H, W)
    return input_tensor, t


def goTesting(args):
    logging.info('Testing Start!!!')

    # Use args.results_dir for output location
    testSave_dir = os.path.join(args.results_dir, 'result')
    os.makedirs(testSave_dir, exist_ok=True)

    output_channels = args.miniBatch_size

    # Load model
    model = Unet_Lite(
        in_channels=args.miniBatch_size,
        out_channels=output_channels,
        f_maps=[64, 64, 64],
        num_groups=32,
        final_sigmoid=True
    ).cuda()

    model.cuda(args.local_rank)

    if args.local_rank == 0:
        model_path = args.checkpoint_path
        if not os.path.exists(model_path):
            logging.error(f"Model checkpoint '{model_path}' not found.")
            return

        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # Discover TIF files in test directory
    tif_files = sorted(
        glob.glob(os.path.join(args.test_path, '*.tif'))
        + glob.glob(os.path.join(args.test_path, '*.tiff'))
    )

    if not tif_files:
        logging.error(f"No TIF files found in {args.test_path}")
        return

    print(f"Found {len(tif_files)} TIF stack(s) to process")
    pbar = tqdm(total=len(tif_files), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    with torch.no_grad():
        for filepath in tif_files:
            filename = os.path.basename(filepath)
            print(f"Loading stack: {filename}")

            # Load one stack at a time
            image_tensor = _load_single_tif(filepath)
            print(f"  shape: {image_tensor.shape}")

            input_tensor, t = _prepare_test_input(image_tensor)
            del image_tensor  # free raw image memory

            mean = torch.mean(input_tensor)
            input_tensor = input_tensor - mean
            input_tensor = input_tensor[0].cuda(args.local_rank, non_blocking=True)
            output = torch.zeros(input_tensor.shape)

            miniBatch = args.miniBatch_size
            timeFrame = input_tensor.shape[1]
            assert miniBatch <= timeFrame, "miniBatch size must <= time frame length !!!"

            for j in range(0, timeFrame):
                if j + miniBatch <= timeFrame:
                    input_batch = input_tensor[:, j:j + miniBatch, ...]
                    output_p = model(input_batch).cpu().detach()
                    if j == 0:
                        output[:, j:j + output_channels, ...] = output_p
                    else:
                        output[:, j + output_channels - 1, ...] = output_p[:, -1, ...]
                        output[:, j:j + output_channels - 1, ...] = 0.5 * (output_p[:, :-1, ...] + output[:, j:j + output_channels - 1, ...])

            output = output[:, :t, ...] + mean
            output_image = np.squeeze(output.numpy()) * 1.0
            del input_tensor, output  # free GPU/CPU memory

            result_name = os.path.join(testSave_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{filename}")
            output_image = np.clip(output_image, 0, 65535)
            io.imsave(result_name, output_image.astype(np.uint16), check_contrast=False)
            del output_image

            torch.cuda.empty_cache()

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(f'Testing...\t\t{mem}')
            pbar.update(1)

    pbar.close()
    print("Saved: " + testSave_dir)
    print("Test End")

