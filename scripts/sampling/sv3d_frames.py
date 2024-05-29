import os

import cv2
import numpy as np
from PIL import Image

path = '/home/yifanyang/container_us/zyhe/muaz/output/sv3d_u/fm_bm_tm_compression/no_compression'
save_path='output/gt/'
os.makedirs(save_path, exist_ok=True)

images = [2, 4, 7, 22, 24, 29]
# find all the gifs in the path
import glob

gifs = glob.glob(path + '/*.gif')

# check if any gifs in gifs start with string i where i in images, and save those gif names
images = [os.path.basename(gif) for gif in gifs if any(gif.split('/')[-1].startswith(str(i)+'_') for i in images)]
print(images)

for image in images:
# Open the GIF
    gif_path = os.path.join(path, image)
    img = Image.open(gif_path)

    # Extract each frame and ensure it's in RGB
    frames = []
    try:
        while True:
            # Ensure the frame is in RGB
            # rgb_frame = img.convert('RGB')
            # # Copy the frame and append its array representation to the list
            # frames.append(np.array(rgb_frame.copy()))
            # img.seek(img.tell() + 1)

            img.seek(img.tell())
            rgb_frame = img.convert('RGBA')  # Use 'RGBA' to handle transparency

            # Handle transparency by blending with a white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(rgb_frame, mask=rgb_frame.split()[3])  # 3 is the alpha channel

            # Copy the frame and append its array representation to the list
            frames.append(np.array(background))
            img.seek(img.tell() + 1)

    except EOFError:
        pass  # End of sequence

    gt_frames = frames[0]
    # gt_frames = Image.fromarray(gt_frames)
    # gt_frames.save(os.path.join(path, '4_0.png'))
    # exit(0)

    rows = 3
    cols = 7

    frame_array = np.stack(frames, axis=0)
    # Reshape to (rows, cols, frame height, frame width, channels)
    frame_grid = frame_array.reshape(rows, cols, frame_array.shape[1], frame_array.shape[2], frame_array.shape[3])

    # Concatenate frames in each row
    row_concat = [np.concatenate(frame_grid[i], axis=1) for i in range(rows)]
    # Concatenate all rows
    full_grid = np.concatenate(row_concat, axis=0)

    # Use PIL to save the array as an image
    final_image = Image.fromarray(full_grid)
    final_image.save(os.path.join(save_path, image.replace('.gif', '.png')))