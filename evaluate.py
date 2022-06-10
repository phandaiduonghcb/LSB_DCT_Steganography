import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from sewar import mse, ssim, psnr

original_im_paths = glob.glob('original_images/*')
tiff_png_paths = glob.glob('encoded_images/GIMP_tiff_to_png/*')
lsb_paths = glob.glob('encoded_images/*.png')

for original_im_path in original_im_paths:
    origninal_im = plt.imread(original_im_path)*255
    origninal_im = origninal_im.astype(np.uint8)

    img_name = os.path.basename(original_im_path).split('.')[0]

    print('### '+original_im_path)
    for encoded_path in tiff_png_paths + lsb_paths:
        if img_name in os.path.basename(encoded_path):
            encoded_im = plt.imread(encoded_path)*255
            encoded_im = encoded_im.astype(np.uint8)
            print('-- ' + encoded_path.split('/')[-1] + ' :')
            print(f'MSE={mse(origninal_im,encoded_im)},SSIM={ssim(origninal_im,encoded_im)},PSNR={psnr(origninal_im,encoded_im)}')