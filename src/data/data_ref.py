import os.path
from PIL import Image
from glob import glob
from tqdm import tqdm

import tifffile
import numpy as np

# 1. 5 classes
# if __name__ == '__main__':
#     data_dir = 'data/fold/train'
#     for img_path in tqdm(glob(f'{data_dir}/img/*.jpg')):
#         img_name = os.path.basename(img_path).split('.')[0]
#         img = Image.open(img_path)
#         new_mask = np.zeros((5, img.size[1], img.size[0]), dtype='uint8')
#         obg = False
#         for mask_path in glob(f'{data_dir}/mask/*/{img_name}*.tif'):
#             mask = tifffile.imread(mask_path)
#             obg = True
#             if '1. Microaneurysms' in mask_path:
#                 new_mask[0, :, :][mask == True] = 255
#             elif '2. Haemorrhages' in mask_path:
#                 new_mask[1, :, :][mask == True] = 255
#             elif '3. Hard Exudates' in mask_path:
#                 new_mask[2, :, :][mask == True] = 255
#             elif '4. Soft Exudates' in mask_path:
#                 new_mask[3, :, :][mask == True] = 255
#             elif '5. Optic Disc' in mask_path:
#                 new_mask[4, :, :][mask == True] = 255
#             else:
#                 pass
#         if obg:
#             tifffile.imwrite(os.path.join(data_dir, 'mask', f'{img_name}.tif'), new_mask, compression='LZW')

# 2 classes 1+2, 3+4
# if __name__ == '__main__':
#     data_dir = 'data/fold_1/test'
#     for img_path in tqdm(glob(f'{data_dir}/img/*.jpg')):
#         img_name = os.path.basename(img_path).split('.')[0]
#         img = Image.open(img_path)
#         new_mask = np.zeros((2, img.size[1], img.size[0]), dtype='uint8')
#         obg = False
#         for mask_path in glob(f'{data_dir}/mask/*/{img_name}*.tif'):
#             mask = tifffile.imread(mask_path)
#             obg = True
#             if '1. Microaneurysms' in mask_path or '2. Haemorrhages' in mask_path:
#                 new_mask[0, :, :][mask == True] = 255
#             elif '3. Hard Exudates' in mask_path or '4. Soft Exudates' in mask_path:
#                 new_mask[1, :, :][mask == True] = 255
#             else:
#                 pass
#         if obg:
#             tifffile.imwrite(os.path.join(data_dir, 'mask', f'{img_name}.tif'), new_mask, compression='LZW')


# 4 classes 1,2,3,4
if __name__ == '__main__':
    for split in ['train', 'test']:
        data_dir = f'data/fold_3/{split}'
        for img_path in tqdm(glob(f'{data_dir}/img/*.jpg')):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path)
            new_mask = np.zeros((3, img.size[1], img.size[0]), dtype='uint8')
            obg = False
            for mask_path in glob(f'{data_dir}/mask/*/{img_name}*.tif'):
                mask = tifffile.imread(mask_path)
                mask_ = np.zeros((img.size[1], img.size[0]), dtype='uint8')
                obg = True
                if '1. Microaneurysms' in mask_path or '2. Haemorrhages' in mask_path:
                    mask_[:, :][mask == True] = 255
                    new_mask[0, :, :] = mask_
                elif '3. Hard Exudates' in mask_path:
                    mask_[:, :][mask == True] = 255
                    new_mask[1, :, :] = mask_
                elif '4. Soft Exudates' in mask_path:
                    mask_[:, :][mask == True] = 255
                    new_mask[2, :, :] = mask_
                else:
                    pass
            if obg:
                tifffile.imwrite(os.path.join(data_dir, 'mask', f'{img_name}.tif'), new_mask, compression='LZW')