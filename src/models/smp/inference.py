import json
import os.path
import time
import cv2
import torchvision
import numpy as np
from src.models.smp.model import OCTSegmentationModel
from glob import glob

def to_tensor(
    x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')


get_tensor = torchvision.transforms.ToTensor()


def preprocessing_img(
    img_path: str,
    input_size: int,
):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (input_size, input_size))
    # image = to_tensor(np.array(image))
    return image


def main():
    model_path = 'models/4_class/MA-Net'
    with open(f'{model_path}/config.json', 'r') as file:
        model_cfg = json.load(file)
    model = OCTSegmentationModel.load_from_checkpoint(
        checkpoint_path=f'{model_path}/weights.ckpt',
        encoder_weights=None,
        arch=model_cfg['architecture'],
        encoder_name=model_cfg['encoder'],
        model_name=model_cfg['model_name'],
        in_channels=3,
        classes=model_cfg['classes'],
        map_location='cuda:0',
    )
    model.eval()

    for img_path in glob('/home/vladislav/PythonProject/Git-hub/oct_segmentation/data/visualization_/img/*'):
        image = preprocessing_img(
            img_path=img_path,
            input_size=model_cfg['input_size'],
        )
        from PIL import Image
        img = Image.open(img_path)
        img = img.resize((1024, 1024))
        masks = model.predict(
            images=np.array([image]),
            device='cuda',
        )[0]
        masks = cv2.resize(masks, (1024, 1024))
        from src.models.smp.utils import get_img_mask_union_pil
        from src.data.utils import CLASS_IDS, CLASS_COLORS_RGB
        color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
        for class_name in CLASS_IDS:
            mask = masks[:, :, CLASS_IDS[class_name] - 1] * 255
            img = get_img_mask_union_pil(
                img=img,
                mask=mask,
                color=CLASS_COLORS_RGB[class_name],
            )
            class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[class_name])
            color_mask.paste(class_img, (0, 0),  Image.fromarray(mask.astype('uint8')))
        img.save(f'{model_path}/masks/{os.path.basename(img_path).split(".")[0]}_union.png')
        color_mask.save(f'{model_path}/masks/{os.path.basename(img_path).split(".")[0]}_color_mask.png')



# def main():
#     for img_path in glob('/home/vladislav/PythonProject/Git-hub/oct_segmentation/data/visualization_/img/*'):
#
#         from PIL import Image
#         img = Image.open(img_path)
#         img = img.resize((1024, 1024))
#
#         from src.models.smp.utils import get_img_mask_union_pil
#         from src.data.utils import CLASS_IDS, CLASS_COLORS_RGB
#         color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
#
#         model_path = 'models/1_class/DeepLabV3'
#         with open(f'{model_path}/config.json', 'r') as file:
#             model_cfg = json.load(file)
#         model = OCTSegmentationModel.load_from_checkpoint(
#             checkpoint_path=f'{model_path}/weights.ckpt',
#             encoder_weights=None,
#             arch=model_cfg['architecture'],
#             encoder_name=model_cfg['encoder'],
#             model_name=model_cfg['model_name'],
#             in_channels=3,
#             classes=model_cfg['classes'],
#             map_location='cuda:0',
#         )
#         model.eval()
#         image = preprocessing_img(
#             img_path=img_path,
#             input_size=model_cfg['input_size'],
#         )
#         masks_1 = model.predict(
#             images=np.array([image]),
#             device='cuda',
#         )[0]
#         masks_1 = cv2.resize(masks_1, (1024, 1024))
#
#         model_path = 'models/3_class/PAN'
#         with open(f'{model_path}/config.json', 'r') as file:
#             model_cfg = json.load(file)
#         model = OCTSegmentationModel.load_from_checkpoint(
#             checkpoint_path=f'{model_path}/weights.ckpt',
#             encoder_weights=None,
#             arch=model_cfg['architecture'],
#             encoder_name=model_cfg['encoder'],
#             model_name=model_cfg['model_name'],
#             in_channels=3,
#             classes=model_cfg['classes'],
#             map_location='cuda:0',
#         )
#         model.eval()
#         image = preprocessing_img(
#             img_path=img_path,
#             input_size=model_cfg['input_size'],
#         )
#         masks_3 = model.predict(
#             images=np.array([image]),
#             device='cuda',
#         )[0]
#         masks_3 = cv2.resize(masks_3, (1024, 1024))
#
#         for class_name in CLASS_IDS:
#             if CLASS_IDS[class_name] - 1 < 3:
#                 mask = masks_3[:, :, CLASS_IDS[class_name] - 1] * 255
#                 img = get_img_mask_union_pil(
#                     img=img,
#                     mask=mask,
#                     color=CLASS_COLORS_RGB[class_name],
#                 )
#             else:
#                 mask = masks_1 * 255
#                 img = get_img_mask_union_pil(
#                     img=img,
#                     mask=mask,
#                     color=CLASS_COLORS_RGB[class_name],
#                 )
#             class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[class_name])
#             color_mask.paste(class_img, (0, 0),  Image.fromarray(mask.astype('uint8')))
#         img.save(f'{model_path}/masks/{os.path.basename(img_path).split(".")[0]}_union.png')
#         color_mask.save(f'{model_path}/masks/{os.path.basename(img_path).split(".")[0]}_color_mask.png')


if __name__ == '__main__':
    main()