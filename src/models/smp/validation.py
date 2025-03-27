import os
import json
import cv2
import tifffile
import numpy as np

from PIL import Image
from glob import glob
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from src.models.smp.model import OCTSegmentationModel
from src.models.smp.utils import get_img_mask_union_pil
from src.data.utils import CLASS_IDS, CLASS_COLORS_RGB


def preprocessing_img(
    img_path: str,
    input_size: int,
):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (input_size, input_size))
    return image


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'micro',
):
    y_true[y_true > 0] = 1.0
    y_pred[y_pred > 0] = 1.0
    dice = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    iou = jaccard_score(y_true=y_true, y_pred=y_pred, average=average)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    metrics = {
        'Dice': dice,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
    }
    return metrics


if __name__ == '__main__':
    model_path = 'models/3_class/PAN'
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

    data_dir = '/home/vladislav/PythonProject/Git-hub/oct_segmentation/data/validation'
    for img_path in glob(f'{data_dir}/img/*'):
        dice = 0.0
        img_name = os.path.basename(img_path)
        image = preprocessing_img(
            img_path=img_path,
            input_size=model_cfg['input_size'],
        )
        img = Image.open(img_path)
        img = img.resize((1024, 1024))
        masks = model.predict(
            images=np.array([image]),
            device='cuda',
        )[0]
        masks = cv2.resize(masks, (1024, 1024))

        masks_gr = tifffile.imread(f'{data_dir}/mask/{img_name.split(".")[0]}.tiff')
        masks_gr = cv2.resize(
            masks_gr,
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST,
        )
        color_mask = Image.new('RGB', size=img.size, color=(128, 128, 128))
        for class_name in CLASS_IDS:
            if class_name != 'Vasa vasorum':
                mask = masks[:, :, CLASS_IDS[class_name] - 1] * 255

                print(
                    img_name,
                    class_name,
                    compute_metrics(
                        y_true=masks_gr[:, :, CLASS_IDS[class_name] - 1],
                        y_pred=masks[:, :, CLASS_IDS[class_name] - 1]
                    )
                )

                img = get_img_mask_union_pil(
                    img=img,
                    mask=mask,
                    color=CLASS_COLORS_RGB[class_name],
                )
                class_img = Image.new('RGB', size=img.size, color=CLASS_COLORS_RGB[class_name])
                color_mask.paste(class_img, (0, 0), Image.fromarray(mask.astype('uint8')))
            img.save(f'{model_path}/masks/{os.path.basename(img_path).split(".")[0]}_union.png')
            color_mask.save(f'{model_path}/masks/{os.path.basename(img_path).split(".")[0]}_color_mask.png')