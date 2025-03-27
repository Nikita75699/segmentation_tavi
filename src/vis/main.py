from typing import Tuple

import cv2
from ultralytics import YOLO
import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image

from src.data.utils import CLASS_COLORS_RGB, CLASS_IDS, CLASS_IDS_REVERSED
from src.models.smp.model import SegmentationModel

yolo_model = YOLO("models/yolo/best.pt")  # load a custom model

model = SegmentationModel.load_from_checkpoint(
    'models/unet_resnet50_0407_0031/weights.ckpt',
    arch='unet',
    encoder_name='resnet50',
    model_name='unet_resnet50_0407_0031',
    in_channels=3,
    classes=[
        'Microaneurysms',
        'Haemorrhages',
        'Hard Exudates',
        'Soft Exudates',
        'Optic Disc'
    ],
    lr=0.0001,
    optimizer_name='Adam',
)
model.eval()
get_tensor = torchvision.transforms.ToTensor()


def to_tensor(
    x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')
    # return x.transpose([0, 1, 2]).astype('float32')


def sem_processing(*args):
    (source_image,) = args

    return inference(source_image=source_image.copy())


def yolo_processing(*args):
    (source_image,) = args
    results = yolo_model(source_image)
    mask_example = results[0].masks[0].data[0].cpu().numpy()
    source_image = source_image.resize((mask_example.shape[1], mask_example.shape[0]))
    source_image = np.array(source_image)

    color_mask_pred = np.zeros((mask_example.shape[0], mask_example.shape[1], 3))
    color_mask_pred[:, :] = (128, 128, 128)

    for res in results:
        for box, mask in zip(res.boxes, res.masks):
            class_id = int(box.cls.cpu().numpy()[0])
            if box.conf.cpu().numpy()[0] >= 0.2 and class_id > 0:
                mask = mask.data[0].cpu().numpy()
                color_mask_pred[mask[:, :] == 1] = CLASS_COLORS_RGB[CLASS_IDS_REVERSED[class_id + 1]]  # type: ignore

                source_image = get_img_mask_union(
                    img_0=source_image,
                    alpha_0=1,
                    img_1=mask,
                    alpha_1=0.5,
                    color=CLASS_COLORS_RGB[CLASS_IDS_REVERSED[class_id + 1]],  # type: ignore
                )

        return Image.fromarray(np.array(source_image).astype('uint8')), Image.fromarray(
            np.array(color_mask_pred).astype('uint8'),
        )


def get_img_mask_union(
    img_0: np.ndarray,
    alpha_0: float,
    img_1: np.ndarray,
    alpha_1: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.addWeighted(
        np.array(img_0).astype('uint8'),
        alpha_0,
        (cv2.cvtColor(np.array(img_1).astype('uint8'), cv2.COLOR_GRAY2RGB) * color).astype(
            np.uint8,
        ),
        alpha_1,
        0,
    )


def inference(
    source_image: Image.Image,
):
    source_image = np.array(source_image)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    source_image = cv2.resize(source_image, (1024, 1024))

    image = to_tensor(np.array(source_image.copy()))
    y_hat = model(torch.Tensor([image]).to('cuda'))
    mask_pred = y_hat.sigmoid()
    mask_pred = (mask_pred > 0.5).float()
    mask_pred = mask_pred.permute(0, 2, 3, 1)
    mask_pred = mask_pred.squeeze().cpu().numpy().round()

    color_mask_pred = np.zeros(source_image.shape)
    color_mask_pred[:, :] = (128, 128, 128)

    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

    for class_id in CLASS_IDS_REVERSED:
        m = np.zeros((source_image.shape[1], source_image.shape[0]))
        m[mask_pred[:, :, class_id - 1] == 1] = 1  # type: ignore

        source_image = get_img_mask_union(
            img_0=source_image,
            alpha_0=1,
            img_1=m,
            alpha_1=0.5,
            color=CLASS_COLORS_RGB[CLASS_IDS_REVERSED[class_id]],  # type: ignore
        )

        color_mask_pred[mask_pred[:, :, class_id - 1] == 1] = CLASS_COLORS_RGB[CLASS_IDS_REVERSED[class_id]]  # type: ignore

    return Image.fromarray(np.array(source_image).astype('uint8')), Image.fromarray(
        np.array(color_mask_pred).astype('uint8'),
    )


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    source_image = gr.Image(
                        label='Source Image',
                        image_mode='RGB',
                        type='pil',
                        sources=['upload'],
                        # height=640,
                    )
                with gr.Row():
                    start_sem = gr.Button('SEM')
                    start_yolo = gr.Button('YOLO')
            with gr.Column(scale=1):
                results = gr.Gallery(height=640, columns=1, object_fit='scale-down')

        start_sem.click(
            sem_processing,
            inputs=[
                source_image,
            ],
            outputs=results,
        )
        start_yolo.click(
            yolo_processing,
            inputs=[
                source_image,
            ],
            outputs=results,
        )

    demo.launch(
        server_name='0.0.0.0',
        server_port=7883,
    )


if __name__ == '__main__':
    main()