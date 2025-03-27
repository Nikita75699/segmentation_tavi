import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from dataloader import Dataset
from augmenation import get_training_augmentation
from augmenation import get_preprocessing
from segmentation_models_pytorch import utils
from torch.utils.data import DataLoader
import wandb
from autobatch import autobatch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torchinfo

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

CLASSES = ['aortic']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

decoders_params = [
    {"decoder": "FPN", "input_size": 512, "encoder": "efficientnet-b4", "learning_rate": 0.001, "optimizer": "RAdam"},
    {"decoder": "DeepLabV3Plus", "input_size": 512, "encoder": "se_resnext101_32x4d", "learning_rate": 0.0001, "optimizer": "RAdam"},
    {"decoder": "MAnet", "input_size": 896, "encoder": "efficientnet-b6", "learning_rate": 0.001, "optimizer": "RAdam"},
    {"decoder": "Linknet", "input_size": 640, "encoder": "efficientnet-b4", "learning_rate": 0.001, "optimizer": "RAdam"},
    {"decoder": "UnetPlusPlus", "input_size": 512, "encoder": "resnet101", "learning_rate": 0.0001, "optimizer": "Adam"},
    {"decoder": "PSPNet", "input_size": 640, "encoder": "se_resnext101_32x4d", "learning_rate": 0.001, "optimizer": "RAdam"}
]

datasets = ["fold_4", "fold_5"]

def create_model(encoder, decoder, encoder_weights='imagenet', classes=CLASSES, activation=ACTIVATION):
    if decoder == 'UnetPlusPlus':
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
    elif decoder == 'Linknet':
        return smp.Linknet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
    elif decoder == 'MAnet':
        return smp.MAnet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
    elif decoder == 'FPN':
        return smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
    elif decoder == 'DeepLabV3Plus':
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
    elif decoder == 'PSPNet':
        return smp.PSPNet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")


for dataset_f in datasets:

    DATA_DIR = f'/home/nikita/huge_ssd/Data/tavi_seg/tavi_data_01_02_25/{dataset_f}'

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    dataset = Dataset(x_train_dir, y_train_dir, classes=['aortic'])
    image, mask = dataset[0]

    augmented_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        classes=['aortic'],
    )

    for params in decoders_params:
        decoder = params["decoder"]
        encoder = params["encoder"]
        learning_rate = params["learning_rate"]
        optimizer_name = params["optimizer"]
        input_size = params["input_size"]



        model_name = f"{decoder}_{encoder}_{dataset_f}"
        model = create_model(encoder, decoder).to(DEVICE)
        input_tensor = torch.randn(2, 3, input_size, input_size).to(DEVICE)

        flops = FlopCountAnalysis(model, input_tensor)
        params = sum(p.numel() for p in model.parameters())

        print(f"Model: {model_name}")
        print(f"FLOPs: {flops.total() / 1e9:.3f} GFLOPs")  # В гигафлопах
        print(f"Params: {params / 1e6:.3f}M")  # В миллионах параметров

#         train_batch_size = autobatch(model, input_size)

#         # train_batch_size = 8
#         print(f"Training with dataset: {dataset_f}, encoder: {encoder}, decoder: {decoder}, lr: {learning_rate}, optimizer: {optimizer_name}, train_batch_size: {train_batch_size}")
#         wandb.init(project='aorta_segmentation7', name=model_name, reinit=True)

#         preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')

#         train_dataset = Dataset(
#             x_train_dir,
#             y_train_dir,
#             augmentation=get_training_augmentation(),
#             preprocessing=get_preprocessing(preprocessing_fn),
#             classes=CLASSES,
#             input_size=input_size,
#         )

#         valid_dataset = Dataset(
#             x_valid_dir,
#             y_valid_dir,
#             preprocessing=get_preprocessing(preprocessing_fn),
#             classes=CLASSES,
#             input_size=input_size,
#         )

#         train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=12)

#         valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=4)

#         loss = smp.utils.losses.DiceLoss()

#         metrics = [
#             smp.utils.metrics.IoU(threshold=0.5),
#             smp.utils.metrics.Fscore(threshold=0.5),
#             smp.utils.metrics.Precision(threshold=0.5),
#             smp.utils.metrics.Recall(threshold=0.5),
#         ]


#         if optimizer_name == "RAdam":
#             optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == "Adam":
#             optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         else:
#             raise ValueError(f"Unknown optimizer: {optimizer_name}")

#         train_epoch = smp.utils.train.TrainEpoch(
#             model,
#             loss=loss,
#             metrics=metrics,
#             optimizer=optimizer,
#             device=DEVICE,
#             verbose=True,
#         )

#         valid_epoch = smp.utils.train.ValidEpoch(
#             model,
#             loss=loss,
#             metrics=metrics,
#             device=DEVICE,
#             verbose=True,
#         )

#         max_score = 0

#         for i in range(30):
#             print('\nEpoch: {}'.format(i))
#             train_logs = train_epoch.run(train_loader)
#             valid_logs = valid_epoch.run(valid_loader)

#             wandb.log({
#                 "epoch": i,
#                 "train_loss": train_logs['dice_loss'],
#                 "train_iou": train_logs['iou_score'],
#                 "train_dice": train_logs['fscore'],  # Dice Score из smp
#                 "valid_loss": valid_logs['dice_loss'],
#                 "valid_iou": valid_logs['iou_score'],
#                 "valid_dice": valid_logs['fscore'],  # Dice Score из smp
#             })

#             os.makedirs(f'model/{model_name}', exist_ok=True)

#             if max_score < valid_logs['iou_score']:
#                 max_score = valid_logs['iou_score']
#                 torch.save(model, f'model/{model_name}/best_model.pth')
#                 print('Best model saved!')

#             if i == 25:
#                 optimizer.param_groups[0]['lr'] = 1e-5
#                 print('Decrease decoder learning rate to 1e-5!')

# wandb.finish()
