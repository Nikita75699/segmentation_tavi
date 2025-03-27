import os
import torch
import segmentation_models_pytorch as smp
from dataloader import Dataset
from augmenation import get_training_augmentation
from augmenation import get_preprocessing
from torch.utils.data import DataLoader
import wandb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from segmentation_models_pytorch import utils
from torch.cuda import memory_reserved, memory_allocated, empty_cache
from datetime import datetime
import ssl
from segmentation_models_pytorch import UnetPlusPlus, Linknet, MAnet, FPN, PSPNet, DeepLabV3Plus
from autobatch import autobatch

ssl._create_default_https_context = ssl._create_unverified_context

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# DATA_DIR = '/home/nikita/huge_ssd/Data/tavi_seg/tavi_data_01_02_25'

# x_train_dir = os.path.join(DATA_DIR, 'train')
# y_train_dir = os.path.join(DATA_DIR, 'trainannot')

# x_valid_dir = os.path.join(DATA_DIR, 'val')
# y_valid_dir = os.path.join(DATA_DIR, 'valannot')

# dataset = Dataset(x_train_dir, y_train_dir, classes=['aortic'])
# image, mask = dataset[0]

# augmented_dataset = Dataset(
#     x_train_dir,
#     y_train_dir,
#     augmentation=get_training_augmentation(),
#     classes=['aortic'],
# )

CLASSES = ['aortic']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

def create_model(encoder, decoder, encoder_weights='imagenet', classes=CLASSES, activation=ACTIVATION):
    try:
        if decoder == 'UnetPlusPlus':
            return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
        elif decoder == 'Linknet':
            return smp.Linknet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
        elif decoder == 'MAnet':
            return smp.MAnet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
        elif decoder == 'FPN':
            return smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
        elif decoder == 'PSPNet':
            return smp.PSPNet(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
        elif decoder == 'DeepLabV3Plus':
            return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, classes=len(classes), in_channels=3, activation=activation)
        else:
            raise ValueError(f"Unknown decoder: {decoder}")
    except ValueError as e:
        # Получаем текущее время для имени файла
        error_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"loging/log_{error_time}.txt"

        # Записываем ошибку в файл
        with open(log_filename, "w") as log_file:
            log_file.write(f"Error occurred at: {error_time}\n")
            log_file.write(f"Error details: {str(e)}\n")

        print(f"Error: {e}. Error logged in {log_filename}")
        torch.cuda.empty_cache()  # Освободить память GPU
        return None  # Пропустить trial

# def objective(trial):


#     input_size = trial.suggest_categorical("input_size", [512, 640, 896])
#     learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])
#     optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RAdam", "RMSprop"])
#     decoder_name = trial.suggest_categorical("decoder", ["MAnet", "DeepLabV3Plus"])
#     encoder_name = trial.suggest_categorical("encoder", ["efficientnet-b7"])

#{'input_size': 640, 'learning_rate': 1e-05, 'optimizer': 'RMSprop', 'decoder': 'FPN', 'encoder': 'resnet50'}
encoder_name = 'se_resnext50_32x4d'
decoder_name = 'MAnet'

input_size = 512
learning_rate = 0.001
print(f"Training with encoder: {encoder_name}, decoder: {decoder_name}")
model_name = f"{decoder_name}_{encoder_name}"
print(model_name)
model = create_model(encoder_name, decoder_name)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
train_batch_size = autobatch(model, input_size, optimizer)
print(train_batch_size)
valid_batch_size = train_batch_size

#         preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')

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
#         valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

#         loss = smp.utils.losses.DiceLoss()

#         metrics = [
#             smp.utils.metrics.IoU(threshold=0.5),
#             smp.utils.metrics.Fscore(threshold=0.5),
#             smp.utils.metrics.Precision(threshold=0.5),
#             smp.utils.metrics.Recall(threshold=0.5),
#         ]

#         if optimizer_name == "Adam":
#             optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == "RAdam":
#             optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == "RMSprop":
#             optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

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

#         best_iou = 0
#         for epoch in range(30):
#             print('\nEpoch: {}'.format(epoch))
#             train_logs = train_epoch.run(train_loader)
#             valid_logs = valid_epoch.run(valid_loader)
#             current_iou = valid_logs['iou_score']

#             # Отчет для Optuna
#             trial.report(current_iou, epoch)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()

#             os.makedirs(f'model/{model_name}_{trial.number}', exist_ok=True)

#             if current_iou > best_iou:
#                 best_iou = current_iou
#                 torch.save(model, f'model/{model_name}_{trial.number}/best_model_{trial.number}.pth')
#                 print('Best model saved!')

#             if epoch == 25:
#                 optimizer.param_groups[0]['lr'] = 1e-5
#                 print('Decrease decoder learning rate to 1e-5!')

#         return best_iou
#     except torch.cuda.OutOfMemoryError as e:
#         # Получаем текущее время для имени файла
#         error_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         log_filename = f"loging/log_{error_time}.txt"

#         # Записываем ошибку в файл
#         with open(log_filename, "w") as log_file:
#             log_file.write(f"Error occurred at: {error_time}\n")
#             log_file.write(f"Error details: {str(e)}\n")

#         print(f"CUDA out of memory: {e}. Error logged in {log_filename}")
#         torch.cuda.empty_cache()  # Освободить память GPU
#         return None  # Пропустить trial
# # Настройка и запуск оптимизации
# study = optuna.create_study(
#     direction='maximize',
#     sampler=TPESampler(),
#     pruner=HyperbandPruner()
# )
# study.optimize(objective, n_trials=150)

# # Вывод результатов
# print("Best trial:")
# print(f"  Value: {study.best_trial.value}")
# print("  Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"{key}: {value}")