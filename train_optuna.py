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
from optuna.pruners import HyperbandPruner, PatientPruner
from segmentation_models_pytorch import utils
from datetime import datetime
import ssl
from autobatch import autobatch
import csv
import sys

ssl._create_default_https_context = ssl._create_unverified_context

gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

DATA_DIR = '/home/nikita/huge_ssd/Data/tavi_seg/tavi_data_01_02_25/fold'

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

CLASSES = ['aortic']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

def log_to_file(log_type, message):
    os.makedirs("loging", exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"loging/{log_type}_log_{current_time}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"{current_time} - {message}\n")
    print(f"{log_type.capitalize()} logged: {log_filename}")

def create_model(encoder, decoder, encoder_weights, classes=CLASSES, activation=ACTIVATION):
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
        error_message = f"create_model error: {str(e)}"
        log_to_file(f"error {decoder}_{encoder} optuna1", error_message)
        torch.cuda.empty_cache()
        return None

def objective(trial):
    try:
        input_size = trial.suggest_categorical("input_size", [512, 640, 896])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RAdam", "RMSprop"])
        decoder_name = trial.suggest_categorical("decoder", ["UnetPlusPlus", "Linknet", "MAnet", "FPN", "PSPNet", "DeepLabV3Plus"])
        encoder_name = trial.suggest_categorical("encoder", ['timm-regnetx_120', "resnet101", "efficientnet-b6",  "se_resnext50_32x4d", "resnet50", "efficientnet-b0", 'efficientnet-b4', "se_resnext101_32x4d",])

        print(f"Training with encoder: {encoder_name}, decoder: {decoder_name}")
        model_name = f"{decoder_name}_{encoder_name}_{trial.number}"
        try:
            model = create_model(encoder_name, decoder_name, encoder_weights='imagenet')
        except KeyError:
            print(f"Encoder '{encoder_name}' does not support 'imagenet' weights. Using random initialization.")
            model = create_model(encoder_name, decoder_name, encoder_weights=None)

        if model is None:
            return None  # Пропустить trial, если модель не создана

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "RAdam":
            optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        train_batch_size = autobatch(model, input_size, optimizer)
        valid_batch_size = train_batch_size

        wandb.init(
            project='aorta_segmentation_050325',
            name=model_name,
            config={
                "input_size": input_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "decoder": decoder_name,
                "encoder": encoder_name,
                'batch_size': train_batch_size
            },
            reinit=True)

        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')

        train_dataset = Dataset(
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
            input_size=input_size,
        )

        valid_dataset = Dataset(
            x_valid_dir,
            y_valid_dir,
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
            input_size=input_size,
        )

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

        loss = smp.utils.losses.DiceLoss()

        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(threshold=0.5),
            smp.utils.metrics.Precision(threshold=0.5),
            smp.utils.metrics.Recall(threshold=0.5),
        ]

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        csv_dir = "results_metrics"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{model_name}_{trial.number}.csv")
        fieldnames = ["epoch", "train_loss", "train_iou", "train_dice",
                      "valid_loss", "valid_iou", "valid_dice"]
        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        best_iou = 0
        for epoch in range(45):
            print('\nEpoch: {}'.format(epoch))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            current_iou = valid_logs['iou_score']

            wandb.log({
                "epoch": epoch,
                "train_loss": train_logs['dice_loss'],
                "train_iou": train_logs['iou_score'],
                "train_dice": train_logs['fscore'],  # Dice Score из smp
                "valid_loss": valid_logs['dice_loss'],
                "valid_iou": valid_logs['iou_score'],
                "valid_dice": valid_logs['fscore'],  # Dice Score из smp
            })

            with open(csv_path, mode="a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": train_logs['dice_loss'],
                    "train_iou": train_logs['iou_score'],
                    "train_dice": train_logs['fscore'],
                    "valid_loss": valid_logs['dice_loss'],
                    "valid_iou": valid_logs['iou_score'],
                    "valid_dice": valid_logs['fscore'],
                })

            trial.report(current_iou, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            os.makedirs(f'model/{model_name}_{trial.number}', exist_ok=True)

            if current_iou > best_iou:
                best_iou = current_iou
                torch.save(model, f'model/{model_name}_{trial.number}/best_model_{trial.number}.pth')
                print('Best model saved!')

            if epoch == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

        wandb.finish()
        log_to_file(f"success {model_name}_{trial.number} optuna1", f"Trial {model_name}_{trial.number} finished successfully with best IOU: {best_iou}")

        return best_iou
    except optuna.TrialPruned:
        wandb.finish()
        log_to_file(f"pruned {model_name}_{trial.number} optuna1", f"Trial {model_name}_{trial.number} was pruned.")
        raise optuna.TrialPruned()

    except torch.cuda.OutOfMemoryError as e:
        log_to_file(f"error CUDA  {model_name}_{trial.number} optuna1", f"Trial {model_name}_{trial.number} - CUDA out of memory: {str(e)}")
        print(f"CUDA out of memory: {e}. Error logged.")
        torch.cuda.empty_cache()
        return None

    except Exception as e:
        log_to_file(f"error {model_name}_{trial.number} optuna1", f"Trial {model_name}_{trial.number} - Exception: {str(e)}")
        print(f"Exception in trial {model_name}_{trial.number}: {e}. Logged.")
        torch.cuda.empty_cache()
        return None
# Настройка и запуск оптимизации
study = optuna.create_study(
    study_name=f"optuna-example",
    storage="sqlite:///example.db",
    direction='maximize',
    sampler=TPESampler(),
    load_if_exists=True,
    pruner=optuna.pruners.PatientPruner(optuna.pruners.HyperbandPruner(), patience=10)
)
study.optimize(objective, n_trials=200)

print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")
  
