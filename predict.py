import torch
import numpy as np
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = 'cuda'

SAVE = 'images'
DATA_DIR = '/home/nikita/Projects/Data/seg_torch_tavi/seg'
MODEL_PATH = '/home/nikita/huge_ssd/Project/tavi_detection/segmentation_unet/Linknet_efficientnet-b5/best_model.pth'
IMAGE_PATH = '/home/nikita/huge_ssd/Data/segmentation (copy)/images/val/952903_0003_10_007.png'

with torch.no_grad():
    best_model = torch.load(MODEL_PATH)
    best_model.eval()

image = cv2.imread('/home/nikita/huge_ssd/Data/segmentation (copy)/images/val/952903_0003_10_007.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (640, 640))
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image_vis = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)


image_rgb = image_rgb.astype(np.float32) / 255.0
x_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(DEVICE).unsqueeze(0)

with torch.no_grad():
    pr_mask = best_model(x_tensor)
pr_mask = (pr_mask.squeeze().cpu().numpy().round())

pr_mask_ = np.zeros(pr_mask.shape)
pr_mask_[pr_mask != 1] = 1

mask_red = np.zeros_like(image_vis)
mask_red[pr_mask_ == 0] = [0, 0, 255]
alpha = 0.4
mask_with_alpha = cv2.addWeighted(image_vis, 1 - alpha, mask_red, alpha, 0)

save_path='predicted_mask.png'

cv2.imwrite(os.path.join(SAVE, save_path), mask_with_alpha)

