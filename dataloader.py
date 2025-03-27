from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np

class Dataset(BaseDataset):

    CLASSES = ['aortic']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            input_size=896
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.input_size = input_size

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i], 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size))
        mask = np.expand_dims(mask, axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

# , self.images_fps[i]