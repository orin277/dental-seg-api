import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class SegmentationTransform:
    def __init__(self, image_size=512):
        self.train_transform = A.Compose([
            A.Resize(height=image_size, width=image_size, p=1.0,
                     interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),

            A.HorizontalFlip(p=0.5),
            A.Affine(
                rotate=(-10, 10),
                scale=(0.9, 1.1),
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                p=0.5,
            ),

            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(height=image_size, width=image_size, p=1.0,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

        self.val_transform_for_image = A.Compose([
            A.Resize(height=image_size, width=image_size, p=1.0,
                     interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        
    def transform_image(self, image):
        transformed = self.val_transform_for_image(image=image)
        return transformed['image']

    def __call__(self, image, mask, is_train=True):
        if is_train:
            transformed = self.train_transform(image=image, mask=mask)
        else:
            transformed = self.val_transform(image=image, mask=mask)

        return transformed['image'], transformed['mask']
    
