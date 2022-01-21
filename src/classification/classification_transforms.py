import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import *


def get_transform(img_size=512, transform_type='base'):

    transform = None
    if transform_type == 'base':
        transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.OneOf([
                    A.RandomGamma(),
                    A.GaussNoise()
                ], p=0.5),
                A.OneOf([
                    A.OpticalDistortion(p=0.4),
                    A.GridDistortion(p=0.2),
                    A.IAAPiecewiseAffine(p=0.4),
                ], p=0.5),
                A.OneOf([
                    A.HueSaturationValue(10, 15, 10),
                    A.CLAHE(clip_limit=4),
                    A.RandomBrightnessContrast(),

                ], p=0.5),
                A.Equalize(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                A.Resize(img_size, img_size, p=1.0),
                # A.Normalize(
                #     mean=(0.485, 0.456, 0.406),
                #     std=(0.229, 0.224, 0.225)
                # ),
                ToTensorV2()
            ])
    if transform_type == 'try_new_strong':
        transform = A.Compose([
                           A.RandomResizedCrop(img_size, img_size, scale=(0.9, 1), p=1),
                           A.HorizontalFlip(p=0.5),
                           A.ShiftScaleRotate(p=0.5),
                           A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                           A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                           A.CLAHE(clip_limit=(1,4), p=0.5),

                               A.OneOf([
                                   A.OpticalDistortion(distort_limit=1.0),
                                   A.GridDistortion(num_steps=5, distort_limit=1.),
                                   A.ElasticTransform(alpha=3),
                               ], p=0.2),
                               A.OneOf([
                                   A.GaussNoise(var_limit=[10, 50]),
                                   A.GaussianBlur(),
                                   A.MotionBlur(),
                                   A.MedianBlur(),
                               ], p=0.2),
                          A.Resize(img_size, img_size),
                            IAAPiecewiseAffine(p=0.2),
                            IAASharpen(p=0.2),
                              A.Cutout(max_h_size=int(img_size * 0.1), max_w_size=int(img_size * 0.1), num_holes=5, p=0.5),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2()
])

    if transform_type == 'val':
        transform = A.Compose([
            A.Resize(img_size, img_size, p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ], p=1.0)

    return transform