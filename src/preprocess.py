from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A


INFER_TRANSFORMS = A.Compose([
    A.JpegCompression(p=0.5),
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.3),
    A.CenterCrop(height=128, width=128, always_apply=True),
    A.Normalize(),
    ToTensorV2(),
], p=1.0)
