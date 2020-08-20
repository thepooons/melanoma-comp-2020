import cv2
import torch


def get_image(
    image_path,
    transforms
):
    """Lightweight dataset + dataloader

    Args:
        image_path (str): path of the image
        transforms (Albumentations): transforms to apply on the images

    Returns:
        torch.Tensor: the image is preprocessed and converted into a
        torch.Tensor
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image=image)
    return image


def infer_once(image_path, transforms, model):
    image = get_image(
        image_path=image_path,
        transforms=transforms,
    )['image']
    model.eval()

    prediction = torch.sigmoid(model(image.unsqueeze(dim=0))).detach()
    return prediction.numpy()
