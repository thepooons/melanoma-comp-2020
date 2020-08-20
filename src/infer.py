import torch


def get_image(
    image,
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
    image = transforms(image=image)
    return image


def infer_once(image, transforms, model):
    image = get_image(
        image=image,
        transforms=transforms,
    )['image']
    model.eval()

    prediction = torch.sigmoid(model(image.unsqueeze(dim=0))).detach()
    return prediction.numpy()
