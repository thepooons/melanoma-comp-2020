import torch


def get_image(
    image,
    transforms
):
    """Lightwieght Dataset+Dataloader

    Args:
        image (cv2.Image): this is the image uploaded from the `file_uploader`
        transforms (A.compose): transformations to be applied to the image

    Returns:
        numpy.array: the prediction of the model on the image uploaded
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
