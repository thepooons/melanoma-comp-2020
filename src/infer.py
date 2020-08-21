import torch


def get_image(
    image,
    transforms,
    debug=False
):
    """Lightwieght Dataset+Dataloader

    Args:
        image (cv2.Image): this is the image uploaded from the `file_uploader`
        transforms (A.compose): transformations to be applied to the image

    Returns:
        numpy.array: the prediction of the model on the image uploaded
    """
    if debug:
        print('~:~:~preprocessing the image')
    image = transforms(image=image)
    return image


def infer_once(image, transforms, model, debug=False):
    if debug:
        print('~:~:~making predictions x1')
    image = get_image(
        image=image,
        transforms=transforms,
        debug=debug,
    )['image']
    model.eval()

    prediction = torch.sigmoid(model(image.unsqueeze(dim=0))).detach()
    return prediction.numpy()
