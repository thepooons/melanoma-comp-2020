from src.model import get_model, load_weights
from src.preprocess import INFER_TRANSFORMS
from src.infer import infer_once
import numpy as np


DEBUG = True
MODEL_CKPT_PATH = r'data\model_ckpts\model_epoch=11_val_auc=0.8486.ckpt'
RESNEST_MODEL = 'resnest50_fast_4s1x64d'  # ~FIXED~
IMAGE_PATH = r'data\images\melanoma7.jpg'  # select from streamlit upload
TTA = 4
THRESHOLD = 0.5

# create the model and load the weights
model = get_model(RESNEST_MODEL)
model = load_weights(
    model=model,
    model_ckpt_path=MODEL_CKPT_PATH,
    )
if DEBUG:
    print(type(model))

predictions = []
for _ in range(TTA):
    predictions.append(infer_once(
        image_path=IMAGE_PATH,
        transforms=INFER_TRANSFORMS,
        model=model
    ))

diagnosis = 'Benign' if np.mean(predictions) < THRESHOLD else \
    'Malignant Melanoma'
print('the model diagnosed the image as:', diagnosis)
