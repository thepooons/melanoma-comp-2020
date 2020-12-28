import streamlit as st
import PIL.Image as Image
import cv2
from src.model import get_model, load_weights
from src.preprocess import INFER_TRANSFORMS
from src.infer import infer_once
import numpy as np
import sys
import gc

if __name__ == "__main__":
    try:
        if '.ckpt' in sys.argv[1].lower():
            MODEL_CKPT_PATH = sys.argv[1]
    except IndexError:
        print('Please download the network weights and provide its path')

    DEBUG = False
    RESNEST_MODEL = 'resnest50_fast_4s1x64d'  # ~FIXED~
    TTA = 4 if DEBUG else 8
    THRESHOLD = 0.5

    if DEBUG:
        print('~:~:~THIS PROGRAM IS WORKING IN DEBUG MODE~:~:~')

    st.write('# SIIM ISIC Melanoma Classification Demo')
    st.write('This app is backed with a flavor of\
        [ResNeSt](https://arxiv.org/abs/2004.08955) Model trained on imagenet\
        weights and finetuned on SIIM 2020 data by\
        [Puneet Singh](https://twitter.com/p69ns).')

    add_selectbox = st.sidebar.selectbox(
        "How would like to make predictions?",
        ("TTA Averaged", "Single inference")
    )
    if add_selectbox == "Single Inference":
        TTA = 1
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_buffer = st.file_uploader("Upload Image:", type="jpg")

    try:
        if file_buffer is not None:
            image = Image.open(file_buffer)
            image = cv2.cvtColor(
                np.array(image).transpose((1, 0, 2)),
                cv2.COLOR_BGR2RGB
            )
        if DEBUG:
            st.write(image.shape)
        st.image(image, use_column_width=True)

        model = get_model(RESNEST_MODEL)
        model = load_weights(
            model=model,
            model_ckpt_path=MODEL_CKPT_PATH,
            )
        if DEBUG:
            print(type(model))

        predictions = []
        for _ in range(TTA):
            predictions.append(
                infer_once(
                    image=image,
                    transforms=INFER_TRANSFORMS,
                    model=model,
                    debug=DEBUG,
                )
            )

        diagnosis = 'Benign' if np.mean(predictions) < THRESHOLD\
            else 'Malignant Melanoma'

        st.info('**DISCLAIMER: This app is in no way a system eligible\
            to make diagnoses which match the level of a professional.\
            It is just an experiment to explore the limits of CNNs.**')
        st.write("The model diagnosed the image as:", diagnosis)

        # preform clean-up to save cloud vm from exploding
        del predictions, model, image
        gc.collect()

    except Exception:
        st.write('Upload an image of skin lesion for further actions')
