# melanoma-comp-2020
This repository houses the code for the streamlit + heroku app backed with a CNN fine-tuned on the SIIM ISIC Melanoma Classification Competition on Kaggle.

# todo
- [x] Create a clean inference pipeline:
    - [x] Load the ckpts into cpu memory (the model is trained on gpu, and the weight tensors need to be changed to cpu)
    - [x] Freeze the layers of the network
    - [x] Make predictions using the frozen network
- [x] Create a local web app using streamlit:
    - [x] Disclaimer, regarding incompetency of the model to compete with professional medical diagnosis
    - [x] Upload the image
    - [x] Run the model's `make_prediction` method to create the prediction on the image uploaded
- [ ] Upload to heroku:
    - [ ] Create neccessary changes to repository to make it work on heroku