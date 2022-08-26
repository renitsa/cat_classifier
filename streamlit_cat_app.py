import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
#import pickle as pkl
import tensorflow as tf
#from tensorflow.keras import models, layers
#from keras.preprocessing import image as tfimage
import cv2
import os

def encode(opencv_image):
    image_from_array = Image.fromarray(opencv_image, 'RGB')
    resized_image = image_from_array.resize((128,128))
    p = np.expand_dims(resized_image, 0)
    predict_img = tf.cast(p, tf.float32)
    return predict_img

pred_dict = {0:'Cat', 1:'Not Cat'}

#model_path = 'C:\\Users\\Lucian\\GitRepos\\DSR_practical_DS\\universal_cat_noncat_app\\CNN_cats.h5'
model_path = os.path.join(os.getcwd(),'CNN_cats.h5')
model = tf.keras.models.load_model(model_path)

st.title('Universal cat/non-cat classifier')
uploaded_file = st.file_uploader('Upload cat/non-cat image files', type= ['png', 'jpg'], accept_multiple_files=False)

#imdata = uploaded_file.read()
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    predict_img = encode(opencv_image)
    #predict_img = encode(opencv_image)
    #st.write(predict_img.shape)
    prediction = model.predict(predict_img)[0,0]
    st.write(f'Prediction: {pred_dict[int(prediction)]}')

else:
    st.write('No image uploaded yet')