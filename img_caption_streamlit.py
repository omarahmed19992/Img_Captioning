# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 07:49:06 2024

@author: Dubai
"""
###### import libs
import os 
import pickle 
import numpy as np 
from tqdm import tqdm 

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import streamlit as st



###### load resnet50 
resnet_model = ResNet50()
resnet_model = Model(inputs=resnet_model.inputs, outputs= resnet_model.layers[-2].output)

@st.cache_resource
def load_model():
    model = load_model('best_model.h5')
    return model 
model = load_model()

# Load tokenizer 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 35


#### genrate caption
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenzier, max_length):
    in_text = "startseq"  # add start tag for generation
    for i in range(max_length):
        sequence = tokenzier.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        #convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        in_text += " " + word
        
        # stop if we reach end
        if word == "endseq":
            break
    return in_text



####### test part
st.write("""
         # Image Captioning for Flickr dataset
         """)
         
file = st.file_uploader("Please upload an Image", type=["jpg", "png", "jpeg", "jfif"])

if file is None:
    st.text("Please upload image")
    
else:
    #image_path = "E:/omar/AiProjects/img_caption/Images/3637013_c675de7705.jpg"
    image = load_img(file, target_size=(224,224))
    st.image(image, use_column_width=True)
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image)
    feature = resnet_model.predict(image, verbose=0)
    
    #predict_caption(model, feature, tokenizer, max_length)
    
    output = "This image caption most like is " + predict_caption(model, feature, tokenizer, max_length)
    st.success(output)


