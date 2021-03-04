import streamlit as st 
import pandas as pd 

import spacy 
import time 
from io import StringIO


def load_model():
    global nlp 
    global textcat
    nlp = spacy.load('model/model_hate_speech.h5')
    textcat = nlp.get_pipe('textcat')

def predict(hateSpeech):
    print("hateSpeech = ", hateSpeech)
    hateSpeech = [hateSpeech]
    txt_docs = list(nlp.pipe(hateSpeech))
    scores, _ = textcat.predict(txt_docs)
    print(scores)
    predicted_classes = scores.argmax(axis=-1)
    print(predicted_classes)
    result = ['Racist' if lbl == 0 else 'Not Racist' for lbl in predicted_classes]
    print(result)
    return(result)

def run():
    st.sidebar.info('You can either enter the news item online in the textbox or upload a txt file')    
    st.set_option('deprecation.showfileUploaderEncoding', False)       
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Txt file"))    
    
    st.title("Predicting Hate Speech")
    st.header('This app is created to predict if you are racist or not')
 
    if add_selectbox == "Online":
        text1 = st.text_area('Enter some text')
        output = ""
        if st.button("Predict"):
            output = predict(text1)
            output = str(output[0])  # since its a list, get the 1st item
            st.success(f"The text indicates you are {output}")     
            st.balloons()   
    elif add_selectbox == "Txt file":        
        output = ""
        file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])           
        if st.button("Predict"):
            text_news = file_buffer.read()  
            
            # in the latest stream-lit version ie. 68, we need to explicitly convert bytes to text
            st_version = st.__version__  # eg 0.67.0
            versions = st_version.split('.')           
            if int(versions[1]) > 67:
                text_news = text_news.decode('utf-8')
            
            print(text_news)
            output = predict(text_news)
            output = str(output[0])
            st.success(f"The text indicates you are {output}")      
            st.balloons()    

if __name__ == '__main__':
    load_model()
    run()