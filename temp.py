import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

#nltk.download('punkt')
#nltk.download('stopwords')




model_type=st.sidebar.selectbox(label='Select the Model', options=['Select the Model','svm' ,'Neural Network'])
if model_type=='svm':
    pickle_in=open("postprediction.pkl","rb")
    model=pickle.load(pickle_in)
    def welcome():
        return "Welcome All"
    
    
    def predict_stock(text):
       #processed_text = preprocess_text(text)
        #vectorized_text = vectorizer.transform([processed_text])
        prediction=model.predict(text)
        print(prediction)
        return prediction
    def main():
       st.title("Prediction using svm")
       html_temp = """
       <div style="background-color:teal;padding:10px">
       <h2 style="color:white;text-align:center;">Stock Prediction Using Streamlit </h2>
       </div>
       """
       st.markdown(html_temp,unsafe_allow_html=True)
       text = st.text_input("Enter text")
      
       result=""
       if st.button("Predict"):
           result=predict_stock(text)
       st.success('The output is {}'.format(result))
    if __name__=='__main__':
        main()

                
             
            
    
