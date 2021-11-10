import streamlit as st
st.title("Password Strength Prediction")
import numpy as np
import pandas as pd

import pickle

xgb_classifier=pickle.load(open('passwordstrength.pkl','rb'))
df=pd.read_csv('data.csv',',',error_bad_lines=False)
df.dropna(inplace=True)
import random
password_tuple=np.array(df)
random.shuffle(password_tuple)
y=[labels[1] for labels in password_tuple]
x=[passwords[0] for passwords in password_tuple]
predict=np.array(['abc1234'])
def word_divide(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(tokenizer=word_divide)
x=vector.fit_transform(x)
input=st.text_input("Enter password")
st.text("you entered {}".format(input))
if st.button('Check the Strength'):
    predict=np.array([input])
    predict=vector.transform(predict)
    z=predict.shape[1]
    st.text(z)
    predict=np.reshape(predict,(1,z))
    y_pred=xgb_classifier.predict(predict)
    st.text("0 for weak, 1 for average, 2 for strong")
    st.text(y_pred)

