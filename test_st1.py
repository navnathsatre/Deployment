import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

   
df = pd.read_csv('D:/DATA SCIENCE\Data sets/day15/claimants.csv')
df.drop(["CASENUM"],inplace=True,axis = 1)
df = df.dropna().reset_index()
df.drop(["index"],inplace=True,axis = 1)

#st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('C:/OneDrive/Desktop/Logistic_Model.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

#st.subheader('Predicted Result')
#st.write('No' if prediction_proba[0][1] > 0.5 else 'Yes')

st.subheader('Prediction Probability')
st.write(prediction_proba)

output=pd.concat([df,pd.DataFrame(prediction_proba)],axis=1)

output.to_csv('C:/output.csv')
