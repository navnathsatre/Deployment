import numpy as np
import pickle
import pandas as pd
import streamlit as st 
#from PIL import Image
pickle_in = open("Loan_status_sav.sav","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

def predict_note_authentication(Gender,Married,Dependents,Education):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Gender
        in: query
        type: object 
        required: true
      - name: Married
        in: query
        type: object 
        required: true
      - name: Dependents
        in: query
        type: object 
        required: true
      - name: Education
        in: query
        type: object 
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[Gender,Married,Dependents,Education]])
    print(prediction)
    return prediction



def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Gender = st.sidebar.selectbox('Gender',('Male','Female'))
    Married = st.sidebar.selectbox('Married',('Yes','No'))
    Dependents = st.sidebar.selectbox('Dependents',('0','1','2','3+'))
    Education = st.sidebar.selectbox('Education',('Graduate','Not Graduate'))
    result=''
    if st.button("Predict"):
        result=pd.DataFrame(predict_note_authentication(Gender,Married,Dependents,Education))
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()