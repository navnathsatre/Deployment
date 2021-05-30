import pandas as pd
import streamlit as st 
from pickle import load
st.title('Model Deployment : Predict Loan Status')
st.sidebar.header('User Input Parameters')

def user_input_features():
    GENDER = st.sidebar.selectbox('Gender',('Male','Female'))
    MARIED = st.sidebar.selectbox('Married',('Yes','No'))
    DEPENDENTS = st.sidebar.selectbox('Dependents',('0','1','2','3+'))
    EDUCATION = st.sidebar.selectbox('Education',('Graduate','Not Graduate'))
    SELF_EMPLOYED = st.sidebar.selectbox('Self_Employed',('Yes','No'))
    APPLICANTINCOME = st.sidebar.number_input('ApplicantIncome')
    CO_APPLICANTINCOME = st.sidebar.number_input('CoapplicantIncome')
    LOANAMOUNT = st.sidebar.number_input('LoanAmount')
    LOANAMOUNT_TERM = st.sidebar.number_input('Loan_Amount_Term')
    CREDIT_HISTORY = st.sidebar.selectbox('Credit_History',('0','1'))
    PROPERTY_AREA = st.sidebar.selectbox('Property_Area',('Rural','Semiurban','Urban'))
    data = {'GENDER':GENDER,'MARIED':MARIED,'DEPENDENTS':DEPENDENTS,'EDUCATION':EDUCATION,'SELF_EMPLOYED':SELF_EMPLOYED,
            'APPLICANTINCOME':APPLICANTINCOME,'CO_APPLICANTINCOME':CO_APPLICANTINCOME,'LOANAMOUNT':LOANAMOUNT,'LOANAMOUNT_TERM':LOANAMOUNT_TERM,
           'CREDIT_HISTORY':CREDIT_HISTORY,'PROPERTY_AREA':PROPERTY_AREA}
    features = pd.DataFrame(data,index=[0])
    return features
df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)
#Load the model from disk
loded_model = load(open('Loan_status_sav.sav','rb'))

prediction = loded_model.predict(df)
st.subheader('Prediction Resulr')
st.write('Congratulations, You are eligible for loan' if 'Loan_Status'== Y else 'Sorry')    
    
    
                                  