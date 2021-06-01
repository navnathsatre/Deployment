#%%writefile myapp.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st 
st.set_page_config(layout="wide")
html_temp = """ 
<div style ="background-color:	#98FB98;padding:13px"> 
<H1 style ="color:black;text-align:center;">Loan Status Prediction</H1> 
</div> 
"""  
st.markdown(html_temp, unsafe_allow_html = True)     # display the front end aspect
pickle_in = open('classifier.pkl', 'rb') # loading the trained model
classifier = pickle.load(pickle_in)
@st.cache()
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender,Married,Education,Self_Employed,Credit_History,Dependents,Property_Area,LoanAmount,Loan_Amount_Term,ApplicantIncome,CoapplicantIncome):   
    Dependents_0,Dependents_1,Dependents_2,Dependents_3,Property_Area_Rural,Property_Area_Semiurban,Property_Area_Urban=0,0,0,0,0,0,0
    if Dependents == '0':
        Dependents_0 = '0'
    if Dependents == '1':
        Dependents_1 = '1'
    if Dependents == '2':
        Dependents_2 = '2'
    if Dependents == '3+':
        Dependents_3 = '3+'   
    if Property_Area == 'Rural':
        Property_Area_Rural = 'Rural'
    if Property_Area == 'Semiurban':
        Property_Area_Semiurban = 'Semiurban'  
    if Property_Area == 'Urban':
        Property_Area_Urban = 'Urban'      
    if Gender == "Female":    # Pre-processing user input    
        Gender = 0
    else:
        Gender = 1
    if Married == "No":
        Married = 0
    else:
        Married = 1
    if Education == "Not Graduate":
        Education = 0
    else:
        Education = 1
    if Self_Employed == "No":
        Self_Employed = 0
    else:
        Self_Employed = 1
    if Credit_History == "Good(1)":
        Credit_History = 1
    else:
        Credit_History = 0    
    if Dependents_0  == "0":
        Dependents_0 = 1
    else:
        Dependents_0 = 0
    if Dependents_1  == "1":
        Dependents_1 = 1
    else:
        Dependents_1 = 0
    if Dependents_2  == "2":
        Dependents_2 = 1
    else:
        Dependents_2 = 0
    if Dependents_3  == "3+":
        Dependents_3 = 1
    else:
        Dependents_3 = 0
    if Property_Area_Rural == "Rural":
        Property_Area_Rural = 1
    else:
        Property_Area_Rural = 0
    if Property_Area_Semiurban == "Semiurban":
        Property_Area_Semiurban = 1
    else:
        Property_Area_Semiurban = 0    
    if Property_Area_Urban == "Urban":
        Property_Area_Urban = 1
    else:
        Property_Area_Urban = 0        
    X = np.log(np.cbrt(LoanAmount+1))
    LoanAmount = (X - -0.7675283643313486)/1.4166409313468482    #LoanAmount = X_std * (max - min) + min     
    Y = np.log(np.log(Loan_Amount_Term))
    Loan_Amount_Term = (Y - 0.9102350933653259)/0.9100771874598105    #Loan_Amount_Term = Y_std * (max - min) + min    
    TotalIncome = ApplicantIncome+CoapplicantIncome
    Z = np.log(np.log(TotalIncome+1))
    TotalIncome = (Z - 1.9843722721856991)/0.440626609448286    #TotalIncome = Z_std * (max - min) + min    
    # Making predictions 
    a = [[Gender,Married,Education,Self_Employed,Credit_History,Dependents_0,Dependents_1,Dependents_2,Dependents_3,Property_Area_Rural,Property_Area_Semiurban,Property_Area_Urban,LoanAmount,Loan_Amount_Term,TotalIncome]]
    arr = np.array(a)
    prediction = classifier.predict(arr) 
    if prediction == 0:
        pred = 'Sorry to inform your loan is **rejected**'
    else:
        pred = 'Congratulations your loan is **approved**'
    return pred

def main():       # this is the main function in which we define our webpage  (front end elements of the web page) 

    # following lines create boxes in which user can enter data required to make prediction 
    col1, col2, col3= st.beta_columns(3)
    Gender=col1.selectbox('Gender',("Male","Female"))
    Married=col2.selectbox('Marital Status',("Yes","No"))
    Dependents=col3.selectbox('Dependents',('0','1','2','3+')) 
    col4, col5, col6= st.beta_columns(3)
    Education = col4.selectbox('Education',('Graduate','Not Graduate'))
    Self_Employed = col5.selectbox('Self Employed',('No','Yes'))
    ApplicantIncome = col6.number_input("Applicant Income",min_value=150,max_value=81000)
    col7, col8, col9= st.beta_columns(3)
    CoapplicantIncome = col7.number_input("Coapplicant Income",min_value=0,max_value=50000)     
    LoanAmount = col8.number_input("Loan Amount(in thousands)",min_value=9,max_value=800)
    Loan_Amount_Term = col9.number_input('Loan Amount Term(in months)',min_value=12,max_value=500)
    col10, col11= st.beta_columns(2)
    Credit_History = col10.selectbox("Credit_History",["Good(1)","Bad(0)"])
    Property_Area = col11.selectbox('Property_Area',('Rural','Semiurban','Urban'))
    result =""
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender,Married,Education,Self_Employed,Credit_History,Dependents,Property_Area,LoanAmount,Loan_Amount_Term,ApplicantIncome,CoapplicantIncome) 
        st.success('{}'.format(result))
     
if __name__=='__main__': 
    main()

# from pyngrok import ngrok 

# public_url = ngrok.connect('8501')
# public_url

# !streamlit run myapp.py &>/dev/null&