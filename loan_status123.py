import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in=open("classifier.pkl","rb")

classifier=pickle.load(pickle_in)

def predict_loan_status(final_loan_test):
    prediction=classifier.predict(final_loan_test)
    print(prediction)
    if prediction==0:
     prediction="Sorry to inform your loan is rejected"
    else:
     prediction="Congratulations your loan is approved"
    return prediction



def main():
    ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History=0,0,0,0,0
    st.title("Streamlit (ML App)")
    html_temp="""    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">Your Loan Status</h1>
    </div>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Gender = st.sidebar.selectbox("Gender",("Male","Female"))
    Married = st.sidebar.selectbox("Married",("Yes","No"))
    Dependents = st.sidebar.selectbox("Dependents",("0","1","2","3+"))
    Education = st.sidebar.selectbox("Education",("Graduate","Not Graduate"))
    Self_Employed = st.sidebar.selectbox("Self_Employed",("Yes","No"))
    ApplicantIncome = st.sidebar.number_input("ApplicantIncome",min_value=150,max_value=81000)
    CoapplicantIncome = st.sidebar.number_input("CoapplicantIncome",min_value=0,max_value=50000)
    LoanAmount = st.sidebar.number_input("LoanAmount",min_value=9,max_value=800)
    Loan_Amount_Term = st.sidebar.number_input("Loan_Amount_Term",min_value=12,max_value=500)
    Credit_History = st.sidebar.radio("Credit_History",["Good","Bad"])
    Property_Area = st.sidebar.selectbox("Property_Area",("Urban","Rural","Semiurban"))
    #Loan_Status = st.selectbox("Loan_Status",())   
    
    TotalIncome=ApplicantIncome+CoapplicantIncome
    num_data_t=pd.DataFrame({"LoanAmount":[LoanAmount],"Loan_Amount_Term":[Loan_Amount_Term],"TotalIncome":[TotalIncome]})
    cat_data_t=pd.DataFrame({"Gender":[Gender],"Married":[Married],"Dependents":[Dependents],"Education":[Education],"Self_Employed":[Self_Employed],"Credit_History":[Credit_History],"Property_Area":[Property_Area]})
    

    cat_data_t1=pd.DataFrame({"Gender":[Gender],"Married":[Married],"Education":[Education],"0":0,"1":0,"2":0,"3+":0,"Self_Employed":[Self_Employed],"Credit_History":[Credit_History],"Urban":0,"Rural":0,"Semiurban":0})
    print(cat_data_t)
    print(num_data_t)
    cat_data_t1['Gender']=cat_data_t['Gender'].replace({'Male':1,'Female':0})
    cat_data_t1['Credit_History']=cat_data_t['Credit_History'].replace({'Good':1,'Bad':0})
    cat_data_t1['Married']=cat_data_t['Married'].replace({'Yes':1,'No':0})
    cat_data_t1['Education']=cat_data_t['Education'].replace({'Graduate':1,'Not Graduate':0})
    cat_data_t1['Self_Employed']=cat_data_t['Self_Employed'].replace({'Yes':1,'No':0})
    Dependents=cat_data_t['Dependents']
    cat_data_t1[Dependents]=1
    Property_Area=cat_data_t["Property_Area"]
    cat_data_t1[Property_Area]=1
    #cat_data_t['Credit_History'] = cat_data_t['Credit_History'].astype('int64')
    
    #cat_data_t = pd.get_dummies(cat_data_t)
    
     
    
    num_data_trans_t = pd.DataFrame()   
    num_data_trans_t['LoanAmount'] =np.log(np.cbrt(num_data_t["LoanAmount"]+1))
    num_data_trans_t['Loan_Amount_Term']=np.log(np.log(num_data_t['Loan_Amount_Term']))
    num_data_trans_t['TotalIncome'] =np.log(np.log(num_data_t["TotalIncome"]+1))
    
    print(num_data_trans_t)
    num_scale_t=pd.DataFrame()
    X_std=(num_data_trans_t['LoanAmount']-0.7675283643313486)/1.4166409313468482
    #X_scaled = X_std *1.1420337352027754 + 1.1488093008679707
    num_scale_t['LoanAmount']=X_std
    
    X_std=(num_data_trans_t['Loan_Amount_Term']-0.9102350933653259)/0.9100771874598105
    #X_scaled = X_std *0.6995737799386361 + 1.2711498482847257
    num_scale_t['Loan_Amount_Term']=X_std
    X_std=(num_data_trans_t['TotalIncome']-1.9843722721856991)/0.440626609448286
    #X_scaled=X_std*0.3966034460097756+2.113176026085734
    num_scale_t['TotalIncome'] =X_std

    final_loan_test = pd.concat([cat_data_t1, num_scale_t], axis=1)
    cols=list(["Gender","Married","Education","Self_Employed","Credit_History","0","1","2","3+","Rural","Semiurban","Urban","LoanAmount","Loan_Amount_Term","TotalIncome"])
    final_loan_test=final_loan_test[cols[0:]]
    print(final_loan_test)
    
    
    
    result=""
    if st.button("Predict"):
        result=predict_loan_status(final_loan_test)
    
    st.success(' {}'.format(result))
    
if __name__ == "__main__":
    main()








