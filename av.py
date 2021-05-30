#Importing important libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# from pickle import dump
# from pickle import load
# loan = pd.read_csv("D:/PROJECTS/Loan Prediction/train.csv")
# # change the data type of Credit_History column
# loan['Credit_History'] = loan['Credit_History'].astype('O')
# #First we split numeric and chategoric columns seperately
# cat_data = []
# num_data = []
# for col,types in enumerate(loan.dtypes):
#     if types=='object':
#         cat_data.append(loan.iloc[:,col])
#     else:
#         num_data.append(loan.iloc[:,col])
# cat_data = pd.DataFrame(cat_data).T
# num_data = pd.DataFrame(num_data).T
# #missing value imputation for categorical variable using mode
# cat_data.fillna(cat_data.mode().iloc[0], inplace=True)
# cat_data.isnull().sum().any() # no more missing data 
# #missing value imputation for numeric variable using backfill (use next valid observation to fill gap)
# num_data.fillna(method='bfill', inplace=True)
# num_data.isnull().sum().any() #no more missing data
# #Add both ApplicantIncome and CoapplicantIncome to TotalIncome on train data
# num_data['TotalIncome'] = num_data['ApplicantIncome'] + num_data['CoapplicantIncome']
# num_data.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)
# # Transform the Target column
# target_values = {'Y':1,'N':0}
# target = cat_data['Loan_Status']
# y = target.map(target_values)
# cat_data.drop(columns={'Loan_Status','Loan_ID'}, axis=1, inplace=True)
# cat_data['Gender'].replace({'Male':1,'Female':0},inplace=True)
# cat_data['Married'].replace({'Yes':1,'No':0},inplace=True)
# cat_data['Education'].replace({'Graduate':1,'Not Graduate':0},inplace=True)
# cat_data['Self_Employed'].replace({'Yes':1,'No':0},inplace=True)
# cat_data['Credit_History'] = cat_data['Credit_History'].astype('int64')
# # # transform other columns
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# cat_data[['Dependents']]=le.fit_transform(cat_data[['Dependents']])
# cat_data[['Property_Area']]=le.fit_transform(cat_data[['Property_Area']])
# #cat_data = pd.get_dummies(cat_data)
# # #variable transformation
# num_data_trans = pd.DataFrame()   
# num_data_trans['LoanAmount'] =np.log(np.cbrt(num_data["LoanAmount"]+1))
# num_data_trans['Loan_Amount_Term']=np.log(np.log(num_data['Loan_Amount_Term']))
# num_data_trans['TotalIncome'] =np.log(np.log(num_data["TotalIncome"]+1))
# from sklearn.preprocessing import MinMaxScaler
# scale= MinMaxScaler()
# num_scale = pd.DataFrame(scale.fit_transform(num_data_trans),columns=num_data_trans.columns)
# final_loan = pd.concat([cat_data, num_scale, y], axis=1)
# # spliting data 
# X = final_loan.iloc[:,:-1]
# Y = final_loan.Loan_Status
# #Balancing the data
# from collections import Counter
# from imblearn.combine import SMOTETomek
# #print('Original data set shape %s' % Counter(Y))
# imb = SMOTETomek(sampling_strategy='minority',random_state=1)
# X_res,Y_res = imb.fit_resample(X,Y)
# #print('Resample data set shape %s' % Counter(Y_res))
# from sklearn.metrics import make_scorer,accuracy_score,precision_score,recall_score,f1_score
# from sklearn.model_selection import KFold, cross_val_score, cross_validate,GridSearchCV
# #Spliting Data
# from sklearn.model_selection import train_test_split
# scoring = {'accuracy' : make_scorer(accuracy_score), 
#            'precision' : make_scorer(precision_score),
#            'recall' : make_scorer(recall_score), 
#            'f1_score' : make_scorer(f1_score)}
# # #KNN
# from sklearn.neighbors import KNeighborsClassifier
# # # Using PPScore we drop some variable which has less participation in model building
# # # X_knn=X_res.drop(['Self_Employed','Dependents_0','Dependents_1','Dependents_3+',
# # #                   'Property_Area_Urban',"Gender",'Married',"Dependents_2"],axis=1)
# # #X_knn=X_res.drop(['Dependents','Property_Area'],axis=1)
# X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=20)
# # We predicted k=5 value in training
# model_KNN = KNeighborsClassifier(n_neighbors=5)
# model_KNN.fit(X_train, y_train)
# # #print(model_KNN.score(X_train, y_train))
# # #print(model_KNN.score(X_test,y_test))
# # # from sklearn.model_selection import StratifiedKFold
# # # kfold = StratifiedKFold(shuffle=True,n_splits=11, random_state=1)
# # # results = cross_validate(estimator=model_KNN,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# # # print(pd.DataFrame(results).mean())
# #SVM
# from sklearn.svm import SVC
# X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=30)
# model_SVC=SVC(C=0.7,kernel='poly',gamma=16)
# model_SVC.fit(X_train, y_train)
# # model_SVC.score(X_test,y_test)

# # #Extra Tree Classifier
# from sklearn.ensemble import ExtraTreesClassifier
# X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=40)
# model_ETC=ExtraTreesClassifier(criterion='entropy',max_features='sqrt',random_state=1,n_estimators=90,max_depth=8)
# model_ETC.fit(X_train, y_train)
# # # print(model_ETC.score(X_train, y_train))
# # # print(model_ETC.score(X_test,y_test))
# # kfold = KFold(shuffle=True, n_splits=9,random_state=30)
# # results = cross_validate(estimator=model_ETC,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# # # print(pd.DataFrame(results).mean())
# # #XGBM
# from xgboost import XGBClassifier
# X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=66)
# model_XGB=XGBClassifier(learning_rate =0.005, n_estimators=96,eval_metric='mlogloss',max_depth=8,use_label_encoder=False)
# model_XGB.fit(X_train, y_train)
# # # print(model_XGB.score(X_train, y_train))
# # # print(model_XGB.score(X_test,y_test))
# # kfold = KFold(shuffle=True, n_splits=11,random_state=30)
# # results = cross_validate(estimator=model_XGB,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# # # print(pd.DataFrame(results).mean())
# #create the ensemble model
# from sklearn.ensemble import VotingClassifier     #Heterogenious
# X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=80)
# #create the submodels
# estimators=[]
# estimators.append(('KNN', model_KNN))
# estimators.append(('SVM',model_SVC))
# estimators.append(('ETC',model_ETC))
# estimators.append(('XGB',model_XGB))
# combine_model=VotingClassifier(estimators)
# combine_model.fit(X_train, y_train)
# kfold = KFold(shuffle=True, n_splits=9,random_state=30)
# results = cross_validate(estimator=combine_model,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# print(pd.DataFrame(results).mean())
# result = combine_model.score(X_train, y_train)
# print("combine_model score :",result)

# # saving the model 
# import pickle 
# pickle_out = open("classifier_pkl.pkl", mode = "wb") 
# pickle.dump(combine_model, pickle_out) 
# pickle_out.close()

# #%%writefile app.py 
import pickle
import streamlit as st 
# loading the trained model
pickle_in = open('classifier_pkl.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender,Married,Dependents,Education,Self_Employed,TotalIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):   
    # Pre-processing user input    
    if Gender == "Female":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "No":
        Married = 0
    else:
        Married = 1
        
    if Dependents  == "0":
        Dependents = 0
    elif Dependents == "1":
        Dependents = 1
    elif Dependents == "2":
        Dependents = 2
    else:
        Dependents = 3
    
    if Education == "Not Graduate":
        Education = 0
    else:
        Education = 1
        
    if Self_Employed == "No":
        Self_Employed = 0
    else:
        Self_Employed = 1
        
    TotalIncome = TotalIncome
    LoanAmount = LoanAmount
    Loan_Amount_Term = Loan_Amount_Term
    
    if Credit_History == "Unclear Debts(0)":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    if Property_Area == "Rural":
        Property_Area = 0
    elif Property_Area == "Semiurban":
        Property_Area = 1
    else:
        Property_Area = 2
 
    # Making predictions 
    a = [[Gender, Married, Dependents, Education, Self_Employed, np.array(TotalIncome),np.array(LoanAmount), np.array(Loan_Amount_Term), Credit_History, Property_Area]]
    arr = np.array(a)
    prediction = classifier.predict(arr)
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:tomato;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Yes","No")) 
    Dependents = st.selectbox('Dependents',('0','1','2','3+'))
    Education = st.selectbox('Education',('Graduate','Not Graduate'))
    Self_Employed = st.selectbox('Self_Employed',('No','Yes'))
    TotalIncome = st.slider('Enter applicant & co-applicant income', min_value=0, max_value=90000, step=1)
    #TotalIncome = st.number_input("TotalIncome") 
    LoanAmount = st.number_input("loan amount")
    Loan_Amount_Term = st.number_input('Loan_Amount_Term')
    Credit_History = st.selectbox('Credit_History',("Unclear Debts(0)","Clear Debts(1)"))
    Property_Area = st.selectbox('Property_Area',('Rural','Semiurban','Urban'))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender,Married,Dependents,Education,Self_Employed,TotalIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area) 
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()







