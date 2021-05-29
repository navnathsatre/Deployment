#!/usr/bin/env python
# coding: utf-8
# To predict loan status
#Importing important libraries
import pandas as pd
import numpy as np
from pickle import dump
from pickle import load
#import matplotlib.pyplot as plt
#import seaborn as sns
#import warnings
#warnings.filterwarnings('ignore')
loan = pd.read_csv("D:/PROJECTS/Loan Prediction/train.csv")
# change the data type of Credit_History column
loan['Credit_History'] = loan['Credit_History'].astype('O')
#First we split numeric and chategoric columns seperately
cat_data = []
num_data = []
for col,types in enumerate(loan.dtypes):
    if types=='object':
        cat_data.append(loan.iloc[:,col])
    else:
        num_data.append(loan.iloc[:,col])

cat_data = pd.DataFrame(cat_data).T
num_data = pd.DataFrame(num_data).T
#missing value imputation for categorical variable using mode
cat_data.fillna(cat_data.mode().iloc[0], inplace=True)
cat_data.isnull().sum().any() # no more missing data 
#missing value imputation for numeric variable using backfill (use next valid observation to fill gap)
num_data.fillna(method='bfill', inplace=True)
num_data.isnull().sum().any() #no more missing data
#Add both ApplicantIncome and CoapplicantIncome to TotalIncome on train data
num_data['TotalIncome'] = num_data['ApplicantIncome'] + num_data['CoapplicantIncome']
num_data.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)
# Transform the Target column
target_values = {'Y':1,'N':0}
target = cat_data['Loan_Status']
y = target.map(target_values)
cat_data.drop(columns={'Loan_Status','Loan_ID'}, axis=1, inplace=True)
cat_data['Gender'].replace({'Male':1,'Female':0},inplace=True)
cat_data['Married'].replace({'Yes':1,'No':0},inplace=True)
cat_data['Education'].replace({'Graduate':1,'Not Graduate':0},inplace=True)
cat_data['Self_Employed'].replace({'Yes':1,'No':0},inplace=True)
cat_data['Credit_History'] = cat_data['Credit_History'].astype('int64')
# # transform other columns
cat_data = pd.get_dummies(cat_data)
# #variable transformation
num_data_trans = pd.DataFrame()   
num_data_trans['LoanAmount'] =np.log(np.cbrt(num_data["LoanAmount"]+1))
num_data_trans['Loan_Amount_Term']=np.log(np.log(num_data['Loan_Amount_Term']))
num_data_trans['TotalIncome'] =np.log(np.log(num_data["TotalIncome"]+1))
from sklearn.preprocessing import MinMaxScaler
scale= MinMaxScaler()
num_scale = pd.DataFrame(scale.fit_transform(num_data_trans),columns=num_data_trans.columns)
final_loan = pd.concat([cat_data, num_scale, y], axis=1)
# # spliting data 
X = final_loan.iloc[:,:-1]
Y = final_loan.iloc[:,-1]
#Balancing the data
from collections import Counter
from imblearn.combine import SMOTETomek
print('Original data set shape %s' % Counter(Y))
imb = SMOTETomek(sampling_strategy='minority',random_state=1)
X_res,Y_res = imb.fit_resample(X,Y)
print('Resample data set shape %s' % Counter(Y_res))
from sklearn.metrics import make_scorer,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate,GridSearchCV
#Spliting Data
from sklearn.model_selection import train_test_split
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
#KNN
from sklearn.neighbors import KNeighborsClassifier
# Using PPScore we drop some variable which has less participation in model building
X_knn=X_res.drop(['Self_Employed','Dependents_0','Dependents_1','Dependents_3+',
                  'Property_Area_Urban',"Gender",'Married',"Dependents_2"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X_knn,Y_res,test_size=0.20,random_state=20)
# We predicted k=5 value in training
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train, y_train)
# model_KNN.score(X_test,y_test)
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(shuffle=True,n_splits=11, random_state=1)
# results = cross_validate(estimator=model_KNN,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# pd.DataFrame(results).mean()
#SVM
from sklearn.svm import SVC
X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=30)
model_SVC=SVC(C=0.7,kernel='poly',gamma=16)
model_SVC.fit(X_train, y_train)
# model_SVC.score(X_test,y_test)
# kfold = KFold(shuffle=True, n_splits=9,random_state=30)
# results = cross_validate(estimator=model_SVC,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# pd.DataFrame(results).mean()
# #Extra Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=40)
model_ETC=ExtraTreesClassifier(criterion='entropy',max_features='sqrt',random_state=1,n_estimators=90,max_depth=8)
model_ETC.fit(X_train, y_train)
# model_ETC.score(X_test,y_test)
# kfold = KFold(shuffle=True, n_splits=9,random_state=30)
# results = cross_validate(estimator=model_ETC,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# pd.DataFrame(results).mean()
# #XGBM
from xgboost import XGBClassifier
X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=66)
model_XGB=XGBClassifier(learning_rate =0.005, n_estimators=96,eval_metric='mlogloss',max_depth=8,use_label_encoder=False)
model_XGB.fit(X_train, y_train)
# model_XGB.score(X_test,y_test)
# kfold = KFold(shuffle=True, n_splits=11,random_state=30)
# results = cross_validate(estimator=model_XGB,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# pd.DataFrame(results).mean()
#create the ensemble model
from sklearn.ensemble import VotingClassifier     #Heterogenious
X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,test_size=0.20,random_state=80)
#create the submodels
estimators=[]
estimators.append(('KNN', model_KNN))
estimators.append(('SVM',model_SVC))
estimators.append(('ETC',model_ETC))
estimators.append(('XGB',model_XGB))
combine_model=VotingClassifier(estimators)
combine_model.fit(X_train, y_train)
# combine_model.score(X_test,y_test)
# kfold = KFold(shuffle=True, n_splits=9,random_state=30)
# results = cross_validate(estimator=combine_model,X=X_res,y=Y_res, cv=kfold, scoring=scoring,return_train_score=True)
# pd.DataFrame(results).mean()
# save the model to disk
dump(combine_model, open('Loan_status_sav.sav','wb'))
# load the model from disk
loaded_model = load(open('Loan_status_sav.sav', 'rb'))
result = loaded_model.score(X, Y)
print(result)







# # # Work on test Data
# # importing test file
# loan_test = pd.read_csv('test.csv')
# # First we split numeric and chategoric columns seperately
# cat_data_t = []
# num_data_t = []
# for col,types in enumerate(loan_test.dtypes):
#     if types=='object':
#         cat_data_t.append(loan_test.iloc[:,col])
#     else:
#         num_data_t.append(loan_test.iloc[:,col])

# cat_data_t = pd.DataFrame(cat_data_t).T
# num_data_t= pd.DataFrame(num_data_t).T
# #missing value imputation for categorical variable using mode
# cat_data_t.fillna(cat_data_t.mode().iloc[0], inplace=True)
# cat_data_t.isnull().sum().any() # no more missing data 
# #missing value imputation for numeric variable using backfill (use next valid observation to fill gap)
# num_data_t.fillna(method='bfill', inplace=True)
# num_data_t.isnull().sum().any() #no more missing data
# # Add both ApplicantIncome and CoapplicantIncome to TotalIncome on train data
# num_data_t['TotalIncome'] = num_data_t['ApplicantIncome'] + num_data_t['CoapplicantIncome']
# num_data_t.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)
# cat_data_t.drop(columns={'Loan_ID'}, axis=1, inplace=True)
# cat_data_t['Gender'].replace({'Male':1,'Female':0},inplace=True)
# cat_data_t['Married'].replace({'Yes':1,'No':0},inplace=True)
# cat_data_t['Education'].replace({'Graduate':1,'Not Graduate':0},inplace=True)
# cat_data_t['Self_Employed'].replace({'Yes':1,'No':0},inplace=True)
# cat_data_t['Credit_History'] = cat_data_t['Credit_History'].astype('int64')
# # transform other columns
# cat_data_t = pd.get_dummies(cat_data_t)
# cat_data_t.head()
# num_data_trans_t = pd.DataFrame()   
# num_data_trans_t['LoanAmount'] =np.log(np.cbrt(num_data_t["LoanAmount"]+1))
# num_data_trans_t['Loan_Amount_Term']=np.log(np.log(num_data_t['Loan_Amount_Term']))
# num_data_trans_t['TotalIncome'] =np.log(np.log(num_data_t["TotalIncome"]+1))
# ### Normalizaiton of data
# num_scale_t = pd.DataFrame(scale.fit_transform(num_data_trans_t),columns=num_data_trans_t.columns)
# num_scale_t.head()
# final_loan_test = pd.concat([cat_data_t, num_scale_t], axis=1)
# final_loan_test.head()
# final_loan_test.shape
# combine_model.fit(X_res, Y_res)
# predicted_values=combine_model.predict(final_loan_test)
# predicted_loan_status=pd.DataFrame(predicted_values,columns=['Predicted_Loan_status'])
# final_prediction = pd.concat([loan_test['Loan_ID'],predicted_loan_status],axis=1)
# final_prediction

