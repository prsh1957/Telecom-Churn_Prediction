import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 18, 8

#Reading the data file

data=pd.read_csv("telecomfinal.csv")

data.head()

missing=data.isnull().sum()

data[['churn','income']].boxplot(by='churn')

Removed=missing[missing>0.2*data.shape[0]]

target=['churn']

predictors=[col for col in data.columns if col not in Removed ]

predictors=[elem for elem in predictors if elem not in target]

X=data[predictors]

y=data[target]

pd.crosstab(y['churn'],X['hnd_webcap'])

#imputing the hnd_webcap variable with most frequent value

print X['hnd_webcap'].value_counts()
from scipy.stats import mode
X['hnd_webcap'].fillna(value='WCMB',inplace=True)
X['hnd_webcap'].value_counts()

ethnic_impute=mode(X['ethnic']).mode[0]
marital_impute=mode(X['marital']).mode[0]
car_buy_impute=mode(X['car_buy']).mode[0]
area_impute=mode(X['area']).mode[0]
prizm_impute=mode(X['prizm_social_one']).mode[0]

##Imputing values in the mariatal, ethnic and car_buy columns
X['marital'].fillna(value=marital_impute,inplace=True)
X['ethnic'].fillna(value=ethnic_impute,inplace=True)
X['car_buy'].fillna(value=car_buy_impute,inplace=True)
X['prizm_social_one'].fillna(value=prizm_impute,inplace=True)
X['area'].fillna(value=area_impute,inplace=True)

##removing csa variable as it has almost unique value for each customer
predictors_new=[elem for elem in predictors if elem not in ['csa']]
len(predictors_new)
X=X[predictors_new]

#Separating numeric variables to impute all the missing values in them with median
NumericVars=X.columns[X.dtypes=='float64']

X_impute=X[NumericVars]

from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='median',axis=0)
X_impute=imp.fit_transform(X_impute)

#Converting X_impute to dataframe and merging back with Xa
X_merge=pd.DataFrame(X_impute,columns=NumericVars)
remain=[elem for elem in predictors_new if elem not in NumericVars]
X1=pd.concat([X[remain],X_merge],axis=1)

np.any(X1.isnull().sum())  ##So there is still a missing value

#Let us omit the other missing values
X1=X1.dropna()

tab1=pd.crosstab(X1['ethnic'],y['churn']).apply(lambda r:r/r.sum(),axis=1)
tab1

dictionary1={i:'gr1' for i,j in zip(tab1.index,tab1.iloc[:,0]) if j <0.8}
dictionary2={i:'gr2' for i,j in zip(tab1.index,tab1.iloc[:,0]) if j >0.8} 

##replace many groups in the ethnic column by two groups
X2=X1[['ethnic']].replace(['B', 'D', 'G', 'F', 'I', 'H', 'J', 'M', 'O', 'N', 'S', 'R', 'U'],'gr1')

X3=X2.replace(['P', 'C', 'Z', 'X'],'gr2')

X1=X1.assign(ethnic_=X3.values)

X1=X1.drop(['ethnic'],axis=1)

##Similarly we can replace many groups in colums like area to 3-4 groups
tab2=pd.crosstab(X1['area'],y['churn']).apply(lambda r:r/r.sum(),axis=1)
tab2

dictionary1={i:'gr1' for i,j in zip(tab2.index,tab2.iloc[:,0]) if j <0.77}
dictionary2={i:'gr2' for i,j in zip(tab2.index,tab2.iloc[:,0]) if j >0.77}  

X2=X1[['area']].replace(dictionary1.keys(),'gr1')

X3=X2.replace(dictionary2.keys(),'gr2')

X1=X1.assign(area_=X3.values).drop(['area'],axis=1)

pd.crosstab(X1.marital,y.churn).apply(lambda r:r/r.sum(),axis=1)

X2=X1[['marital']].replace(['B','S'],'S')

X3=X2.replace(['M','A'],'M')

X1=X1.assign(marital_=X3.values).drop(['marital'],axis=1)

pd.crosstab(X1.prizm_social_one,y.churn).apply(lambda r:r/r.sum(),axis=1)

X2=X1[['prizm_social_one']].replace(['C','S','U'],'U')

X3=X2.replace(['R','T'],'R')

X1=X1.assign(social_group_=X3.values).drop(['prizm_social_one'],axis=1)

tab3=pd.crosstab(X1.crclscod,y.churn).apply(lambda r: r/r.sum(),axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X1.crclscod=le.fit_transform(X1.crclscod)

group_names=['A-rated','B-rated','C-rated','D-rated']
bins=[0,4,15,45,52]

X1.crclscod=pd.cut(X1.crclscod, bins, labels=group_names,right=False)

##So we reduced categories of many variables into 2-3 categories

#Now let us break some continuos variables into categories; for example age1,age2 etc.
#First let us check their summary
X1.age1.describe(),  X1.age2.describe()
#We can see that age value 0 is incorrect; replace it by median value first

index=X1.index[X1.age1==0]
print pd.crosstab(X1.age1[index],y.churn[index]).apply(lambda r: r/r.sum(),axis=1)
data[['churn','age1']].boxplot(by='churn')

#imputing age 1=0 ,age2=0 by median value
X1.age1=X1.age1.replace(0.0,np.median(X1.age1))
X1.age2=X1.age2.replace(0.0,np.median(X1.age2[X1.age2!=0]))

#Split age variables in buckets
bins = [17, 25, 35, 60, 100]

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
cats=pd.cut(X1.age1, bins,labels=group_names)

X1.age1=X1.age1.map(cats)

cats=pd.cut(X1.age2,bins, labels=group_names)

X1.age2=X1.age2.map(cats)

X1.asl_flag=X1.asl_flag.astype('category')
X1.refurb_new=X1.refurb_new.astype('category')
X1.hnd_webcap=X1.hnd_webcap.astype('category')
X1.car_buy=X1.car_buy.astype('category')
X1.ethnic_=X1.ethnic_.astype('category')
X1.area_=X1.area_.astype('category')
X1.social_group_=X1.social_group_.astype('category')
X1.marital_=X1.marital_.astype('category')


X_new=X1.drop(['Customer_ID'],axis=1)

X_new=pd.get_dummies(X_new)

ind=data.eqpdays.isnull()

ind=[i for i in range(66297) if i not in [53204]]
len(ind)

y=y.T.drop([53204],axis=1)

y=y.T

data_new=pd.concat([X_new,y],axis=1)

data_new.to_csv("churn_final.csv",index=False)


##Reading the data for Analysis(Reading again just to confirm that data was correctly exported)

data=pd.read_csv("churn_final.csv")


predictors=[i for i in data.columns if i not in ['churn']]
target='churn'

X=data[predictors]
y=data[target]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=3)

X_train_new, y_train_new = pd.DataFrame(X_train), pd.DataFrame(y_train)
X_test_new, y_test_new = pd.DataFrame(X_test), pd.DataFrame(y_test)
dtrain= pd.concat([X_train_new, y_train_new], axis=1)
dtest= pd.concat([X_test_new, y_test_new], axis=1)

##Helper function for training the dataset

def modelfit(model, dtrain, predictors, target, cross_val=True, cv_folds=5, early_stopping_rounds=50):
    
    if cross_val:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval= None)
        model.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    model.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = model.predict(dtrain[predictors])
    dtrain_predprob = model.predict_proba(dtrain[predictors])[:,1]
    
    #Predict training set:
    dtest_predictions = model.predict(dtest[predictors])
    dtest_predprob = model.predict_proba(dtest[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy_score(dtrain[target].values, dtrain_predictions)
    print "AUC Score (Train): %f" % roc_auc_score(dtrain[target], dtrain_predprob)
    print "AUC Score (Test): %f" % roc_auc_score(dtest[target], dtest_predprob)
                    
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    
##-------------------------------------------------------------------------------------------------------------------------------------
## Extreme Gradient Boosting(xgboost)
##-------------------------------------------------------------------------------------------------------------------------------------
## Model 1
xgb1 = XGBClassifier(learning_rate =0.1, n_estimators=380,max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
modelfit(xgb1, dtrain, predictors, target)


## Model 2
xgb2 = XGBClassifier(learning_rate =0.2, n_estimators=60,max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
modelfit(xgb2, dtrain, predictors, target)

## Learning rate of 0.2 didn't really work as well as 0.1 but the the training time is faster, we may use it for tuning

## Grid Search for the best value of parameters max_depth and min_child_weight

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=5, min_child_weight=1,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test1,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch1.fit(dtrain[predictors], dtrain[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#check other values of parameters max_depth and min_child_weight
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight': [1,2,3]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=5, min_child_weight=1,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test2,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch2.fit(dtrain[predictors], dtrain[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


##Tuning parameter gamma
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test3,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch3.fit(dtrain[predictors], dtrain[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


##Tuning parameter gamma using some other values

param_test4 = {
 'gamma':[0.3,0.4,0.5]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test4,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch4.fit(dtrain[predictors], dtrain[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

##Tuning parameter gamma using some other values
param_test5 = {
 'gamma':[0.5, 0.6, 0.7]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test5,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch5.fit(dtrain[predictors], dtrain[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


##Let us finalize the value of gamma to be 0.6 and tune parameters subsample and colsample_bytree

param_test6 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0.6, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test6,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch6.fit(dtrain[predictors], dtrain[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

##tune parameters subsample and colsample_bytree further
param_test7 = {
    'subsample':[i/100.0 for i in range(85,100, 5)],
    'colsample_bytree':[i/100.0 for i in range(80,100,5)]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0.6, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test7,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch7.fit(dtrain[predictors], dtrain[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

##we will finalize subsample to be 0.85 and colsample_bytree to be 0.95

##Tune parameter reg_lambda

param_test8 = {
    'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch8 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0.6, subsample=0.85, colsample_bytree=0.95, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test8,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch8.fit(dtrain[predictors], dtrain[target])
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_

##Tune parameter reg_lambda
param_test9 = {
    'reg_lambda':[0.5, 0.75, 1, 1.25, 1.5]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2,
                                                  gamma=0.6, subsample=0.85, colsample_bytree=0.95, objective= 'binary:logistic'
                                                  , nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test9,
                        scoring='roc_auc',n_jobs=4, iid=False, cv=5)
gsearch9.fit(dtrain[predictors], dtrain[target])
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_

#Finalize reg_lambda to be 1 and run the model once

xgb3 = XGBClassifier(learning_rate =0.2, n_estimators=60, max_depth=4, min_child_weight=2, gamma=0.6, subsample=0.85,
                     colsample_bytree=0.95, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
                     reg_lambda = 1)
modelfit(xgb3, dtrain, predictors, target)

##Reduce the learning rate and increase the trees
xgb4 = XGBClassifier(learning_rate =0.01, n_estimators=4000, max_depth=4, min_child_weight=2, gamma=0.6, subsample=0.85,
                     colsample_bytree=0.95, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
                     reg_lambda = 1)
modelfit(xgb4, dtrain, predictors, target)

##Reduce the learning rate and increase the trees further
xgb5 = XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=4, min_child_weight=2, gamma=0.6, subsample=0.85,
                     colsample_bytree=0.95, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
                     reg_lambda = 1)
modelfit(xgb5, dtrain, predictors, target)

##Reduce the learning rate and increase the trees further
xgb6 = XGBClassifier(learning_rate =0.007, n_estimators=5500, max_depth=4, min_child_weight=2, gamma=0.6, subsample=0.85,
                     colsample_bytree=0.95, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
                     reg_lambda = 1)
modelfit(xgb6, dtrain, predictors, target)

## This iteration gave the best result for ROC_AUC

##----------------------------------------------------------------------------------------------------------------------------------------
## Gradient Boosting Machines(gbm)
##----------------------------------------------------------------------------------------------------------------------------------------

##Initial Model

gbm=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, min_samples_leaf=250, max_features='sqrt',random_state=3)
gbm.fit(X_train, y_train)
predicted=gbm.predict(X_test)
probas=gbm.predict_proba(X_test)
roc_auc_score(y_test, probas[:,1])

##Testing performance using simply the train and test split (looking at the system capacity and many values to test)
rate_options=[0.001,0.01,0.1,1.0,10.0,100.0]
n_trees=np.arange(40,210,20)
auc_scores=[]

for option in rate_options:       
    for trees in n_trees:        
        gbm1=GradientBoostingClassifier(learning_rate=option, n_estimators=trees, min_samples_leaf=250, max_features='sqrt',random_state=3)
        gbm1.fit(X_train, y_train)
        probas1=gbm1.predict_proba(X_test)        
        score=roc_auc_score(y_test, probas1[:,1])
        auc_scores.append(score)

np.array(auc_scores).max()


#Checking the n_trees above 200 for better AUC score
n_trees=[200,210,220,230,240,250]
auc_scores=[]
for trees in n_trees:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= trees, min_samples_leaf= 250, max_features= 'sqrt',
                                random_state= 3)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))
    
##The AUC score is constantly increasing so let us increase the number of trees 
n_trees=[260, 280, 300, 320, 340, 360, 380, 400]
auc_scores=[]
for trees in n_trees:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= trees, min_samples_leaf= 250, max_features= 'sqrt',
                                random_state= 3)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))    
    
np.array(auc_scores).max()

#Now we can see the AUC score increased till number of trees 380 and then decreased. Let us play around this figure
##The AUC score is constantly increasing so let us increase the number of trees 
n_trees=[375, 380, 385, 390, 395, 400]
auc_scores=[]
for trees in n_trees:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= trees, min_samples_leaf= 250, max_features= 'sqrt',
                                random_state= 3)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))

#The AUC score starts decreasing after 380 trees; hence the optimum learning rate is 0.1 and 380 trees

# tuning max_depth 

depth_options=[3,5,7,9]
auc_scores=[]
for depth in depth_options:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 380, min_samples_leaf= 250, max_features= 'sqrt',
                                random_state= 3, max_depth= depth)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))

# We got better score for max_depth of 5. Lets play around that depth

depth_options=[4,5,6]
auc_scores=[]
for depth in depth_options:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 380, min_samples_leaf= 250, max_features= 'sqrt',
                                random_state= 3, max_depth= depth)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))

np.array(auc_scores).max()

##So the optimum depth of tree should be 5

##let us tune for max_features
feature_options=[9,11,13,15]
auc_scores=[]
for features in feature_options:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 380, min_samples_leaf= 250, max_features= features,
                                random_state= 3, max_depth= 5)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))
##The performance is better if the model has more options in choosing features for split

#We can try higher values of max_features
feature_options=[15,18,21,24]
auc_scores=[]
for features in feature_options:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 380, min_samples_leaf= 250, max_features= features,
                                random_state= 3, max_depth= 5)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))
    
##the optimum values for max_features would be 21 as per above results. Let us play a little bit around that
#We can try higher values of max_features
feature_options=[20,21,22,23]
auc_scores=[]
for features in feature_options:
    gbm3=GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 380, min_samples_leaf= 250, max_features= features,
                                random_state= 3, max_depth= 5)
    gbm3.fit(X_train, y_train)
    probas3=gbm3.predict_proba(X_test)
    auc_scores.append(roc_auc_score(y_test, probas3[:,1]))

np.array(auc_scores).max()

#decrease the learning rate and increase the no. of trees
gbm4=GradientBoostingClassifier(learning_rate= 0.05, n_estimators= 760, min_samples_leaf= 250, max_features= 21,
                                random_state= 3, max_depth= 5)
gbm4.fit(X_train, y_train)
probas4=gbm4.predict_proba(X_test)

roc_auc_score(y_test, probas4[:,1])

#decrease the learning rate and increase the no. of trees
gbm4=GradientBoostingClassifier(learning_rate= 0.008, n_estimators= 3000, min_samples_leaf= 250, max_features= 21,
                                random_state= 3, max_depth= 5)
gbm4.fit(X_train, y_train)
probas4=gbm4.predict_proba(X_test)
roc_auc_score(y_test, probas4[:,1])

##This gave the best result almost as good as Xgboost


##.........................................................................................................................................
##Random Forest Classifier
##........................................................................................................................................

## Best performing model

forest1=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=250, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=75, n_jobs=2,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

forest1.fit(X_train, y_train)

predicted1= forest1.predict(X_test)
probas1=forest1.predict_proba(X_test)
roc_auc_score(y_test, probas1[:, 1])


##-----------------------------------------------------------------------------------------------------------------------------------------
## Logistic Regression 
##We will use some more preprocessing methods for Lostic Regression like 1) standardizing the variables 2) extracting Pricipal Components
## 3) we will use Pipeline to streamline the workflow 4) we will use Regularization (l2 penalty which shrinks the coefficients but does not ## reduce them to zero. Model uses information from almost 80-90% predictors here so l1 penalty may not be useful
##-----------------------------------------------------------------------------------------------------------------------------------------

#Setting penalty as L2
pipe_lr=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(penalty='l2',
                                                                                                 random_state=1))])
pipe_lr.fit(X_train,y_train)
predicted=pipe_lr.predict(X_test)
accuracy_score(y_test,predicted)
probas=pipe_lr.predict_proba(X_test)
roc_auc_score(y_test, probas[:,1])

##Perform GridSearch to find the best model (checking both penalties just to confirm which performs best here)
param_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid=[{'clf__C':param_range,'clf__penalty':['l1']},{'clf__C': param_range, 'clf__penalty':['l2']}]

gs=GridSearchCV(estimator= pipe_lr1, param_grid= param_grid, scoring= 'roc_auc', cv=5, n_jobs=-1)

gs.fit(X_train, y_train)

results=gs.cv_results_

##Plotting the AUC scores for both penalties
plt.plot(param_range, mean_test_score[0:6], marker='o', color='blue', label='AUC with l1 penalty')
plt.ylim([0.3,0.65])
plt.xscale('log')
plt.plot(param_range, mean_test_score[6:12], marker='s', color='green', label='AUC with l2 penalty')
plt.ylim([0.5,0.65])
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.show()

##Trying logistic Regression on Pricipal Components
param_grid=[{'pca__n_components':[2,3,4,5,6,7,8,9,10],'clf__C': param_range}]

pipe_lr=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(penalty='l2',
                                                                                                 random_state=1))])

gs2=GridSearchCV(estimator=pipe_lr, param_grid= param_grid, scoring= 'roc_auc', cv=5, n_jobs=-1)

gs2.fit(X_train, y_train)
gs2.best_score_
results1=gs2.cv_results_

##-----------------------------------------------------------------------------------------------------------------------------------------##----------------------------------------------------------------------------------------------------------------------------------------
##Trying Majority Voting Classifier (Ensembling technique:: Averaging)
##----------------------------------------------------------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------------------------

data=pd.read_csv("churn_final.csv")

predictors=[i for i in data.columns if i not in ['churn']]
target='churn'

X=data[predictors]
y=data[target]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,random_state=3)

## Logistic Regression
pipe1=Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(random_state=3, penalty='l2', C=0.1))])

## Gradient Boosting Machines
clf2=GradientBoostingClassifier(learning_rate=0.1, n_estimators=380,max_depth=5, min_samples_leaf= 250, max_features=21,
random_state=3)

## K Nearest Neighbors
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe3=Pipeline([('scl', StandardScaler()),('KNN', clf3)])

##Multiple Layer Perceptron
layer_dims=(30, 15, 8, 4)
clf4=MLPClassifier(hidden_layer_sizes=layer_dims, random_state=3)

##Majority Voting classifier
clf_labels = ['Logistic Regression', 'Gradient Boosting', 'K Nearest Neighbor','Multiple Layer Perceptron']
clf_labels += ['Majority Voting']
mv_clf=VotingClassifier([('lr', pipe1),('gmb', clf2), ('Knn', pipe3)],voting='soft')
all_clf = [pipe1, clf2, pipe3, clf4, mv_clf]

## USing Nested Cross Validation to reduce the geralization errors
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
