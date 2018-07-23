setwd("C:\\PRSH Downloads\\Stanford ISLR\\Kaggle_Stacking")
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(nnet)
library(MLmetrics)
library(ggplot2)
library(kernlab)

##....................................................................................................##
##Impute_Missing function
##Method: Create 10 groups in the respective variable and check groups' behaviour wrt the target variable 
##Check the behaviour of missing group wrt target variable as well. Replace all missing values with the mean 
#value of the group behaving similarly to that of missing group .

##Note:This method uses target variable for imputation; Practically, there are many other methods to impute like knn, bagTree, EM algorithm
## This method was used for learning purposes and as outcome variable is not there for completely new data, may not be the nest/reproducible on new data
##.....................................................................................................##

impute_missing<-function(x){
  data<-x
  for (i in 1:ncol(data)){
    if(any(is.na(data[,i]))){
      if(class(data[,i])=="numeric" | class(data[,i])=="integer"){
        data%>%group_by(quantile=ntile(data[,i],10))->data2
        tbl1<-table(data2$quantile,data2$churn)/(nrow(data2)/10)
        index<-which(is.na(data2[,i]))
        tbl2<-table(data2$churn[index])/length(index)
        quant<-quantile(data[,i],p=(0:10)/10,na.rm=T)
        condition1<-tbl1[,1]>0.9*tbl2[1] & tbl1[,1]<1.1*tbl2[1]
        if(any(condition1) && sum(condition1)>1){
          diff<-abs(tbl1[condition1,1]-tbl2[1])
          condition2<-diff==min(diff)
          ind<-as.numeric(names(condition2[condition2]))
        } 
        else{
          ind<-which(condition1)
        }
        condition<-length(ind)==1
        if (condition){
          data[,i][index]<-(quant[ind+1]+quant[ind])/2
        }
      } else if(class(data[,i])=="character"){
        tbl1<-table(data[,i],data$churn)
        good_rate<-tbl1[,1]/rowSums(tbl1)
        index<-which(is.na(data2[,i]))
        tbl2<-table(data2$churn[index])/length(index)
        
        condition1<-good_rate>0.9*tbl2[1] & good_rate<1.1*tbl2[1]
        if(any(condition1) && sum(condition1)>1){
          diff<-abs(good_rate[condition1]-tbl2[1])
          condition2<-diff==min(diff)
          ind<-names(condition2[condition2])
        }
        else{
          ind<-names(condition1)
        }
        condition<-length(ind)==1
        if (condition){
          data[,i][index]<-ind
        }
      }else{
        data
      }
    }
  }
  data
  }

##....................................................................................................##
##Outlier treatment::delete_outliers function
##Note:This is a generic method used for learning purpose; Preactically, one would talk to 
#client and then treat outliers as per business rationale
##....................................................................................................##

delete_outliers<-function(x){
  data<-x
  for (i in 1:ncol(data)){
    if(class(data[,i])=="numeric" | class(data[,i])=="integer"){
      quantile<-quantile(data[,i],prob=c(0.25,0.75),na.rm=T)
      q_range<-5*IQR(data[,i])
      Min<-quantile[1]-q_range
      Max<-quantile[2]+q_range
      data[,i][data[,i]>Max]<-NA
      data[,i][data[,i]<Min]<-NA
      
    }else{
      x
    }
  }
  data
}

##Reading the data

data<-read.csv("telecomfinal.csv",sep=",",na.strings = c("NA","UNKNOWN"," "))

##Checking the Number of missing values in the data

Missing<-lapply(data,function(x)sum(is.na(x)))

##Deleting columns that have more than 5% of the records missing

missing<-Missing[Missing>=0.05*nrow(data)]

data1<-data[,!(colnames(data) %in% names(missing))] 

summary(data1)

str(data1)

##Convert customerID into character type

data1$Customer_ID<-as.character(data1$Customer_ID)

##Group factors with many levels into factors with 3-4 levels

lapply(data1,function(x)sum(is.na(x)))

length(unique(data1$crclscod))

table=table(data1$crclscod,data1$churn)

table/rowSums(table) ##the behaviour of crclscod wrt to churn

#We can group these 53 levels ofcrclscod into 3-5 levels

##Grouping crclscod using vectorization
lookup=c("A"="2", "A2"="2", "AA"="2", "B"="2",  "B2"="2", "BA"="2", "C"="2",  "C2"="2",
         "CA"="2", "CC"="2", "CY"="2", "D"="2",  "D2"="2", "DA"="2", "E"="2",  "EA"="2",
         "EF"="2", "EM"="2", "G"="2",  "GA"="2", "GY"="2", "I"="2",  "IF"="2", "J"="2",
         "JF"="2", "K"="2","L"="2",  "M"="2",  "O"="2",  "P1"="2", "U"="2",  "U1"="2",
         "Z"="2",  "Z1"="2", "Z2"="2", "Z4"="2", "ZA"="2", "ZY"="2","A3"="1","TP"="1",
         "C5"="3", "D4"="3", "D5"="3", "E2"="3", "E4"="3", "EC"="3", "H"="3",  "S"="3",
         "V1"="3", "W"="3",  "Y"="3",  "Z5"="3", "ZF"="3")

data1$crclscod_d=lookup[data1$crclscod]

data1=select(data1,-"crclscod")

#Grouping ethnic variable factor levels into 3 groups

unique(data1$ethnic)

table1=table(data1$ethnic,data1$churn)

table1/(rowSums(table1))

lookup1=c("B"="3", "D"="3", "F"="3", "G"="3", "H"="3", "I"="3", "J"="3", "N"="3",
          "O"="3", "R"="3", "S"="3", "U"="3","M"="2", "P"="2", "X"="2", "Z"="2", "C"="1")

data1$ethnic_d=lookup1[data1$ethnic]

data1=select(data1,-"ethnic")

#Grouping area variable into 3 groups

table2=table(data1$area,data1$churn)

table2/rowSums(table2)

lookup2=c("ATLANTIC SOUTH AREA"="1", "CENTRAL/SOUTH TEXAS AREA"="1", "DC/MARYLAND/VIRGINIA AREA"="1",
          "GREAT LAKES AREA"="1", "HOUSTON AREA"="1", "MIDWEST AREA"="1", "OHIO AREA"="1",
          "TENNESSEE AREA"="1","CALIFORNIA NORTH AREA"="2", "CHICAGO AREA"="2", "DALLAS AREA"="2",
          "LOS ANGELES AREA"="2", "NEW ENGLAND AREA"="2","NEW YORK CITY AREA"="2","NORTH FLORIDA AREA"="2",
          "PHILADELPHIA AREA"="2","SOUTHWEST AREA"="2","NORTHWEST/ROCKY MOUNTAIN AREA"="3",
          "SOUTH FLORIDA AREA"="3")

data1$area_d=lookup2[data1$area]

data1=select(data1,-"area")

## Variable csa has too many levels and it is very unique as it's a locality;remove

data1=select(data1,-"csa")

##Impute Data1 using above function; replace missing values of mtrcycle, truck, marital by mode; 
##Omit other NA values as they are extemely small in number and could be unique 

data_imputed<-impute_missing(data1)
data_imputed$mtrcycle[is.na(data_imputed$mtrcycle)]<- 0
data_imputed$truck[is.na(data_imputed$truck)]<- 0
data_imputed$marital[is.na(data_imputed$marital)]<- 'U'
data_imputed=na.omit(data_imputed)

##eqpdays & totmrc_Mean variables have negative values; replace with some other value

data_imputed$eqpdays[data_imputed$eqpdays<0]<-10
data_imputed$totmrc_Mean[data_imputed$totmrc_Mean<0]<-0

##----------------------------------------------------------------------------------------------
##Feature Engineering
##feature_Engg_param function will be used for methods like Logistic Regression etc,
##where we have to perform One Hot Encoding, variable transformations etc.
##Feature_Engg_nonparam will be used for Trees, Xgboost, Random Forest etc.
##We will create new bucket variables, for some variables which have extremely skewed data and couldn't be be transformed
##For some variables like Income, we will create binary variable missing_income(yes/no) to see any pattern
## We will create recency & relative_usage variables which tell how much customer used the services in past 3 months wrt to last 6 months or lifetime usage 
##We will create non_optimal_plan variable to understand how much are the overusage charages as a percentage of total charges to customes
##----------------------------------------------------------------------------------
feature_Engg_param<-function(df_orig, df_imputed){
  
  data1<-df_orig
  
  data2<-df_imputed
  
  data1<-data1[data1$Customer_ID %in% data2$Customer_ID,]
  
  data2$missing_income<-ifelse(is.na(data1$income),'yes','no')
  
  data2$non_optimal_plan<-(data2$ovrrev_Mean/data2$avgrev)*100
  
  data2$recent_usage<-ifelse(data2$avg3mou>data2$avg6mou,'higher','lower')
  
  data2$relative_usage<-ifelse(data2$mou_Mean>data2$avgmou,'higher','lower')
  
  #Creating bucket variables for variables related to service quality
  #Variables: drop_blk_Mean, comp_vce_Mean, plcd_vce_Mean etc.
  
  #1 drop_blk_Mean
  ind1=(data2$drop_blk_Mean<median(data2$drop_blk_Mean))
  
  ind2=(data2$drop_blk_Mean >= median(data2$drop_blk_Mean) & 
    data2$drop_blk_Mean< quantile(data2$drop_blk_Mean)[4])
  
  ind3=(data2$drop_blk_Mean >= quantile(data2$drop_blk_Mean)[4])
  
  data2$blocked_voice_calls[ind1]='low'
  
  data2$blocked_voice_calls[ind2]='medium'
  
  data2$blocked_voice_calls[ind3]='high'
  
  #2 comp_vce_Mean
  ind1=(data2$comp_vce_Mean<median(data2$comp_vce_Mean))
  
  ind2=(data2$comp_vce_Mean >= median(data2$comp_vce_Mean) & 
          data2$comp_vce_Mean< quantile(data2$comp_vce_Mean)[4])
  
  ind3=(data2$comp_vce_Mean >= quantile(data2$comp_vce_Mean)[4])
  
  data2$completed_voice_calls[ind1]='low'
  
  data2$completed_voice_calls[ind2]='medium'
  
  data2$completed_voice_calls[ind3]='high'

  #3 drop_vce_Range
  ind1=(data2$drop_vce_Range<median(data2$drop_vce_Range))
  
  ind2=(data2$drop_vce_Range >= median(data2$drop_vce_Range) & 
          data2$drop_vce_Range< quantile(data2$drop_vce_Range)[4])
  
  ind3=(data2$drop_vce_Range >= quantile(data2$drop_vce_Range)[4])
  
  data2$dropped_voice_range[ind1]='low'
  
  data2$dropped_voice_range[ind2]='medium'
  
  data2$dropped_voice_range[ind3]='high'
  
  #4 plcd_vce_Mean
  ind1=(data2$plcd_vce_Mean< median(data2$plcd_vce_Mean))
  
  ind2=(data2$plcd_vce_Mean >= median(data2$plcd_vce_Mean) & 
          data2$plcd_vce_Mean< quantile(data2$plcd_vce_Mean)[4])
  
  ind3=(data2$plcd_vce_Mean >= quantile(data2$plcd_vce_Mean)[4])
  
  data2$placed_voice_calls[ind1]='low'
  
  data2$placed_voice_calls[ind2]='medium'
  
  data2$placed_voice_calls[ind3]='high'
  
  ##Coverting character variables into Factors
  
  vars_=c('dropped_voice_range','completed_voice_calls','blocked_voice_calls','relative_usage',
        'recent_usage','missing_income','area_d','ethnic_d','crclscod_d','placed_voice_calls'
        )
  data2[,vars_]=lapply(data2[,vars_],factor)
  
  ## Performing One Hot Encoding 
  
  ohefit=dummyVars(form=churn~.,data = data2[,-58], fullRank = TRUE)
  data2_d=as.data.frame(predict(ohefit, newdata=data2[,-58])) #this returns a matrix which needs to be converted
  
  ## Standardizing variables and performing Yeo Johnson transformation
  ## Yeo Johnson transformation makes data normally distributed and works well on negative as well as zero values
  
  prepfit=preProcess(data2_d, method = c('center','scale','YeoJohnson'))
  data2_d_=predict(prepfit, data2_d)
  
  id=data2[,'Customer_ID']
  
  data2_d_$churn<-as.factor(data2$churn)
  
  ##Splitting the dataset for training and testing.
  
  set.seed(3)
  
  inTraining<-createDataPartition(data2_d_$churn,p=0.8,list=F)
  
  train=data2_d_[inTraining,]
  
  test=data2_d_[-inTraining,]
  
  list(id=id, train=train, test=test,data=data2)
}

#--------------------------------------------------------------------------------------------
##Feature Engineering for non-parametric methods like Tree, Xgboost, gbm

feature_Engg_nonparam<-function(df_orig, df_imputed){
  
  data1<-df_orig
  
  data2<-df_imputed
  
  data1<-data1[data1$Customer_ID %in% data2$Customer_ID,]
  
  data2$missing_income<-ifelse(is.na(data1$income),'yes','no')
  
  data2$non_optimal_plan<-(data2$ovrrev_Mean/data2$avgrev)*100
  
  data2$recent_usage<-ifelse(data2$avg3mou>data2$avg6mou,'higher','lower')
  
  data2$relative_usage<-ifelse(data2$mou_Mean>data2$avgmou,'higher','lower')
  
  #Creating bucket variables for variables related to service quality
  
  #1 drop_blk_Mean
  ind1=(data2$drop_blk_Mean<median(data2$drop_blk_Mean))
  
  ind2=(data2$drop_blk_Mean >= median(data2$drop_blk_Mean) & 
          data2$drop_blk_Mean< quantile(data2$drop_blk_Mean)[4])
  
  ind3=(data2$drop_blk_Mean >= quantile(data2$drop_blk_Mean)[4])
  
  data2$blocked_voice_calls[ind1]='low'
  
  data2$blocked_voice_calls[ind2]='medium'
  
  data2$blocked_voice_calls[ind3]='high'
  
  #2 comp_vce_Mean
  ind1=(data2$comp_vce_Mean<median(data2$comp_vce_Mean))
  
  ind2=(data2$comp_vce_Mean >= median(data2$comp_vce_Mean) & 
          data2$comp_vce_Mean< quantile(data2$comp_vce_Mean)[4])
  
  ind3=(data2$comp_vce_Mean >= quantile(data2$comp_vce_Mean)[4])
  
  data2$completed_voice_calls[ind1]='low'
  
  data2$completed_voice_calls[ind2]='medium'
  
  data2$completed_voice_calls[ind3]='high'
  
  #3 drop_vce_Range
  ind1=(data2$drop_vce_Range<median(data2$drop_vce_Range))
  
  ind2=(data2$drop_vce_Range >= median(data2$drop_vce_Range) & 
          data2$drop_vce_Range< quantile(data2$drop_vce_Range)[4])
  
  ind3=(data2$drop_vce_Range >= quantile(data2$drop_vce_Range)[4])
  
  data2$dropped_voice_range[ind1]='low'
  
  data2$dropped_voice_range[ind2]='medium'
  
  data2$dropped_voice_range[ind3]='high'
  
  #4 plcd_vce_Mean
  ind1=(data2$plcd_vce_Mean< median(data2$plcd_vce_Mean))
  
  ind2=(data2$plcd_vce_Mean >= median(data2$plcd_vce_Mean) & 
          data2$plcd_vce_Mean< quantile(data2$plcd_vce_Mean)[4])
  
  ind3=(data2$plcd_vce_Mean >= quantile(data2$plcd_vce_Mean)[4])
  
  data2$placed_voice_calls[ind1]='low'
  
  data2$placed_voice_calls[ind2]='medium'
  
  data2$placed_voice_calls[ind3]='high'
  
  ##Coverting character variables into Factors
  
  vars_=c('dropped_voice_range','completed_voice_calls','blocked_voice_calls','relative_usage',
          'recent_usage','missing_income','area_d','ethnic_d','crclscod_d','placed_voice_calls',
          'churn')
  data2[,vars_]=lapply(data2[,vars_],factor)
  
  id=data2[,'Customer_ID']
  
  data2=data2[, names(data2)[-58]]
  
  ##Splitting the dataset into train and test datset
  
  set.seed(3)
  
  inTraining<-createDataPartition(data2$churn,p=0.8,list=F)
  
  train=data2[inTraining,]
  
  test=data2[-inTraining,]
  
  list(id=id, train=train, test=test)
}

##---------------------------------------------------------------------------------------------------
##Helper function to train the models
## We will make use of do.call to call the train function on library Caret
## We will make use of area under'ROC' curve as the evaluation metric
## We will store the predictions on each fold of training set and return then as result
##In addition we will store fitted model and return that as the value of the function
##This function is very importnat as it allows us to make the minimum changes in the 
##training controls for different methods and helps streamline the process
##-------------------------------------------------------------------------------------------------------
modelfit<-function(data,data_imputed,train_params, specific_params, 
                   other_params,nonparametric=TRUE){
  
  if (nonparametric){
    
  data_train<-feature_Engg_nonparam(data,data_imputed)$train
  
  data_test<-feature_Engg_nonparam(data,data_imputed)$test
  
  } else{
    
    data_train<-feature_Engg_param(data,data_imputed)$train
    
    data_test<-feature_Engg_param(data,data_imputed)$test
  }
  
  data_train$churn=as.factor(ifelse(data_train$churn== 0,'No','Yes'))
  
  data_test$churn=as.factor(ifelse(data_test$churn== 0,'No','Yes'))
  
  set.seed(124)
  
  model<-do.call(train,c(list(form=churn~.,data=data_train),train_params,
                         specific_params,other_params))
  
  ##Model Performance Report 
  
  predicted=predict.train(model, newdata = data_test, type='prob')
  
  pred=predict.train(model, newdata = data_test, type='raw')
  
  classes=c('No','Yes')
  
  testdata=data.frame(obs=data_test$churn,pred=pred, predicted)
  
  average_ROC=twoClassSummary(testdata, lev=classes)
  
  cat("Average CV ROC:",average_ROC[1])

  ans=list(model=model, score=average_ROC, test_probs=predicted)
  
  return(ans)
}  

##remove unneccessary objects for further analysis. 
rm(lookup,lookup2,lookup1,table,table1,table2, Missing, missing, delete_outliers)

##...................................................................................
##Using parallel Processing
##...................................................................................
library(MASS)
library(klaR)
library(doParallel)
cl <- makePSOCKcluster(2)
registerDoParallel(cl)

##-------------------------------------------------------------------------------------------------------
##Xgboost
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="xgbTree")   

CARET.TUNE.GRID <-  expand.grid(nrounds=380, 
                                max_depth= 4, 
                                eta=0.1, 
                                gamma= 0.6, 
                                colsample_bytree=0.95, 
                                min_child_weight=2,
                                subsample=0.85
                                )

MODEL.SPECIFIC.PARMS <- list(verbose=0)

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",number = 3,
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,search='grid',
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="ROC")


xgb_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
         CARET.TRAIN.OTHER.PARMS)


##-------------------------------------------------------------------------------------------------------
## Gradient Boosting Machines
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="gbm")   

CARET.TUNE.GRID <-  expand.grid(n.trees=380, 
                                interaction.depth=4, 
                                shrinkage=0.1,
                                n.minobsinnode=250)

MODEL.SPECIFIC.PARMS <- list(verbose=0)

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",number = 3,
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,search='grid',
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="ROC")


gbm_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
         CARET.TRAIN.OTHER.PARMS)

##-------------------------------------------------------------------------------------------------------
## Random Forest
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="rf")   

CARET.TUNE.GRID <-  expand.grid(mtry=10)

MODEL.SPECIFIC.PARMS <- list(verbose=0)

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",number= 3,
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,search='grid',
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="ROC",ntree=70)


rf_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
         CARET.TRAIN.OTHER.PARMS)

##-------------------------------------------------------------------------------------------------------
## Logistic Regression
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="glm")   

CARET.TUNE.GRID <-  NULL

MODEL.SPECIFIC.PARMS <- list(family='binomial')

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",number= 3,
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,metric="ROC")


lr_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
         CARET.TRAIN.OTHER.PARMS, nonparametric = FALSE)

##-------------------------------------------------------------------------------------------------------
## Naive Bayes method
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="nb")   

CARET.TUNE.GRID <-  expand.grid(fL=0, 
                                usekernel=TRUE,
                                adjust=TRUE)
                                                                

MODEL.SPECIFIC.PARMS <- NULL

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",number = 5,
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,search='grid',
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="ROC")


nb_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS)

##Normal Distribution density function gave 'Nan' values in the probability results due to underflow/overflow
##Used Kernel density function
##-------------------------------------------------------------------------------------------------------
## Linear Discriminant Analysis method
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="lda")   

CARET.TUNE.GRID <-  NULL

MODEL.SPECIFIC.PARMS <- list(verbose=0)

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",number = 10,
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,search='grid',
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="ROC")


lda_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS,nonparametric=FALSE)

##-------------------------------------------------------------------------------------------------------
## Support Vector Machines method
##-------------------------------------------------------------------------------------------------------

## set caret training parameters

CARET.TRAIN.PARMS <- list(method="svmLinear")   

CARET.TUNE.GRID <-  expand.grid(C=1)

MODEL.SPECIFIC.PARMS <- list(verbose=0)

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=TRUE,
                                 summaryFunction=twoClassSummary,search='grid',
                                 savePredictions = 'all')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric='ROC')


svm_results=modelfit(data, data_imputed,CARET.TRAIN.PARMS,MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS,nonparametric=FALSE)


##--------------------------------------------------------------------------------
##Creating Predictors for level 1 model(we will use Neural Net)
##--------------------------------------------------------------------------------

xgb_probs=xgb_results$model$pred$Yes

gbm_probs=gbm_results$model$pred$Yes

rf_probs=rf_results$model$pred$Yes

lr_probs=lr_results$model$pred$Yes

nb_probs=nb_results$model$pred$Yes

lda_probs=lda_results$model$pred$Yes

level1_train=data.frame(xgb_results$model$pred$obs,xgb_probs, gbm_probs, rf_probs,
                        lr_probs, lda_probs)

names(level1_train)=c('churn','xgb_probs', 'gbm_probs', 'rf_probs','lr_probs','lda_probs')

level1_test=data.frame(feature_Engg_nonparam(data, data_imputed)$test$churn,
                       xgb_results$test_probs$Yes,gbm_results$test_probs$Yes,
                      rf_results$test_probs$Yes, lr_results$test_probs$Yes,
                      lda_results$test_probs$Yes)

names(level1_test)=c('churn','xgb_probs','gbm_probs','rf_probs','lr_probs','lda_probs')

level1_test$churn=ifelse(level1_test$churn==1,"Yes","No")

level1_test$churn=as.factor(level1_test$churn)

##-------------------------------------------------------------------------------------------------------
## Final level 1 model Neural Net
##-------------------------------------------------------------------------------------------------------


# set caret training parameters
CARET.TRAIN.PARMS <- list(method="nnet") 

CARET.TUNE.GRID <- expand.grid(size=c(2,3,4,5),decay=c(0.001,0.01,0.1))    

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="cv",
                                 number=3,
                                 verboseIter=FALSE,classProbs=TRUE,
                                 summaryFunction=twoClassSummary,
                                 savePredictions = 'none')

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                maximize=FALSE,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="ROC",maxit=400)

MODEL.SPECIFIC.PARMS <- list(verbose=FALSE) #NULL # Other model specific parameters


# train the model

set.seed(825)

l1_nnet_mdl <- do.call(train,c(list(form=churn~.,data=level1_train),
                               CARET.TRAIN.PARMS,
                               MODEL.SPECIFIC.PARMS,
                               CARET.TRAIN.OTHER.PARMS))

predicted_prob=predict.train(l1_nnet_mdl, newdata = level1_test, type='prob')

pred=predict.train(l1_nnet_mdl, newdata = level1_test, type='raw')

classes=c('No','Yes')

testdata=data.frame(obs=level1_test$churn,pred=pred, predicted_prob)

twoClassSummary(testdata, lev=classes)

stopCluster(cl)
