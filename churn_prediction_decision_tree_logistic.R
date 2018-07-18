## Defining Working Directory 
setwd("C:\\PRSH Downloads\\Course Topics_R\\Capstone_Project_R")

library(dplyr)
library(ggplot2)
library(ROCR)
library(car) #This is to check the vif factor
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(caret)
library(pls)
library(ggfortify)

##....................................................................................................##
##Impute_Missing function
##Method: Create 10 groups in the respective variable and check groups' behaviour wrt the target variable 
##Check the behaviour of missing group wrt target variable as well. Replace all missing values with the mean 
#value of the group behaving similarly to that of missing group .
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
##....................................................................................................##

delete_outliers<-function(x){
  data<-x
  for (i in 1:ncol(data)){
    if(class(data[,i])=="numeric" | class(data[,i])=="integer"){
      quantile<-quantile(data[,i],prob=c(0.25,0.75),na.rm=T)
      q_range<-2*IQR(data[,i])
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

data<-read.csv("telecomfinal.csv",sep=",",as.is=T,stringsAsFactors = F,na.strings = c("NA","UNKNOWN"," "))

##Checking the Number of missing values in the data

Missing<-lapply(data,function(x)sum(is.na(x)))

##Deleting columns that have more than 5% of the records missing

missing<-Missing[Missing>=0.05*nrow(data)]

data1<-data[,!(colnames(data) %in% names(missing))] 

summary(data1)

##Converting churn into factor & customer ID into character

data1$churn<-as.factor(data1$churn)

data1$Customer_ID<-as.character(data1$Customer_ID)


##Create boxplots of data usage variables

par(mfrow=c(1,5))

boxplot(data1$datovr_Mean,col="blue")
boxplot(data1$datovr_Range,col="blue")
boxplot(data1$drop_dat_Mean,col="blue")
boxplot(data1$opk_dat_Mean,col="blue")
boxplot(data1$blck_dat_Mean,col="blue")
dev.off()

##We can see that all data usage variables have extremely skewed data;removing them 

data1<-select(data1,-c("datovr_Mean","datovr_Range","drop_dat_Mean","opk_dat_Mean","blck_dat_Mean"))

##mtrcycle & truck,forgntvl etc. variables have skewed data;removing them 

boxplot(data1$mtrcycle,col="blue")
boxplot(data1$truck,col="blue")
boxplot(data1$mou_pead_Mean,col="blue")
boxplot(data1$forgntvl,col="blue")
boxplot(data1$recv_sms_Mean,col="blue")
dev.off()

data1<-select(data1,-c("mtrcycle","truck","forgntvl","csa","mou_pead_Mean","recv_sms_Mean"))

Missing<-lapply(data1,function(x)sum(is.na(x)))  #check missing values again

missing<-Missing[Missing>0]        #22 variables have missing values


##Impute Data1

data_imputed1<-impute_missing(data1)

Missing<-lapply(data_imputed1,function(x)sum(is.na(x)))  #check missing values again

missing<-Missing[Missing>0]        #13 variables have missing values


##As we can see the above function worked on 10 columns.
##let us delete those missing records from other numeric variables  

data_imputed1%>%filter(!(is.na(rev_Range)))->data_imputed1_mod1

data_imputed1_mod1%>%filter(!(is.na(eqpdays)))->data_imputed1_mod1

data_imputed1_mod1%>%filter(!(is.na(change_mou)))->data_imputed1_mod1



#Defining outliers as missing values and again passing them to impute missing function

data_imputed1_mod3<-delete_outliers(data_imputed1_mod1)

data_imputed1_mod4<-impute_missing(data_imputed1_mod3)

Missing_new<-lapply(data_imputed1_mod4,function(x)sum(is.na(x))) #Checking missing values again

missing_new<-Missing_new[Missing_new>0]                  # no missing values

##...................................................................................................##
#delete_missing function, if we cannot impute remaining missing values
##...................................................................................................##

delete_missing<-function(x){
  data<-x
  for (i in 1:ncol(data)){
    if(any(is.na(data[,i]))){
      data%>%filter(!is.na(data[,i]))->data
    }
  }
  data
}

##Let us pass the dataset to delete missing function. it will run only if.na=TRUE

data_final<-delete_missing(data_imputed1_mod4)

##Looking at the behaviour of crclscod with churn

table1<-table(data_final$crclscod,data_final$churn)

table2<-table1/rowSums(table1)

##derived variable->non_optimal plan

data_final$non_optimal_plan<-(data_final$ovrrev_Mean/data_final$avgrev)*100

data_final$billing_rev_accuracy<-data_final$adjrev/data_final$totrev

data_final$recency<-data_final$avg3mou/data_final$avg6mou

data_final$relative_usage<-data_final$mou_Mean/data_final$avgmou

data_final<-delete_missing(data_final) #some values are missing now, delete

data_final<-data_final[!(data_final$recency==Inf),] #some values are Inf, delete

##Analyse variable behaviour for some variables after removing outliers

ggplot(data_final,aes(x=churn,y=mou_Mean))+geom_boxplot(colour="red")  #average mou_Mean is less for churned customers

ggplot(data_final,aes(x=churn,y=adjmou))+geom_boxplot(colour="red")

ggplot(data_final,aes(x=churn,y=change_mou))+geom_boxplot(colour="red")

ggplot(data_final)+geom_count(aes(y=churn,x=crclscod))

ggplot(data_final,aes(x=mou_Mean,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=100)

ggplot(data_final,aes(x=eqpdays,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=100)

ggplot(data_final[data_final$change_mou>0,],aes(x=change_mou,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=40)

ggplot(data_final[data_final$change_mou<=0,],aes(x=change_mou,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=40)

ggplot(data_final,aes(x=mou_Range,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=50)

ggplot(data_final,aes(x=months,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=3)

ggplot(data_final[data_final$ovrmou_Mean>40,],aes(x=ovrmou_Mean,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=0.5)

ggplot(data_final,aes(x=avg6mou,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=100)

ggplot(data_final,aes(x=plcd_vce_Mean,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=20)
ggplot(data_final,aes(x=churn,y=plcd_vce_Mean))+geom_boxplot(colour="green")

ggplot(data_final,aes(x=drop_vce_Range,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=2)
ggplot(data_final,aes(x=churn,y=drop_vce_Range))+geom_boxplot(colour="red")

ggplot(data_final,aes(x=adjmou,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=500)
ggplot(data_final,aes(x=churn,y=adjmou))+geom_boxplot(colour="green")

ggplot(data_final,aes(x=recency,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=0.3)
ggplot(data_final,aes(x=churn,y=recency))+geom_boxplot(colour="green")

ggplot(data_final,aes(x=relative_usage,y=..density..))+geom_freqpoly(mapping=aes(color=churn),binwidth=0.3)
ggplot(data_final,aes(x=churn,y=relative_usage))+geom_boxplot(colour="green")

ggplot(data_final)+geom_count(aes(y=churn,x=marital))


##Finding correlations to understand which variables to select for the model

Correlation<-cor(data_final[,c(1:29,36:42,44:50)])


##Considering correlations between variables if we pick, mou_Mean, we have to leave out avgmou,avg3mou,
##avg3qty,plcd_vce_Mean,comp_vce_Mean,avgqty,avg6mou,avg6qty. If we pick totcalls, we leave out adjqty
##and avg3qty. We can also use "vif" function to do this for us.


##Creating train and test datasets
set.seed(200)
index<-sample(nrow(data_final),0.70*nrow(data_final),replace=F)
train<-data_final[index,]
test<-data_final[-index,]

##...................................................................................................##
#Let us build a decision tree model
##...................................................................................................##


#Creating tree model

train$Target<-factor(train$churn,levels=c("1","0")) #let us try changing the levels of churn

tree_model<-rpart(Target~.,data=train[,-c(43,51)],control=rpart.control(cp=0.0001),method="class",
                  parms=list(split="gini"))

summary(tree_model)  # Tree has more than 1300 terminal nodes

fancyRpartPlot(tree_model)

plotcp(tree_model,minline=T) #To know the optimal number of tree levels

tree_model1<-prune(tree_model,cp=0.0014) #Tree has 8 terminal nodes with cost complexity parameter 0.0014

fancyRpartPlot(tree_model1)

##performance of the tree model
##checking the confusion matrix manually

test$Target<-factor(test$churn,levels=c("1","0"))

predicted_by_model=predict(tree_model1,type="class",newdata=test)

t<-table(predicted_by_model,test$churn) 

mean(predicted_by_model==test$Target) 


##As we can see, tree is not able to obtain a good 'sensitivity' in this model.


##.....................................................................................................##
##.....................................................................................................##
##Bulding a logistic regression model on the final data
##.....................................................................................................##
##.....................................................................................................##

model_logit<-glm(Target~.,data=train[,-c(43,51)],family="binomial")

summary(model_logit)

vif(model_logit) # with vif factor, we can find the multicollinearity

#Removing adjrev from the model

model_logit1<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range + adjqty+
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + comp_vce_Mean + plcd_vce_Mean
                  +avg3mou + avgmou + avg3qty + avgqty + avg6mou + avg6qty + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

summary(model_logit1)

vif(model_logit1)


#Removing avg3mou from the model

model_logit2<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range + adjqty+
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + comp_vce_Mean + plcd_vce_Mean
                  + avgmou + avg3qty + avgqty + avg6mou + avg6qty + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit2)

#Removing adjqty from the model
model_logit3<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range +
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + comp_vce_Mean + plcd_vce_Mean
                  + avgmou + avg3qty + avgqty + avg6mou + avg6qty + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit3)

#Removing comp_vce_Mean from the model
model_logit4<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range +
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + plcd_vce_Mean
                  + avgmou + avg3qty + avgqty + avg6mou + avg6qty + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit4)

#Removing avg6qty from the model
model_logit5<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range +
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + plcd_vce_Mean
                  + avgmou + avg3qty + avgqty + avg6mou  + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit5)

#Removing avgmou from the model
model_logit6<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range +
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + plcd_vce_Mean
                  + avg3qty + avgqty + avg6mou  + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit6)

#Removing avg3qty from the model
model_logit7<-glm(formula=Target ~ mou_Mean + totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range +
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + plcd_vce_Mean
                  + avgqty + avg6mou  + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit7)

#Removing mou_Mean from the model
model_logit8<-glm(formula=Target ~ totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range + owylis_vce_Range + mou_opkv_Range +
                    months + totcalls + eqpdays + custcare_Mean + callwait_Mean +
                    iwylis_vce_Mean + callwait_Range + ccrndmou_Range +
                    ovrrev_Mean + rev_Mean + ovrmou_Mean + plcd_vce_Mean
                  + avgqty + avg6mou  + 
                    crclscod + asl_flag + area + refurb_new + marital + ethnic + 
                    age1 + age2 + models + hnd_price + actvsubs + uniqsubs + 
                    roam_Mean + da_Mean + da_Range + drop_vce_Mean + adjmou + totrev+
                    avgrev + non_optimal_plan+ billing_rev_accuracy+recency+relative_usage,
                  data=train[,-c(43,51)],family="binomial")

vif(model_logit8)

summary(model_logit8)

##Now the model_logit8 does not have any variable with high covariance


#dummy variables only for significant factors
data_final$A2_d<-ifelse(data_final$crclscod=="A2",1,0)
data_final$AA_d<-ifelse(data_final$crclscod=="AA",1,0)
data_final$D5_d<-ifelse(data_final$crclscod=="D5",1,0)
data_final$E4_d<-ifelse(data_final$crclscod=="E4",1,0)
data_final$EA_d<-ifelse(data_final$crclscod=="EA",1,0)

data_final$rocky_d<-ifelse(data_final$area=="NORTHWEST/ROCKY MOUNTAIN AREA",1,0)
data_final$south_florida_d<-ifelse(data_final$area=="SOUTH FLORIDA AREA",1,0)
data_final$ohio_d<-ifelse(data_final$area=="OHIO AREA",1,0)

data_final$marital_S_d<-ifelse(data_final$marital=="S",1,0)

data_final$ethnic_C_d<-ifelse(data_final$ethnic=="C",1,0)
data_final$ethnic_F_d<-ifelse(data_final$ethnic=="F",1,0)
data_final$ethnic_I_d<-ifelse(data_final$ethnic=="I",1,0)
data_final$ethnic_N_d<-ifelse(data_final$ethnic=="N",1,0)
data_final$ethnic_S_d<-ifelse(data_final$ethnic=="S",1,0)
data_final$ethnic_U_d<-ifelse(data_final$ethnic=="U",1,0)
data_final$ethnic_Z_d<-ifelse(data_final$ethnic=="Z",1,0)

data_final$asl_flag_Y<-ifelse(data_final$asl_flag=="Y",1,0)

data_final$refurb_new_R<-ifelse(data_final$refurb_new=="R",1,0)

##Final Best Logistic Model

#Creating train and test datasets again from new dataset with dummy variables

set.seed(200)
index<-sample(nrow(data_final),0.70*nrow(data_final),replace=F)
train<-data_final[index,]
test<-data_final[-index,]

train$Target<-factor(train$churn,levels=c("0","1")) #let us change the levels of churn
test$Target<-factor(test$churn,levels=c("0","1")) #let us change the levels of churn

model_logit9<-glm(formula=Target ~ totmrc_Mean + rev_Range + mou_Range + change_mou + 
                    drop_blk_Mean + drop_vce_Range +
                    months + eqpdays + custcare_Mean  +
                    iwylis_vce_Mean + rev_Mean + plcd_vce_Mean
                  + avgqty + avg6mou +A2_d+AA_d+D5_d+E4_d+EA_d+asl_flag_Y+rocky_d+south_florida_d+ohio_d+
                    refurb_new_R+marital_S_d+ethnic_C_d+ethnic_N_d+ethnic_S_d
                  +ethnic_U_d+ethnic_Z_d+age1+ models + hnd_price + actvsubs + uniqsubs + 
                    totrev+avgrev + non_optimal_plan+recency+relative_usage+I(relative_usage^2)
                  +I(change_mou^2),
                  data=train[,-c(43,51)],family="binomial")

summary(model_logit9)

vif(model_logit9)

#checking the confusion matrix manually
probs=predict(model_logit9,type="response",newdata=test)

logit.pred<-rep("0",length(test$churn))

logit.pred[probs>0.5]<-"1"

logit.pred<-as.factor(logit.pred)

t<-table(logit.pred,test$churn) 

## The sole purpose of this model 
##is to target customers who are likely to churn so that marketing team can target them.
##Note that we can always strike a balance between sensitivity and specificity by changing cutoff prob.


#Performance of the model
library(caret)

pred<-predict(model_logit9,type="response",newdata =test)

pred1<-ifelse(pred>=0.5,1,0)

pred1<-as.factor(pred1)

confusionMatrix(pred1,test$churn,positive = "1")

library(ROCR)
library(gains)

actual<-as.numeric(as.character(test$Target))

predicted<-prediction(actual,pred1)

perf<-performance(predicted,"tpr","fpr")

plot(perf,col="red")

abline(0,1,lty=2,col="grey")

auc<-performance(predicted,"auc")

unlist(auc@y.values)

test$Target<-as.numeric(as.character(test$Target))

gains(test$Target,pred,groups=10)

##.................................................................................................##
##Customer Targeting
##.................................................................................................##

test$probs<-probs

quantile(probs,probs=seq(0.1,1,0.1))

targeted<-test[test$probs>0.7702496,] ##Separate top 50%le of customers for churn

targeted_by_revenue<-arrange(targeted,-totrev) ##Arrange these 50%le customers by revenue

targeted_customers<-targeted_by_revenue[as.numeric(row.names(targeted_by_revenue))<0.40 * nrow(targeted_by_revenue),"Customer_ID"]
##40% of 50% is 20%. Hence we obtain 20% of the total customers to be targeted.