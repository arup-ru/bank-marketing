
#########################   "Analysis of a Portugese Bank Marketing Dataset"   #########################

install.packages("ISwR")
install.packages("VIM")
install.packages("mice")
install.packages("caret")
install.packages("ROCR")
install.packages("randomForest")
install.packages("party")


# original dataset in bankA
# working dataset  in bankB (unknown <- NA; age <- age_group)
# age_group added  in bankC (22nd Column: age_group)



#########################                        #########################
          #####           Bank marketing DATASET               #####
#########################                        #########################


library(ISwR)

# Load Data in Data Frame
bankA <- as.data.frame(read.csv("C:/Users/arup.roy/Documents/bank-marketing-master/bankAdd.csv", sep= ";",header = T))

# Display the variables and first 10 records
str(bankA)
head(bankA,10)


# Replace all 'unknown' values with NA
bankB<-bankA
bankB[bankB=="unknown"]<-NA

summary(bankB$age)
#Min 17 #Max 98 #Mean 40 #Median 38


# Dividing the People into Different Age Groups
for(i in 1 : nrow(bankB)){
  if (bankB$age[i] <= 19){bankB$age_group[i] = 'Teenagers'} 
  else if (bankB$age[i] >= 20 & bankB$age[i] <= 29){bankB$age_group[i] = 'Twenties'} 
  else if (bankB$age[i] >= 30 & bankB$age[i] <= 39){bankB$age_group[i] = 'Thirties'} 
  else if (bankB$age[i] >= 40 & bankB$age[i] <= 49){bankB$age_group[i] = 'Forties'}
  else if (bankB$age[i] >= 50 & bankB$age[i] <= 59){bankB$age_group[i] = 'Fifties'}
  else if (bankB$age[i] >= 60 & bankB$age[i] <= 69){bankB$age_group[i] = 'Sixties'}
  else if (bankB$age[i] >= 70 ){bankB$age_group[i] = 'Seniors'}
}

# saving the data before replacing age_group with age
bankC<-bankB 

bankB$age<-bankB$age_group
bankB<-bankB[1:21]
bankB$age<-as.factor(bankB$age)


# Separating New Customers from the Old ones
oldCust <- subset(bankB, bankB$poutcome != "nonexistent") 
summary(oldCust) 
#05625 Old Customers

newCust <- subset(bankB, bankB$poutcome == "nonexistent") 
summary(newCust) 
#35563 New Customers



#########################                       #########################
            #####         Old Customer DATASET          #####
#########################                       #########################


# Missing value Frequencies
library(VIM) 

aggrPlot <- aggr(oldCust, col=c('darkgreen','red'), ylab=c("Missing Value Histogram","Pattern"))
#default 0.1022 #education 0.0480 #housing 0.0247 #loan 0.0247 #job 0.0065 #marital 0.0032

#Subscription Count
oldCount <- table(oldCust$y)
barplot(oldCount,col=c("red","darkgreen"),legend = rownames(oldCount), main = "Old_Subscriptions")
#no 4126 #yes 1499 


# Impute Missing Values and Check
library(mice)

oldCust2 <- mice(oldCust)
oldCust_com <- complete(oldCust2)

aggrPlot <- aggr(oldCust_com, col=c('darkgreen','red'), ylab=c("Missing Value Histogram","Pattern"))
#none


#Split data into Train and Test subsets
library(caret)
set.seed(101)

oldCust_com$y<-ifelse(oldCust_com$y =='no', 0,1)
oldCust_com$y<-as.factor(oldCust_com$y)

ids <- sample(seq(1, 2), size = nrow(oldCust_com), replace = TRUE, prob = c(.7, .3))

oldCust_train <- oldCust_com[ids==1,]
oldCust_test  <- oldCust_com[ids==2,]


table(oldCust_train$y) #no 2886 #yes 1027
table(oldCust_test$y)  #no 1240 #yes 472



#########################   Logistic Model (oldCust)   #########################


oldCust_logit <- glm(y ~., family=binomial(link='logit'), data = oldCust_train)
summary(oldCust_logit)

oldCust_logitResult <- predict(oldCust_logit, newdata=oldCust_test, type='response')
oldCust_logitResult <- ifelse(oldCust_logitResult >= 0.5,1,0)
oldCust_logitError  <- mean(oldCust_logitResult != oldCust_test$y)

print(paste('Accuracy for Logistic Model (oldCust)',1-oldCust_logitError))
#Accuracy = 84.29%

library(ROCR)

oldCust_logitPred <- prediction(oldCust_logitResult, oldCust_test$y)
oldCust_logitPerf <- performance(oldCust_logitPred, measure = "tpr", x.measure = "fpr")
plot(oldCust_logitPerf)

oldCust_logitAUC <- performance(oldCust_logitPred, measure = "auc")
oldCust_logitAUC <- oldCust_logitAUC@y.values[[1]]

print(paste('Area under the Curve for Logistic Model (oldCust)',oldCust_logitAUC))
#Area under Curve = 78.43%



#########################   Random Forest Model (oldCust)   #########################


library(randomForest)

oldCust_rf<-randomForest(y ~.,data = oldCust_train, importance=TRUE, ntree=1000)

oldCust_rfResult <- predict(oldCust_rf, oldCust_test)
oldCust_rfError  <- mean(oldCust_rfResult != oldCust_test$y)

print(paste('Accuracy for Random Forest Model (oldCust)',1-oldCust_rfError))
#Accuracy = 85.05%

library(ROCR)

oldCust_rfPred <- prediction(as.numeric(oldCust_rfResult), as.numeric(oldCust_test$y))
oldCust_rfPerf <- performance(oldCust_rfPred, measure = "tpr", x.measure = "fpr")
plot(oldCust_rfPerf)

oldCust_rfAUC <- performance(oldCust_rfPred, measure = "auc")
oldCust_rfAUC <- oldCust_rfAUC@y.values[[1]]

print(paste('Area under the Curve for Random Forest Model (oldCust)',oldCust_rfAUC))
#Area under Curve = 80.23%



#########################   Tree Model (oldCust)   #########################


library(party)

oldCust_tree<-ctree(y ~.,data = oldCust_train)
plot(oldCust_tree)

oldCust_treeResult <- predict(oldCust_tree, oldCust_test)
oldCust_treeError  <- mean(oldCust_treeResult != oldCust_test$y)

print(paste('Accuracy for Tree Model (oldCust)',1-oldCust_treeError))
#Accuracy = 84.23%

library(ROCR)

oldCust_treePred <- prediction(as.numeric(oldCust_treeResult), as.numeric(oldCust_test$y))
oldCust_treePerf <- performance(oldCust_treePred, measure = "tpr", x.measure = "fpr")
plot(oldCust_treePerf)

oldCust_treeAUC <- performance(oldCust_treePred, measure = "auc")
oldCust_treeAUC <- oldCust_treeAUC@y.values[[1]]
print(paste('Area under the Curve for Tree Model (oldCust)',oldCust_treeAUC))
#Area under Curve = 77.04%





#########################                      #########################
            #####         New Customer DATASET          #####
#########################                      #########################


# Since they are the new customers, it does not make sense to know 
# their outcome from the previous campaign, 
# number of previous contacts, 
# amount of day passed from their last contact and 
# their default credit with the bank

newCust$poutcome <-NULL
newCust$previous <-NULL
newCust$pdays    <-NULL
newCust$default  <-NULL


# Missing value Frequencies
library(VIM) 

aggrPlot <- aggr(newCust, col=c('darkgreen','red'), ylab=c("Missing Value Histogram","Pattern"))
#education 0.0411 #housing 0.0239 #loan 0.0239 #job 0.0082 #marital 0.0017

#Subscription Count
newCount <- table(newCust$y)
barplot(newCount,col=c("red","darkgreen"),legend = rownames(newCount), main = "New_Subscriptions")
#no 32422 #yes 3141 


# Impute Missing Values and Check
library(mice)

newCust2 <- mice(newCust)
newCust_com <- complete(newCust2)

aggrPlot <- aggr(newCust_com, col=c('darkgreen','red'), ylab=c("Missing Value Histogram","Pattern"))
#none


#Split data into Train and Test subsets
library(caret)
set.seed(102)

newCust_com$y<-ifelse(newCust_com$y =='no', 0,1)
newCust_com$y<-as.factor(newCust_com$y)

id <- sample(seq(1, 2), size = nrow(newCust_com), replace = TRUE, prob = c(.7, .3))

newCust_train <- newCust_com[id==1,]
newCust_test  <- newCust_com[id==2,]

table(newCust_train$y) #no 22745 #yes 2191
table(newCust_test$y)  #no 9677  #yes 950

newCust_trains<- newCust_train 
# saving train data before making the model



#########################   Logistic Model (newCust)   #########################


newCust_logit <- glm(y ~., family=binomial(link='logit'), data = newCust_train)
summary(newCust_logit)

newCust_logitResult <- predict(newCust_logit, newdata=newCust_test, type='response')
newCust_logitResult <- ifelse(newCust_logitResult >= 0.5,1,0)
newCust_logitError  <- mean(newCust_logitResult != newCust_test$y)

print(paste('Accuracy for Logistic Model (newCust)',1-newCust_logitError))
#Accuracy = 92.01%

library(ROCR)

newCust_logitPred <- prediction(newCust_logitResult, newCust_test$y)
newCust_logitPerf <- performance(newCust_logitPred, measure = "tpr", x.measure = "fpr")
plot(newCust_logitPerf)

newCust_logitAUC <- performance(newCust_logitPred, measure = "auc")
newCust_logitAUC <- newCust_logitAUC@y.values[[1]]

print(paste('Area under the Curve for Logistic Model (newCust)',newCust_logitAUC))
#Area under Curve = 64.99%



#########################   Random Forest Model (newCust)   #########################


library(randomForest)

newCust_rf<-randomForest(y ~.,data = newCust_train, importance=TRUE, ntree=1000)

newCust_rfResult <- predict(newCust_rf, newCust_test)
newCust_rfError  <- mean(newCust_rfResult != newCust_test$y)

print(paste('Accuracy for Random Forest Model (newCust)',1-newCust_rfError))
#Accuracy = 91.85%

library(ROCR)

newCust_rfPred <- prediction(as.numeric(newCust_rfResult), as.numeric(newCust_test$y))
newCust_rfPerf <- performance(newCust_rfPred, measure = "tpr", x.measure = "fpr")
plot(newCust_rfPerf)

newCust_rfAUC <- performance(newCust_rfPred, measure = "auc")
newCust_rfAUC <- newCust_rfAUC@y.values[[1]]

print(paste('Area under the Curve for Random Forest Model (newCust)',newCust_rfAUC))
#Area under Curve = 70.04%



#########################   Tree Model (newCust)   #########################


library(party)

newCust_tree<-ctree(y ~.,data = newCust_train)
plot(newCust_tree)

newCust_treeResult <- predict(newCust_tree, newCust_test)
newCust_treeError  <- mean(newCust_treeResult != newCust_test$y)

print(paste('Accuracy for Tree Model (newCust)',1-newCust_treeError))
#Accuracy = 92.11%

library(ROCR)

newCust_treePred <- prediction(as.numeric(newCust_treeResult), as.numeric(newCust_test$y))
newCust_treePerf <- performance(newCust_treePred, measure = "tpr", x.measure = "fpr")
plot(newCust_treePerf)

newCust_treeAUC <- performance(newCust_treePred, measure = "auc")
newCust_treeAUC <- newCust_treeAUC@y.values[[1]]
print(paste('Area under the Curve for Tree Model (newCust)',newCust_treeAUC))
#Area under Curve = 76.54%

