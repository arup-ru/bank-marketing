#########################   "Analysis of a Portugese Bank Marketing Dataset"   #########################


# Installing Libraries ~~~~~~~~~~~~~~~~~~~~

pkgs <- c("VIM", "corrgram", "mice", "DMwR", "ROCR", "descr", "caret", "randomForest", "party", "e1071", "graphics", "factoextra", "NbClust", "dbscan", "mclust", "arules")
install.packages(pkgs)



# Loading Data ~~~~~~~~~~~~~~~~~~~~

allCust <- as.data.frame(read.csv("C:/Users/arup.roy/Documents/bank-marketing-master/bank-additional-full.csv", sep= ";",header = T))

str(allCust) #Numeric=10 #Categorical=11 
nrow(allCust) #41188

head(allCust,10)
plot(allCust)



# Functions for Predictive Models ~~~~~~~~~~~~~~~~~~~~

modelResult<-function(model_train, model_test)
{
  
  library(caret) # Logistic Regression
  
  model_logit       <- glm(y ~., family=binomial(link='logit'), data = model_train)
  model_logitResult <- predict(model_logit, newdata=model_test, type='response')
  model_logitResult <- ifelse(model_logitResult >= 0.5,1,0)
  model_logitError  <- mean(model_logitResult != model_test$y)
  
  model_logitPred <- prediction(model_logitResult, model_test$y)
  model_logitPerf <- performance(model_logitPred, measure = "tpr", x.measure = "fpr")
  model_logitAUC  <- performance(model_logitPred, measure = "auc")
  model_logitAUC  <- model_logitAUC@y.values[[1]]
  
  
  library(randomForest) # Random Forest
  
  model_rf       <-randomForest(y ~.,data = model_train, importance=TRUE, ntree=1000)
  model_rfResult <- predict(model_rf, model_test)
  model_rfError  <- mean(model_rfResult != model_test$y)
  
  model_rfPred <- prediction(as.numeric(model_rfResult), as.numeric(model_test$y))
  model_rfPerf <- performance(model_rfPred, measure = "tpr", x.measure = "fpr")
  model_rfAUC  <- performance(model_rfPred, measure = "auc")
  model_rfAUC  <- model_rfAUC@y.values[[1]]
  
  
  library(party) # Decision Tree 
  
  model_tree       <-ctree(y ~.,data = model_train)
  model_treeResult <- predict(model_tree, model_test)
  model_treeError  <- mean(model_treeResult != model_test$y)
  
  model_treePred <- prediction(as.numeric(model_treeResult), as.numeric(model_test$y))
  model_treePerf <- performance(model_treePred, measure = "tpr", x.measure = "fpr")
  model_treeAUC  <- performance(model_treePred, measure = "auc")
  model_treeAUC  <- model_treeAUC@y.values[[1]]
  
  
  library(e1071) # Naive Bayes
  
  model_nb       <-naiveBayes(y ~.,data = model_train)
  model_nbResult <- predict(model_nb, model_test)
  model_nbError  <- mean(model_nbResult != model_test$y)
  
  model_nbPred <- prediction(as.numeric(model_nbResult), as.numeric(model_test$y))
  model_nbPerf <- performance(model_nbPred, measure = "tpr", x.measure = "fpr")
  model_nbAUC  <- performance(model_nbPred, measure = "auc")
  model_nbAUC  <- model_nbAUC@y.values[[1]]

  
  # Measuring other Metrics from Confusion Matrix ~~~~~~~~~~~~~~~~~~~~
  
  library(descr)
  
  cm_logit     <- as.matrix(CrossTable(x=model_logitResult, y=model_test$y, prop.chisq=FALSE))
  n_logit      <- sum(cm_logit)   # number of instances
  diag_logit   <- diag(cm_logit)  # number of correctly classified instances per class 
  rowsum_logit <- apply(cm_logit, 1, sum) # number of instances per class
  colsum_logit <- apply(cm_logit, 2, sum) # number of predictions per class
  
  accuracy_logit  <- sum(diag_logit) / n_logit
  precision_logit <- diag_logit / colsum_logit 
  recall_logit    <- diag_logit / rowsum_logit 
  f1_logit        <- 2 * precision_logit * recall_logit / (precision_logit + recall_logit)
  
  print(paste('Accuracy, Precision, Recall & F1-score of Logistic Regression Model:', accuracy_logit, precision_logit, recall_logit, f1_logit))
  
  
  cm_rf     <-as.matrix(CrossTable(x=model_rfResult,    y=model_test$y, prop.chisq=FALSE))
  n_rf      <- sum(cm_rf)   # number of instances
  diag_rf   <- diag(cm_rf)  # number of correctly classified instances per class 
  rowsum_rf <- apply(cm_rf, 1, sum) # number of instances per class
  colsum_rf <- apply(cm_rf, 2, sum) # number of predictions per class
  
  accuracy_rf  <- sum(diag_rf) / n_rf
  precision_rf <- diag_rf / colsum_rf 
  recall_rf    <- diag_rf / rowsum_rf 
  f1_rf        <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
  
  print(paste('Accuracy, Precision, Recall & F1-score of Random Forest Model:', accuracy_rf, precision_rf, recall_rf, f1_rf))
  
  
  cm_tree     <-as.matrix(CrossTable(x=model_treeResult,  y=model_test$y, prop.chisq=FALSE))
  n_tree      <- sum(cm_tree)   # number of instances
  diag_tree   <- diag(cm_tree)  # number of correctly classified instances per class 
  rowsum_tree <- apply(cm_tree, 1, sum) # number of instances per class
  colsum_tree <- apply(cm_tree, 2, sum) # number of predictions per class
  
  accuracy_tree  <- sum(diag_tree) / n_tree
  precision_tree <- diag_tree / colsum_tree 
  recall_tree    <- diag_tree / rowsum_tree 
  f1_tree        <- 2 * precision_tree * recall_tree / (precision_tree + recall_tree)
  
  print(paste('Accuracy, Precision, Recall & F1-score of binary Tree Model:', accuracy_tree, precision_tree, recall_tree, f1_tree))
  
  
  cm_nb     <-as.matrix(CrossTable(x=model_nbResult,    y=model_test$y, prop.chisq=FALSE))
  n_nb      <- sum(cm_nb)   # number of instances
  diag_nb   <- diag(cm_nb)  # number of correctly classified instances per class 
  rowsum_nb <- apply(cm_nb, 1, sum) # number of instances per class
  colsum_nb <- apply(cm_nb, 2, sum) # number of predictions per class
  
  accuracy_nb  <- sum(diag_nb) / n_nb
  precision_nb <- diag_nb / colsum_nb 
  recall_nb    <- diag_nb / rowsum_nb 
  f1_nb        <- 2 * precision_nb * recall_nb / (precision_nb + recall_nb)
  
  print(paste('Accuracy, Precision, Recall & F1-score of Naive Bayes Model:', accuracy_nb, precision_nb, recall_nb, f1_nb))
  
  
  # Measuring the AREA UNDER THE CURVE from ROC GRAPHS ~~~~~~~~~~~~~~~~~~~~
  
  library(ROCR)
  
  plot(model_logitPerf           , lwd=2, col = "blue")
  plot(model_rfPerf,   add = TRUE, lwd=2, col = "green")
  plot(model_treePerf, add = TRUE, lwd=2, col = "orange")
  plot(model_nbPerf,   add = TRUE, lwd=2, col = "magenta")
  
  ptr_logit <-paste('Logistic Regression (AUC =',          round (100*model_logitAUC,2), "%)")
  ptr_rf    <-paste('Random Forest         (AUC =',        round (100*model_rfAUC,2),    "%)")
  ptr_tree  <-paste('Binary Tree                 (AUC =',  round (100*model_treeAUC,2),  "%)")
  ptr_nb    <-paste('Naive Bayes               (AUC =',    round (100*model_nbAUC,2),    "%)")
  
  legend("bottomright", legend=c(ptr_logit, ptr_rf, ptr_tree, ptr_nb), col=c("blue", "green", "orange", "magenta"), lwd=2, cex = .75)
  
}



# Performance of Models from Raw Data ~~~~~~~~~~~~~~~~~~~~

set.seed(100)
allCust0 <- allCust

corrgram(allCust0[c(1,11:14,16:21)], order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt, main="Numeric Data")

allCust0$y<-ifelse(allCust0$y =='no', 0,1)
allCust0$y<-as.factor(allCust0$y)

id0 <- sample(seq(1, 2), size = nrow(allCust0), replace = TRUE, prob = c(.7, .3))

allCust0_train <- allCust0[id0==1,];   table(allCust0_train$y) #no 25714 #yes 3272 
allCust0_test  <- allCust0[id0==2,];   table(allCust0_test$y)  #no 10834 #yes 1368

modelResult(allCust0_train, allCust0_test)



# Removal of Unknown Data ~~~~~~~~~~~~~~~~~~~~

allCust1 <- allCust
allCust1[allCust1=="unknown"]<-NA

# Counting Missing values
sapply(allCust1, function(x) sum(is.na(x)))

# Impute Missing Values and Check
library(mice)

allCust11 <- mice(allCust1)
allCust_prep <- complete(allCust11)

aggrPlot <- aggr(allCust_prep, col=c('darkgreen','red'), ylab=c("Missing Value Histogram","Pattern"))
# none


# Classifying (/Factorizing) Categorical Data ~~~~~~~~~~~~~~~~~~~~

levels(allCust_prep$age)<- c("1", "2", "3", "4", "5", "6", "7")

levels(allCust_prep$marital)<-c("3-Divorced", "2-Married", "1-Single")
levels(allCust_prep$education)<-c("1-Primary", "1-Primary", "2-Secondary", "2-Secondary", "0-NoEducation", "3-PostSecondary", "3-PostSecondary")
levels(allCust_prep$default)<-c("0", "1")
levels(allCust_prep$housing)<-c("0", "1")
levels(allCust_prep$loan)<-c("0", "1")

levels(allCust_prep$month)<- c("4-Apr", "8-Aug", "12-Dec", "7-Jul", "6-Jun", "3-Mar", "5-May", "11-Nov", "10-Oct", "9-Sep")
levels(allCust_prep$day_of_week) <- c("5-Fri", "1-Mon", "4-Thu", "2-Tue", "3-Wed")

levels(allCust_prep$poutcome)<-c("0-Fail", "-1-NonExistent", "1-Pass")


allCust_prep$y<-ifelse(allCust_prep$y =='no', 0,1)
allCust_prep$y<-as.factor(allCust_prep$y)


allCust1     <- data.frame(as.numeric(as.factor(allCust_prep$age)),
                           as.numeric(as.factor(allCust_prep$job)),
                           as.numeric(as.factor(allCust_prep$marital)),
                           as.numeric(as.factor(allCust_prep$education)),
                           as.numeric(as.factor(allCust_prep$default)),
                           as.numeric(as.factor(allCust_prep$housing)),
                           as.numeric(as.factor(allCust_prep$loan)),
                           as.numeric(as.factor(allCust_prep$contact)),
                           as.numeric(as.factor(allCust_prep$month)),
                           as.numeric(as.factor(allCust_prep$day_of_week)),
                           allCust_prep$duration, allCust_prep$campaign, allCust_prep$pdays, allCust_prep$previous,
                           as.numeric(as.factor(allCust_prep$poutcome)),
                           allCust_prep$emp.var.rate, allCust_prep$cons.price.idx, allCust_prep$cons.conf.idx, allCust_prep$euribor3m, allCust_prep$nr.employed,
                           as.numeric(as.factor(allCust_prep$y)))

colnames(allCust1) <- c("age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous","poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m","nr.employed","y")



# Performance of Models from Processed Data ~~~~~~~~~~~~~~~~~~~~

set.seed(101)
corrgram(allCust1, order=TRUE, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, main="Numeric Data")

id1 <- sample(seq(1, 2), size = nrow(allCust1), replace = TRUE, prob = c(.7, .3))

allCust1_train <- allCust1[id1==1,];   table(allCust1_train$y) #no 25714 #yes 3272 
allCust1_test  <- allCust1[id1==2,];   table(allCust1_test$y)  #no 10834 #yes 1368

modelResult(allCust1_train, allCust1_test)



# Normalizing (/Balancing) Imbalanced Data ~~~~~~~~~~~~~~~~~~~~

library(DMwR)

allCust2 <-SMOTE(y~.,allCust1,perc.over = 100, perc.under=200)
prop.table(table(allCust2$y)) #no:yes = 50:50



# Performance of Models from Normalized Data ~~~~~~~~~~~~~~~~~~~~

set.seed(102)
corrgram(allCust2, order=TRUE, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, main="Numeric Data")

id2 <- sample(seq(1, 2), size = nrow(allCust2), replace = TRUE, prob = c(.7, .3))

allCust2_train <- allCust1[id2==1,];   table(allCust1_train$y)  
allCust2_test  <- allCust1[id2==2,];   table(allCust1_test$y)  

modelResult(allCust2_train, allCust2_test)



################################################

data(iris)
iris_x <-oldBank[,1:20]
iris.pca.rawdata <- prcomp(iris_x, scale = FALSE, center= FALSE)
iris.pca.rawdata

iris.pca.rawdata$rotation      # eigen vector / rotation matrix / tranformation matrix
head(iris.pca.rawdata$x)       # Transformed data
#or
head(as.matrix(iris_x)%*%iris.pca.rawdata$rotation)

plot(iris.pca.rawdata, type = "l", main='without data normalization')


iris.pca.normdata <- prcomp(iris_x, scale = TRUE, center= TRUE)
iris.pca.normdata$rotation      # eigen vector / rotation matrix / tranformation matrix
head(iris.pca.normdata$x)       # Transformed data
#or
head(as.matrix(scale(iris_x)) %*% iris.pca.normdata$rotation)
plot(iris.pca.normdata, type = "l", main='with data normalization')


biplot(iris.pca.rawdata, choices = 1:2, main='Raw Data')
biplot(iris.pca.normdata, choices = 1:2, main='Norm Data')

cor(iris.pca.normdata$x)


################################################


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }


prc_train <- prc_n[id==1,]
prc_test  <- prc_n[id==2,]

prc_train_labels <- prc[id==1, 21]
prc_test_labels  <- prc[id==2, 21]

library(class)
prc_test_pred <- knn(train = prc_train, test = prc_test,cl = prc_train_labels, k=10)

library(descr)
CrossTable(x=prc_test_labels, y=prc_test_pred, prop.chisq=FALSE)


################################################


rn_train <- sample(nrow(oldBank), floor(nrow(oldBank)*0.7))
train    <- oldBank[rn_train,]
test     <- oldBank[-rn_train,]


model_mlr <- lm(price~ram+speed+screen+hd+ads, data=train) 
prediction <- predict(model_mlr, interval="prediction", newdata =test)

errors <- prediction[,"fit"] - test$price
hist(errors)



full <- lm(y~age+job+marital+education+default+housing+loan+contact+month+day_of_week+duration+campaign+pdays+previous+poutcome+emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m+nr.employed,data=oldBank)
stepB <- stepAIC(full, direction= "backward", trace=TRUE)
summary(stepB)

subsets<-regsubsets(price~ram+hd+speed+screen+ads+trend,data=c_prices, nbest=1,)
sub.sum <- summary(subsets)
as.data.frame(sub.sum$outmat)


#install.packages('FNN')
library(FNN)
dataset <- rbind(c_prices, c(7000,0,32,90,8,15,'no','no','yes',200,2))  

dataset.numeric <- sapply( dataset[,2:11], as.numeric )
#Should convert data to numeric to use knn.reg
dataset.numeric <- as.data.frame(dataset.numeric)
prediction <- knn.reg(dataset.numeric[1:nrow(c_prices),-1], 
                      test = dataset.numeric[nrow(c_prices)+1,-1],
                      dataset.numeric[1:nrow(c_prices),]$price, k = 7 , algorithm="kd_tree")  
prediction$pred

