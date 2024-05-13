rm(list = ls())
graphics.off()
db = read.csv("Training.csv")
test_set = read.csv("Testing.csv")
test_set <- test_set[!(test_set[, 3] == 0 | test_set[, 4] == 0 | test_set[, 5] == 0 | test_set[, 6] == 0), ]
test_set$Outcome = as.factor(test_set$Outcome)
db <- db[!(db[, 3] == 0 | db[, 4] == 0 | db[, 5] == 0 | db[, 6] == 0), ]
db$Outcome = as.factor(db$Outcome)
library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(e1071)
library(ROCR)

#10-fold cross validation
set.seed(123)
x = db[,-9]
y = db[,9]
ctrl = trainControl(method = "cv",
                    number = 10)
library(kernlab)
set.seed(123)
#support vector machine: BEFORE HYPERPARAMETER TUNING
svm_model_linear <- train(x, y, method = "svmLinear", trControl = ctrl)
svm_model_linear #accuracy: 0.7690391, kappa = 0.4774859
svm_model_linear$results
#default C = 1
svm_model_radial <- train(x,y, method = "svmRadial", trControl = ctrl)
svm_model_radial
svm_model_radial$results
#try linear and radial
#sigma = 0.1262944 (constant) and C = 1 (selected through accuracy???)
# C     Accuracy   Kappa    
# 0.25  0.8068324  0.5608509
# 0.50  0.8343058  0.6208634
# 1.00  0.8720214  0.7115465

#HYPERPARAMETER TUNING FOR SUPPORT VECTOR MACHINE: I NEED TO CHECK IF IT IS CORRECT
#tuning the linear model
set.seed(123)
grid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100, 1000))
str(db)
head(db)
ctrl = trainControl(method = "cv",
                    number = 10)
# tr_data = data.frame(x,y)
# svm_linear_tuned <- train(y~.,data = tr_data, method = "svmLinear",
#                          trControl= ctrl,
#                          tuneGrid = grid)
svm_linear_tuned <- train(x,y, method = "svmLinear",
                          trControl= ctrl,
                          tuneGrid = grid)
svm_linear_tuned #C=10
class(svm_linear_tuned)
svm_linear_tuned$results
svm_linear_tuned$finalModel

x11()
plot(svm_linear_tuned)
svm_linear_tuned$bestTune

linsvmtuned_try <- predict(svm_linear_tuned, newdata = db[,-9]) #error: kernlab class prediction calculations failed; returning NAs
cm_svm_train <- table(Pred = linsvmtuned_try, Obs = db$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train
#on test set
linsvmtuned_try <- predict(svm_linear_tuned, newdata = test_set[,-9]) 
cm_svm_train <- table(Pred = linsvmtuned_try, Obs = test_set$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train


#variable importance plot
importance = varImp(svm_linear_tuned)
print(importance)
x11()
plot(importance)








#tuning the radial model 
grid_radial <- expand.grid(sigma = c(0.01, 0.02, 0.05, 0.1),
                           C = c(1, 10, 100, 500, 1000))
set.seed(123)
svm_radial_tuned <- train(x,y, method = "svmRadial",
                          trControl=ctrl,
                          tuneGrid = grid_radial)
svm_radial_tuned
svm_radial_tuned$results
x11()
plot(svm_radial_tuned)
svm_radial_tuned$bestTune #sigma 0.1 and C=500
svm_radial_tuned$finalModel

#prediction on the training set
radial_svm_tuned_pred_train <- predict(svm_radial_tuned, newdata = db[,-9]) 
cm_svm_train <- table(Pred = radial_svm_tuned_pred_train, Obs = db$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train
#On test
radial_svm_tuned_pred_train <- predict(svm_radial_tuned, newdata = test_set[,-9]) 
cm_svm_train <- table(Pred = radial_svm_tuned_pred_train, Obs = test_set$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train
#overfitting!!! i try to reduce C parameter
grid_radial_ovft <- expand.grid(sigma = c(0.01, 0.02, 0.05, 0.1),
                           C = 1)
svm_radial_tuned_ovft <- train(x,y, method = "svmRadial",
                          trControl=ctrl,
                          tuneGrid = grid_radial_ovft)
svm_radial_tuned_ovft
svm_radial_tuned_ovft$bestTune
#prediction on the training set
radial_ovft <- predict(svm_radial_tuned_ovft, newdata = db[,-9]) 
cm_svm_train <- table(Pred = radial_ovft, Obs = db$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train #0.8861
#On test
radial_svm_tuned_pred_test <- predict(svm_radial_tuned_ovft, newdata = test_set[,-9]) 
cm_svm_test <- table(Pred = radial_svm_tuned_pred_test, Obs = test_set$Outcome)
cm_svm_test = as.table(cm_svm_test)
cm_svm_test
metrics_svm_test = confusionMatrix(cm_svm_test)
metrics_svm_test #0.7949

svm_radial_tuned_ovft

#let's try again
#overfitting!!! i try to reduce C parameter
set.seed(123)
grid_radial_ovft <- expand.grid(sigma = c(0.01, 0.02, 0.05, 0.1),
                                C = 0.5)
svm_radial_tuned_ovft <- train(x,y, method = "svmRadial",
                               trControl=ctrl,
                               tuneGrid = grid_radial_ovft)
svm_radial_tuned_ovft
svm_radial_tuned_ovft$bestTune #sigma = 0.1, C = 0.5
svm_radial_tuned_ovft$finalModel #637
#prediction on the training set
radial_ovft <- predict(svm_radial_tuned_ovft, newdata = db[,-9]) 
cm_svm_train <- table(Pred = radial_ovft, Obs = db$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train 
#On test
radial_svm_tuned_pred_test <- predict(svm_radial_tuned_ovft, newdata = test_set[,-9]) 
cm_svm_test <- table(Pred = radial_svm_tuned_pred_test, Obs = test_set$Outcome)
cm_svm_test = as.table(cm_svm_test)
cm_svm_test
metrics_svm_test = confusionMatrix(cm_svm_test)
metrics_svm_test 



importance = varImp(svm_radial_tuned_ovft)
print(importance)
x11()
plot(importance)

x11()
layout(matrix(c(1, 2), nrow = 1))
# Plot 1
plot(svm_radial_tuned_ovft, main = "tuning of sigma in the radial model")
# Plot 2
plot(svm_linear_tuned, main = "tuning of C in the linear model")




#DECISION BOUNDARY PLOT 

