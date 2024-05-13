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

#SIMPLE LOGISTIC REGRESSION
logistic_model <- train(x, y, method = "glm", trControl = ctrl)
logistic_summary <- summary(logistic_model)
logistic_summary #AIC: 1237.5
library(xtable)
print(xtable(logistic_summary, type ="latex"))
#pred on training 
prob_test_visit_logreg_train = predict(logistic_model, newdata = db , type= "prob")
pred_test_visit_logreg_train = ifelse(prob_test_visit_logreg_train$'1' >= 0.5,1,0)
cm_logreg_train = table(Pred=pred_test_visit_logreg_train, Obs = db$Outcome)
cm_logreg_train
metrics_logreg_train = confusionMatrix(cm_logreg_train)
metrics_logreg_train
#pred on testing
prob_test_visit_logreg = predict(logistic_model, newdata = test_set , type= "prob")
pred_test_visit_logreg = ifelse(prob_test_visit_logreg$'1' >= 0.5,1,0)
cm_logreg_test = table(Pred=pred_test_visit_logreg, Obs = test_set$Outcome)
cm_logreg_test
metrics_logreg_test = confusionMatrix(cm_logreg_test)
metrics_logreg_test

#logistic regression with stepwise selection
stepwise_model <- train(x, y, method = "glmStepAIC", trControl = ctrl)
stepwise_summary <- summary(stepwise_model)
stepwise_summary #selects pregnancies, glucose, insulin, BMI, diabetes pedigree, age
#AIC: 1234.1
print(xtable(stepwise_summary, type ="latex"))
#prediction on training of stepwise selection
prob_test_visit_stepwise_train = predict(stepwise_model, newdata = db , type= "prob")
pred_test_visit_stepwise_train = ifelse(prob_test_visit_stepwise_train$'1' >= 0.5,1,0)
cm_stepwise_train = table(Pred=pred_test_visit_stepwise_train, Obs = db$Outcome)
cm_stepwise_train
metrics_stepwise_train = confusionMatrix(cm_stepwise_train)
metrics_stepwise_train
#prediction on testing of stepwise selection
prob_test_visit_stepwise= predict(stepwise_model, newdata = test_set , type= "prob")
pred_test_visit_stepwise= ifelse(prob_test_visit_stepwise$'1' >= 0.5,1,0)
cm_stepwise_test = table(Pred=pred_test_visit_stepwise, Obs = test_set$Outcome)
cm_stepwise_test
metrics_stepwise_test = confusionMatrix(cm_stepwise_test)
metrics_stepwise_test

#logistic regression with lasso
lambda_values <- 10^seq(10, -2, length.out = 100)
tune_grid <- expand.grid(alpha = 1, lambda = lambda_values)
lasso_model <- train(x, y, method = "glmnet", trControl = ctrl, tuneGrid = tune_grid)
lasso_summary <- summary(lasso_model)
lasso_summary
best_model <- lasso_model$finalModel
coef <- coef(best_model, s = lasso_model$bestTune$lambda)
print(coef)
#this is the important line to see what was selected
#the lasso model shrinks the coefficients, but it does not put 
#anything to 0, it does not select anything 

#lasso new
lambda_values_new <- 10^seq(-3, -0.5, length.out = 100)
tune_grid_new <- expand.grid(alpha = 1, lambda = lambda_values_new)
lasso_model_new <- train(x, y, method = "glmnet", trControl = ctrl, tuneGrid = tune_grid_new)
lasso_summary_new <- summary(lasso_model_new)
lasso_summary_new
best_model_new <- lasso_model_new$finalModel
coef_new <- coef(best_model_new, s = lasso_model_new$bestTune$lambda)
print(coef_new)

#logistic regression with lasso: metrics on training set
prob_test_visit_lasso_train = predict(lasso_model, newdata = db , type= "prob")
pred_test_visit_lasso_train = ifelse(prob_test_visit_lasso_train$'1' >= 0.5,1,0)
cm_lasso_train = table(Pred=pred_test_visit_lasso_train, Obs = db$Outcome)
cm_lasso_train
metrics_lasso_train = confusionMatrix(cm_lasso_train)
metrics_lasso_train
#logistic regression with lasso: metrics on testing set
prob_test_visit_lasso = predict(lasso_model, newdata = test_set , type= "prob")
pred_test_visit_lasso = ifelse(prob_test_visit_lasso$'1' >= 0.5,1,0)
cm_lasso_test = table(Pred=pred_test_visit_lasso, Obs = test_set$Outcome)
cm_lasso_test
metrics_lasso_test = confusionMatrix(cm_lasso_test)
metrics_lasso_test

#PARAMETER TUNING: TUNE LAMBDA
#the only real interesting thing about this for the report are the values of lambda
library(pROC)

#LOGREG
roc_obj_logreg = roc(db$Outcome, prob_test_visit_logreg_train$'1')
X11()
roc_logreg <- plot.roc(roc_obj_logreg, main="ROC Curve for Logistic Regression")
coords_obj_logreg <- coords(roc_obj_logreg, "best")
lambda_logreg <- coords_obj_logreg["threshold"]
lambda_logreg #0.2979084
#refit 
prob_test_visit_logreg_train = predict(logistic_model, newdata = db , type= "prob")
pred_test_visit_logreg_train = ifelse(prob_test_visit_logreg_train$'1' >= 0.2979084,1,0)
cm_logreg_train = table(Pred=pred_test_visit_logreg_train, Obs = db$Outcome)
cm_logreg_train
metrics_logreg_train = confusionMatrix(cm_logreg_train)
metrics_logreg_train
#try on test set
prob_test_visit_logreg = predict(logistic_model, newdata = test_set , type= "prob")
pred_test_visit_logreg = ifelse(prob_test_visit_logreg$'1' >= 0.2979084,1,0)
cm_logreg_test = table(Pred=pred_test_visit_logreg, Obs = test_set$Outcome)
cm_logreg_test
metrics_logreg_test = confusionMatrix(cm_logreg_test)
metrics_logreg_test


#stepwise
roc_obj_stepwise <- roc(db$Outcome, prob_test_visit_stepwise_train$'1')
roc_stepwise <- plot.roc(roc_obj_stepwise, main="ROC Curve for Stepwise Selection")
coords_obj_stepwise <- coords(roc_obj_stepwise, "best")
lambda_step <- coords_obj_stepwise["threshold"]
lambda_step#0.3006133
#training
prob_test_visit_stepwise_train = predict(stepwise_model, newdata = db , type= "prob")
pred_test_visit_stepwise_train = ifelse(prob_test_visit_stepwise_train$'1' >= 0.3006133,1,0)
cm_stepwise_train = table(Pred=pred_test_visit_stepwise_train, Obs = db$Outcome)
cm_stepwise_train
metrics_stepwise_train = confusionMatrix(cm_stepwise_train)
metrics_stepwise_train
#testing
prob_test_visit_stepwise = predict(stepwise_model, newdata = test_set , type= "prob")
pred_test_visit_stepwise = ifelse(prob_test_visit_stepwise$'1' >= 0.3006133,1,0)
cm_stepwise_test = table(Pred=pred_test_visit_stepwise, Obs = test_set$Outcome)
cm_stepwise_test
metrics_stepwise_test = confusionMatrix(cm_stepwise_test)
metrics_stepwise_test

#lasso
roc_obj_lasso <- roc(db$Outcome, prob_test_visit_lasso_train$'1')
roc_lasso <- plot.roc(roc_obj_lasso, main="ROC Curve for Lasso")
coords_obj_lasso <- coords(roc_obj_lasso, "best")
lambda_lasso <- coords_obj_lasso["threshold"]
lambda_lasso # 0.3006824

#logistic regression with lasso: metrics on training set
prob_test_visit_lasso_train = predict(lasso_model, newdata = db , type= "prob")
pred_test_visit_lasso_train = ifelse(prob_test_visit_lasso_train$'1' >= 0.3006824,1,0)
cm_lasso_train = table(Pred=pred_test_visit_lasso_train, Obs = db$Outcome)
cm_lasso_train
metrics_lasso_train = confusionMatrix(cm_lasso_train)
metrics_lasso_train
#test
prob_test_visit_lasso = predict(lasso_model, newdata = test_set , type= "prob")
pred_test_visit_lasso = ifelse(prob_test_visit_lasso$'1' >= 0.3006824,1,0)
cm_lasso_test = table(Pred=pred_test_visit_lasso, Obs = test_set$Outcome)
cm_lasso_test
metrics_lasso_train = confusionMatrix(cm_lasso_test)
metrics_lasso_train

#class balancing on lasso 



