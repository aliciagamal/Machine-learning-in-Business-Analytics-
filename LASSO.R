#class balancing for the lasso 
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
set.seed(123)
x = db[,-9]
y = db[,9]
ctrl = trainControl(method = "cv",
                    number = 10)
lambda_values <- 10^seq(10, -2, length.out = 100)
tune_grid <- expand.grid(alpha = 1, lambda = lambda_values)
lasso_model <- train(x, y, method = "glmnet", trControl = ctrl, tuneGrid = tune_grid)
lasso_summary <- summary(lasso_model)
best_model <- lasso_model$finalModel
coef <- coef(best_model, s = lasso_model$bestTune$lambda)
print(coef)

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

#balancing 
set.seed(123)
db_upsampled = upSample(x,y,yname="Outcome")
table(db_upsampled$Outcome)
#806 and 806 
head(db_upsampled)
lasso_model <- train(db_upsampled[,-9], db_upsampled[,9], method = "glmnet", trControl = ctrl, tuneGrid = tune_grid)
lasso_summary <- summary(lasso_model)
best_model <- lasso_model$finalModel
coef <- coef(best_model, s = lasso_model$bestTune$lambda)
print(coef)
#this actually puts things to 0 
prob_test_visit_lasso_train = predict(lasso_model, newdata = db_upsampled , type= "prob")
pred_test_visit_lasso_train = ifelse(prob_test_visit_lasso_train$'1' >= 0.5,1,0)
cm_lasso_train = table(Pred=pred_test_visit_lasso_train, Obs = db_upsampled$Outcome)
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
