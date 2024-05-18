rm(list = ls())
graphics.off()
db = read.csv("Training.csv")
test_set = read.csv("Testing.csv")
test_set <- test_set[!(test_set[, 3] == 0 | test_set[, 4] == 0 | test_set[, 5] == 0 | test_set[, 6] == 0), ]
test_set$Outcome = as.factor(test_set$Outcome)
db <- db[!(db[, 3] == 0 | db[, 4] == 0 | db[, 5] == 0 | db[, 6] == 0), ]
db$Outcome = as.factor(db$Outcome)
dim(test_set)
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

tuneGrid = expand.grid(.cp = seq(0.001, 0.5, 0.001)) #grid of cp values
tree_model <- train(x, y, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid)
tree_summary <- summary(tree_model)
print(tree_model$bestTune) #cp: 0.002
library(rpart.plot)

x11()
rpart.plot(tree_model$finalModel, extra = 1)
#classification tree: metrics on training
prob_train_tree = predict(tree_model, newdata = db, type = "prob")
pred_train_tree = ifelse(prob_train_tree$'1'>=0.5,1,0)
cm_tree_train = table(Pred = pred_train_tree, Obs = db$Outcome)
print(cm_tree_train)
metrics_tree_train = confusionMatrix(cm_tree_train)
print(metrics_tree_train)
#classification tree: metrics on testing
prob_test_tree = predict(tree_model, newdata = test_set, type = "prob")
pred_test_tree = ifelse(prob_test_tree$'1'>=0.5,1,0)
cm_tree_test = table(Pred = pred_test_tree, Obs = test_set$Outcome)
print(cm_tree_test)
metrics_tree_test = confusionMatrix(cm_tree_test)
print(metrics_tree_test)

# Prune the tree (by doing a manual tunegrid2) !!!! here !!! BAD ONLY PREDICTS 0
tuneGrid_2 = expand.grid(.cp = seq(0.5, 1, 0.01)) #grid of cp values
tree_model_2 <- train(x, y, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid_2)
tree_model_2$bestTune
x11()
rpart.plot(tree_model_2$finalModel, extra = 1)
prob_train_tree2 = predict(tree_model_2, newdata = db, type = "prob")
pred_train_tree_2 = ifelse(prob_train_tree2$'1'>=0.5,1,0)
cm_tree_train2 = table(Pred = factor(pred_train_tree_2, levels = c("0","1")), Obs = db$Outcome)
print(cm_tree_train2)
metrics_tree_train2 = confusionMatrix(cm_tree_train2)
print(metrics_tree_train2)
#classification tree: metrics on testing
prob_test_tree2 = predict(tree_model_2, newdata = test_set, type = "prob")
pred_test_tree2 = ifelse(prob_test_tree2$'1'>=0.5,1,0)
cm_tree_test2 = table(Pred = pred_test_tree2, Obs = test_set$Outcome)
print(cm_tree_test2)
metrics_tree_test2 = confusionMatrix(cm_tree_test2)
print(metrics_tree_test2)

#THIS IS THE BEST ONE 
#yet another grid - better, but now a bit of underfitting (better test than train)
tuneGrid_2 = expand.grid(.cp = seq(0.1, 0.5, 0.1)) #grid of cp values
tree_model_2 <- train(x, y, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid_2)
tree_model_2$bestTune
prob_train_tree2 = predict(tree_model_2, newdata = db, type = "prob")
pred_train_tree_2 = ifelse(prob_train_tree2$'1'>=0.5,1,0)
cm_tree_train2 = table(Pred = factor(pred_train_tree_2, levels = c("0","1")), Obs = db$Outcome)
print(cm_tree_train2)
metrics_tree_train2 = confusionMatrix(cm_tree_train2)
metrics_tree_train2
x11()
rpart.plot(tree_model_2$finalModel, extra = 1)
#test
prob_test_tree2 = predict(tree_model_2, newdata = test_set, type = "prob")
pred_test_tree2 = ifelse(prob_test_tree2$'1'>=0.5,1,0)
cm_tree_test2 = table(Pred = pred_test_tree2, Obs = test_set$Outcome)
print(cm_tree_test2)
metrics_tree_test2 = confusionMatrix(cm_tree_test2)
print(metrics_tree_test2)

#that was too simple - intermediate
tuneGrid_2 = expand.grid(.cp = seq(0.03, 0.5, 0.1)) #grid of cp values
tree_model_2 <- train(x, y, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid_2)
tree_model_2$bestTune #0.03
prob_train_tree2 = predict(tree_model_2, newdata = db, type = "prob")
pred_train_tree_2 = ifelse(prob_train_tree2$'1'>=0.5,1,0)
cm_tree_train2 = table(Pred = factor(pred_train_tree_2, levels = c("0","1")), Obs = db$Outcome)
print(cm_tree_train2)
metrics_tree_train2 = confusionMatrix(cm_tree_train2)
metrics_tree_train2
x11()
rpart.plot(tree_model_2$finalModel, extra = 1)
#test
prob_test_tree2 = predict(tree_model_2, newdata = test_set, type = "prob")
pred_test_tree2 = ifelse(prob_test_tree2$'1'>=0.5,1,0)
cm_tree_test2 = table(Pred = pred_test_tree2, Obs = test_set$Outcome)
print(cm_tree_test2)
metrics_tree_test2 = confusionMatrix(cm_tree_test2)
print(metrics_tree_test2)





#classification tree with randomforest
rf_model <- train(x, y, method = "rf", trControl = ctrl)
rf_summary <- summary(rf_model)
rf_summary
# x11()
# varImpPlot(rf_model$finalModel)
#predictions and metrics
# on training set
prob_test_rf_train = predict(rf_model, newdata = db, type = "prob")
pred_test_rf_train = ifelse(prob_test_rf_train$'1' >= 0.5, 1, 0)
cm_rf_train = table(Pred = pred_test_rf_train, Obs = db$Outcome)
print(cm_rf_train)
metrics_rf_train = confusionMatrix(cm_rf_train)
print(metrics_rf_train)
#on test set -- overfitting
prob_test_rf_train = predict(rf_model, newdata = test_set, type = "prob")
pred_test_rf_train = ifelse(prob_test_rf_train$'1' >= 0.5, 1, 0)
cm_rf_train = table(Pred = pred_test_rf_train, Obs = test_set$Outcome)
print(cm_rf_train)
metrics_rf_train = confusionMatrix(cm_rf_train)
print(metrics_rf_train)

#now we want to choose m (mtry) - we tune it
tuneGrid_mtry <- expand.grid(.mtry=seq(1, ncol(x), by = 2))
rf_model_mtry_tuned <- train(x, y, method = "rf", trControl = ctrl, tuneGrid = tuneGrid_mtry)
summary(rf_model_mtry_tuned)
rf_model_mtry_tuned$bestTune
#i sadly get mtry = 1

prob_mtry_rf_train = predict(rf_model_mtry_tuned, newdata = db, type = "prob")
pred_mtry_rf_train = ifelse(prob_mtry_rf_train$'1' >= 0.5, 1, 0)
cm_rf_train_mtry = table(Pred = pred_mtry_rf_train, Obs = db$Outcome)
print(cm_rf_train_mtry)
metrics_rf_train_mtry = confusionMatrix(cm_rf_train_mtry)
print(metrics_rf_train_mtry) #still perfect...

#on test set - not good, still a lot of overfitting...
prob_mtry_rf_test = predict(rf_model_mtry_tuned, newdata = test_set,)
pred_mtry_rf_test = ifelse(prob_mtry_rf_test$'1' >= 0.5, 1, 0)
cm_rf_mtry = table(Pred = pred_mtry_rf_test, Obs = test_set$Outcome)
print(cm_rf_mtry)
metrics_rf_train = confusionMatrix(cm_rf_train)
print(metrics_rf_train)

#so let's tune differently --> "brute force", min.node.size
set.seed(123)
tuneGrid_rf <- expand.grid(.mtry = seq(1, ncol(x), by = 2),
                        .splitrule = "gini",
                        .min.node.size = 150)

tuned_rf_model <- train(x, y, method = "ranger", 
                        n.tree = 1000,
                        importance = "impurity",
                  trControl = ctrl, 
                  tuneGrid = tuneGrid_rf)

print(tuned_rf_model$bestTune) 

# Make predictions on the training set
train_pred <- predict(tuned_rf_model, newdata = db)
test_pred <- predict(tuned_rf_model, newdata = test_set)

# Load the caret package for confusionMatrix function
library(caret)

# Create confusion matrix for the training set
confusionMatrix_train <- confusionMatrix(train_pred, db$Outcome)
print(confusionMatrix_train)

# Create confusion matrix for the test set
confusionMatrix_test <- confusionMatrix(test_pred, test_set$Outcome)
print(confusionMatrix_test)

#variable importance
# Get variable importance
importance <- tuned_rf_model$finalModel$variable.importance
print(importance)
tuned_rf_model$finalModel
# Create a data frame
importance_df <- data.frame(Variable = names(importance),
                            Importance = importance)
# Check the type of data
print(class(importance_df$Importance))

# Order by importance
importance_df <- importance_df[order(-importance_df$Importance), ]
importance_df
library(xtable)
xtable(importance_df, caption = "Variable Importance", type = "latex")
# Load ggplot2
library(ggplot2)
x11()
# Create the plot
ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "red1") +
  coord_flip() +
  theme_light() +
  xlab("Variable") +
  ylab("Importance") +
  ggtitle("Variable Importance Plot")


library(ggplot2)
# Traccia dell'accuratezza al variare del numero di alberi
x11()
ggplot(tuned_rf_model$results, aes(x = mtry, y = Accuracy)) +
  geom_line() +
  labs(title = "Accuracy across different mtry values", x = "mtry", y = "Accuracy")
