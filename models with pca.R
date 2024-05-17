rm(list = ls())
graphics.off()
db = read.csv("Training.csv")
test_set = read.csv("Testing.csv")
test_set <- test_set[!(test_set[, 3] == 0 | test_set[, 4] == 0 | test_set[, 5] == 0 | test_set[, 6] == 0), ]
test_set$Outcome = as.factor(test_set$Outcome)
db <- db[!(db[, 3] == 0 | db[, 4] == 0 | db[, 5] == 0 | db[, 6] == 0), ]
db$Outcome = as.factor(db$Outcome)
library(caret)
library(e1071)
set.seed(123)
x = db[,-9]
y = db[,9]
ctrl = trainControl(method = "cv",
                    number = 10)

#it's better to scale for PCA!
scaled_numericaldata = scale(x)
cor(scaled_numericaldata) #to interpret, this tells us if we need PCA
pca = princomp(scaled_numericaldata)
summary(pca)
pca_scores = pca$scores[,1:4]
data_pca = data.frame(pca_scores, Outcome = y)
head = head(data_pca)
head
library(xtable)
print(xtable(head, type ="latex"))
#let's divide this into training and test since hte test set does not have pca
set.seed(123)
trainIndex <- createDataPartition(data_pca$Outcome, p = .8, list = FALSE, times = 1)
train <- data_pca[ trainIndex,]
test  <- data_pca[-trainIndex,]
model_pca <- train(Outcome ~ .,
                   data = train, method = "glm",
                   trControl = ctrl)
print(model_pca)

predictions_train <- predict(model_pca, newdata = train)
confusionMatrix(predictions_train, train$Outcome)

predictions_test <- predict(model_pca, newdata = test)
confusionMatrix(predictions_test, test$Outcome)

model_pca_step <- train(Outcome ~ .,
                   data = train, method = "glmStepAIC",
                   trControl = ctrl)
print(model_pca_step)

predictions_train_step <- predict(model_pca_step, newdata = train)
confusionMatrix(predictions_train_step, train$Outcome)

predictions_test_step <- predict(model_pca_step, newdata = test)
confusionMatrix(predictions_test_step, test$Outcome)


model_pca_step <- train(Outcome ~ .,
                        data = train, method = "glmStepAIC",
                        trControl = ctrl)

tuneGrid_2 = expand.grid(.cp = seq(0.03, 0.5, 0.1)) #grid of cp values
tree_model_pca <- train(Outcome ~ .,data=train,
                        method = "rpart", trControl = ctrl, 
                        tuneGrid = tuneGrid_2)
print(tree_model_pca)
tree_model_pca$bestTune

predictions_train_tree <- predict(tree_model_pca, newdata = train)
confusionMatrix(predictions_train_tree, train$Outcome)

predictions_test_tree <- predict(tree_model_pca, newdata = test)
confusionMatrix(predictions_test_tree, test$Outcome)

library(rpart.plot)
x11()
rpart.plot(tree_model_pca$finalModel, extra = 1)

tune_svm = expand.grid(C = c(0.01, 0.1, 1, 10, 100, 1000))
svm_model_linear <- train(Outcome ~ .,data=train, method = "svmLinear", trControl = ctrl,
                          tuneGrid = tune_svm)
svm_model_linear$bestTune
print(svm_model_linear)

predictions_train_svm <- predict(svm_model_linear, newdata = train)
confusionMatrix(predictions_train_svm, train$Outcome)

predictions_test_svm <- predict(svm_model_linear, newdata = test)
confusionMatrix(predictions_test_svm, test$Outcome)

train_upsampled = upSample(train[,-5],train[,5],yname="Outcome")
table(train_upsampled$Outcome)

tuneGrid_2 = expand.grid(.cp = seq(0.03, 0.5, 0.1)) #grid of cp values
tree_model_pca <- train(Outcome ~ .,data=train_upsampled,
                        method = "rpart", trControl = ctrl, 
                        tuneGrid = tuneGrid_2)
print(tree_model_pca)
library(rpart.plot)
tree_model_pca$bestTune #cp = 0.03

predictions_train_tree <- predict(tree_model_pca, newdata = train_upsampled)
confusionMatrix(predictions_train_tree, train_upsampled$Outcome)

predictions_test_tree <- predict(tree_model_pca, newdata = test)
confusionMatrix(predictions_test_tree, test$Outcome)


                    
                        