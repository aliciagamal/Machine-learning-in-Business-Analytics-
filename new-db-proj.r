rm(list = ls())
db = read.csv("Training.csv")
library(summarytools)
library(psych)
library(ggplot2)
#EDA
#1-check for missing values 
sum(is.na(db)) #0 missing values
#however we see that bloodpressure= 0, this is not possible. so this represents missing data
#there are no explicit missing values however
#blood pressure, skin thickness, insulin level and BMI = 0 --> this is not possible in real life, it indicates missing data.
#we cannot remove all these features (they are very important)
#nor would it be wise to subsitute them with the mean 
#so we just remove all the instances in which they are 0
db <- db[!(db[, 3] == 0 | db[, 4] == 0 | db[, 5] == 0 | db[, 6] == 0), ]
nrow(db) #1273 instances

#2- check if the dataset is balanced
table(db$Outcome) #0:806 and 1:467: the classes are not well balanced
db$Outcome = as.factor(db$Outcome)
#distribution of target
x11()
barplot(table(db$Outcome), col = "red1",
        main = "Distribution of Diabetes Outcome")
#univariate analysis
#univariate descriptive statistics
describe(db, omit=TRUE, skew=FALSE, IQR = TRUE)
str(db)
db_summary<- dfSummary(db, max.distinct.values = 5)
db_summary %>% view()

#identify outliers
#1. boxplots 
x11()
boxplot(db, col = "red1") 
#insulin has a lot of outliers 

#bivariate analysis: explore the relationship between some predictors and the outcome
#bivariate descriptive statistics

#i have not seen if line 42-47 runs well 
db_with_diabetes = db[db$Outcome == "1", ]
db_without_diabetes = db[db$Outcome == "0", ]
db_summary_diabetes<- dfSummary(db_with_diabetes, max.distinct.values = 5)
db_summary_diabetes %>% view()
db_summary_nodiabetes <- dfSummary(db_without_diabetes, max.distinct.values = 5)
db_summary_nodiabetes %>% view()

#boxplots
x11()
par(mfrow = c(2,2))
boxplot(db$Age~db$Outcome, col = "red1")
boxplot(db$BMI~db$Outcome, col = "red1")
boxplot(db$Pregnancies~db$Outcome, col = "red1")
boxplot(db$Insulin~db$Outcome, col = "red1")
#bivariate descriptive statistics
describe(db~Outcome, skew=FALSE, IQR = TRUE)

#correlation matrix
correlation_matrix = cor(db)
library(ggplot2)
library(reshape2)
melted_correlation = melt(correlation_matrix)
#plot heatmap of the correlation
x11()
ggplot(melted_correlation, aes(Var1, Var2, fill = value, label = round(value, 2))) +
  geom_tile(color = "white") +
  geom_text(color = "black", size = 3) +  # Add text annotations
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0,
                       limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  coord_fixed()
#the highest correlation is 0.71, between bmi and skin thickness

#some useful libraries
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
#logistic regression
logistic_model <- train(x, y, method = "glm", trControl = ctrl)
logistic_summary <- summary(logistic_model)
logistic_summary #1237.5

#logistic regression with stepwise selection
stepwise_model <- train(x, y, method = "glmStepAIC", trControl = ctrl)
stepwise_summary <- summary(stepwise_model)
stepwise_summary #selects pregnancies, glucose, insulin, BMI, diabetes pedigree, age
#AIC: 1234.1

#logistic regression with lasso
lambda_values <- 10^seq(10, -2, length.out = 100)
tune_grid <- expand.grid(alpha = 1, lambda = lambda_values)
lasso_model <- train(x, y, method = "glmnet", trControl = ctrl, tuneGrid = tune_grid)
lasso_summary <- summary(lasso_model)
lasso_summary
best_model <- lasso_model$finalModel
coef <- coef(best_model, s = lasso_model$bestTune$lambda)
print(coef) #this is the important line to see what was selected
#it puts skin thickness to 0, so it does not select that

#PARAMETER TUNING: TUNE LAMBDA
#1. confusion matrix and statistics to check the situation
#see balanced accuracy and low sens or accuracy
#look at ROC-AIC?


#classification tree: pruned with autoprune
tuneGrid = expand.grid(.cp = seq(0.01, 0.5, 0.01)) #grid of cp values
tree_model <- train(x, y, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid)
tree_summary <- summary(tree_model)
print(tree_model$bestTune) #cp: 0.01
library(rpart.plot)
x11()
rpart.plot(tree_model$finalModel, extra = 1)
#classification tree with randomforest
rf_model <- train(x, y, method = "rf", trControl = ctrl)
rf_summary <- summary(rf_model)
rf_summary
x11()
varImpPlot(rf_model$finalModel)

#plot of OOB error rate
x11()
plot(rf_model$finalModel,
     main = "OOB error rate across trees")

#partial dependence plot
#highlights marginal effect of a variable o the class probability
library(pdp)
x11()
partial(rf_model$finalModel, pred.var = "Glucose", plot = TRUE)
partial(rf_model$finalModel, pred.var = "BloodPressure", plot = TRUE)
partial(rf_model$finalModel, pred.var = "SkinThickness", plot = TRUE)
partial(rf_model$finalModel, pred.var = "Insulin", plot = TRUE)
partial(rf_model$finalModel, pred.var = "BMI", plot = TRUE)
partial(rf_model$finalModel, pred.var = "DiabetesPedigreeFunction", plot = TRUE)
partial(rf_model$finalModel, pred.var = "Age", plot = TRUE)

#now we want to choose m (mtry) - we tune it
tuneGrid = expand.grid(.mtry=c(1:ncol(x)))
rf_model_mtry_tuned <- train(x, y, method = "rf", trControl = ctrl, tuneGrid = tuneGrid)
summary(rf_model_mtry_tuned)
rf_model_mtry_tuned
#mtry = 1 --> very weird! accuracy = 1.
#there must be something wrong... we will ask 

library(kernlab)
set.seed(123)
#support vector machine: BEFORE HYPERPARAMETER TUNING
svm_model_linear <- train(x, y, method = "svmLinear", trControl = ctrl)
svm_model_linear #accuracy: 0.7690391, kappa = 0.4774859
#default C = 1
svm_model_radial <- train(x,y, method = "svmRadial", trControl = ctrl)
svm_model_radial
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
svm_linear_tuned <- train(x,y, method = "svmLinear",
                         trControl= ctrl,
                         tuneGrid = grid)
svm_linear_tuned
x11()
plot(svm_linear_tuned)
svm_linear_tuned$bestTune

#tuning the radial model 
grid_radial <- expand.grid(sigma = c(0.01, 0.02, 0.05, 0.1),
                           C = c(1, 10, 100, 500, 1000))
set.seed(123)
svm_radial_tuned <- train(x,y, method = "svmRadial",
                         trControl=ctrl,
                         tuneGrid = grid_radial)
svm_radial_tuned
plot(svm_radial_tuned)
svm_radial_tuned$bestTune

#PCA: principal component analysis 
#now we will perform PCA on the training set
#boxplot to see if I need to scale
x11()
boxplot(x, col = "red1")
#it's better to scale for PCA!
scaled_numericaldata = scale(x)
cor(scaled_numericaldata) #to interpret, this tells us if we need PCA
pca = princomp(scaled_numericaldata)
summary(pca)
#visualize the meaning of loadings (we are taking the first 2 components)
x11()
par(mfrow = c(2,1))
for(i in 1:2)
  barplot(pca$loadings[,i],ylim = c(-1,1), col = "red1")
pca$loadings
#PCA biplot
library(ggfortify)
library(ggplot2)
x11()
autoplot(pca, loadings = TRUE, loadings.label = TRUE)

#which dimension reduction could we do? 

#let's load the test set
test_set = read.csv("Testing.csv")
nrow(test_set) #308 instances
#let's also clean it from missing values
test_set <- test_set[!(test_set[, 3] == 0 | test_set[, 4] == 0 | test_set[, 5] == 0 | test_set[, 6] == 0), ]
nrow(test_set) #156 test instances
table(test_set$Outcome)
#0:111 and 1:45  

#let's now evaluate the prediction performance of each model
#WARNING: I have not run this code yet, idk if it works
#it's just a VERY rough draft!!!

#logistic regression 
prob_test_visit_logreg = predict(logistic_model, newdata = test_set , type= "response")
pred_test_visit_logreg = ifelse(prob_test_visit_logreg >= 0.5,1,0)
table(Pred=pred_test_visit_logreg, Obs = test_set$Outcome)
#logistic regression with stepwise
prob_test_visit_stepwise = predict(stepwise_model, newdata = test_set , type= "response")
pred_test_visit_stepwise = ifelse(prob_test_visit_stepwise >= 0.5,1,0)
table(Pred=pred_test_visit_stepwise, Obs = test_set$Outcome)
#logistic regression with lasso 
prob_test_visit_lasso = predict(lasso_model, newdata = test_set , type= "response")
pred_test_visit_lasso = ifelse(prob_test_visit_lasso >= 0.5,1,0)
table(Pred=pred_test_visit_lasso, Obs = test_set$Outcome)
#classification tree 
pred <- predict(tree_model, newdata= test_set, type="class")
table(Pred=pred, Obs=test_set$Outcome)
#to get the probabilities
predict(tree_model, newdata=test_set, type="prob")
#classification tree with random forest

#svm linear tuned 
library(caret)
linear_svm_tuned_pred <- predict(svm_linear_tuned, newdata = test_set)
table(Pred=linear_svm_tuned_pred, obs=test_set$Outcome)
confusionMatrix(data=linear_svm_tuned_pred, reference = test_set$Outcome)
#svm radial tuned
radial_svm_tuned_pred <- predict(svm_radial_tuned, newdata = test_set)
table(Pred=radial_svm_tuned_pred, obs=test_set$Outcome)
confusionMatrix(data=radial_svm_tuned_pred, reference = test_set$Outcome)

#confusion matrices
#https://do-unil.github.io/mlba/labs/04_Metrics/Ex_ML_Scoring.html#classification-task
#section: confusion matrices & prediction-based measures

#caret::twoClassSummary --> to do AUC - ROC
#after i do ROC I can tune the probablity threshold
#tune it on the training set (on logistic regression ok, in the other models??)






