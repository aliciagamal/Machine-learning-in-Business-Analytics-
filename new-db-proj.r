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
library(magrittr)
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
cor(db)
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

#PCA: principal component analysis 
#now we will perform PCA on the training set
#boxplot to see if I need to scale
# x11()
# boxplot(x, col = "red1")
# #it's better to scale for PCA!
# scaled_numericaldata = scale(x)
# cor(scaled_numericaldata) #to interpret, this tells us if we need PCA
# pca = princomp(scaled_numericaldata)
# summary(pca)
# #visualize the meaning of loadings (we are taking the first 2 components)
# x11()
# par(mfrow = c(2,1))
# for(i in 1:2)
#   barplot(pca$loadings[,i],ylim = c(-1,1), col = "red1")
# pca$loadings
# #PCA biplot
# library(ggfortify)
# library(ggplot2)
# x11()
# autoplot(pca, loadings = TRUE, loadings.label = TRUE)

#which dimension reduction could we do? 
#models
ctrl = trainControl(method = "cv",
                    number = 10)
#logistic regression
logistic_model <- train(x, y, method = "glm", trControl = ctrl)
logistic_summary <- summary(logistic_model)
logistic_summary #AIC: 1237.5

#pred on training 
prob_test_visit_logreg_train = predict(logistic_model, newdata = db , type= "prob")
pred_test_visit_logreg_train = ifelse(prob_test_visit_logreg_train$'1' >= 0.5,1,0)
cm_logreg_train = table(Pred=pred_test_visit_logreg_train, Obs = db$Outcome)
cm_logreg_train
metrics_logreg_train = confusionMatrix(cm_logreg_train)
metrics_logreg_train

#logistic regression with stepwise selection
stepwise_model <- train(x, y, method = "glmStepAIC", trControl = ctrl)
stepwise_summary <- summary(stepwise_model)
stepwise_summary #selects pregnancies, glucose, insulin, BMI, diabetes pedigree, age
#AIC: 1234.1

#prediction on training of stepwise selection
prob_test_visit_stepwise_train = predict(stepwise_model, newdata = db , type= "prob")
pred_test_visit_stepwise_train = ifelse(prob_test_visit_stepwise_train$'1' >= 0.5,1,0)
cm_stepwise_train = table(Pred=pred_test_visit_stepwise_train, Obs = db$Outcome)
cm_stepwise_train
metrics_stepwise_train = confusionMatrix(cm_stepwise_train)
metrics_stepwise_train

#logistic regression with lasso
lambda_values <- 10^seq(10, -2, length.out = 100)
tune_grid <- expand.grid(alpha = 1, lambda = lambda_values)
lasso_model <- train(x, y, method = "glmnet", trControl = ctrl, tuneGrid = tune_grid)
lasso_summary <- summary(lasso_model)
lasso_summary
best_model <- lasso_model$finalModel
coef <- coef(best_model, s = lasso_model$bestTune$lambda)
print(coef) #this is the important line to see what was selected
#the lasso model shrinks the coefficients, but it does not put 
#anything to 0, it does not select anything 

#logistic regression with lasso: metrics on training set
prob_test_visit_lasso_train = predict(lasso_model, newdata = db , type= "prob")
pred_test_visit_lasso_train = ifelse(prob_test_visit_lasso_train$'1' >= 0.5,1,0)
cm_lasso_train = table(Pred=pred_test_visit_lasso_train, Obs = db$Outcome)
cm_lasso_train
metrics_lasso_train = confusionMatrix(cm_lasso_train)
metrics_lasso_train

#PARAMETER TUNING: TUNE LAMBDA
#the only real interesting thing about this for the report are the values of lambda
library(pROC)
roc_obj_logreg = roc(db$Outcome, prob_test_visit_logreg_train$'1')
X11()
roc_logreg <- plot.roc(roc_obj_logreg, main="ROC Curve for Logistic Regression")

roc_obj_stepwise <- roc(db$Outcome, prob_test_visit_stepwise_train$'1')
roc_stepwise <- plot.roc(roc_obj_stepwise, main="ROC Curve for Stepwise Selection")

roc_obj_lasso <- roc(db$Outcome, prob_test_visit_lasso_train$'1')
roc_lasso <- plot.roc(roc_obj_lasso, main="ROC Curve for Lasso")

roc_obj <- roc(db$Outcome, prob_test_visit_logreg_train$'1')
coords_obj <- coords(roc_obj, "best")
lambda_logreg <- coords_obj["threshold"]
lambda_logreg #0.2979084

roc_obj <- roc(db$Outcome, prob_test_visit_stepwise_train$'1')
coords_obj <- coords(roc_obj, "best")
lambda_step <- coords_obj["threshold"]
lambda_step#0.3006133

# Tuning the probability threshold lambda for lasso
roc_obj <- roc(db$Outcome, prob_test_visit_lasso_train$'1')
coords_obj <- coords(roc_obj, "best")
lambda_lasso <- coords_obj["threshold"]
lambda_lasso # 0.3006824

#refit the 3 models with the new thresholds
#logistic normal
prob_test_visit_logreg_train = predict(logistic_model, newdata = db , type= "prob")
pred_test_visit_logreg_train = ifelse(prob_test_visit_logreg_train$'1' >= 0.2979084,1,0)
cm_logreg_train = table(Pred=pred_test_visit_logreg_train, Obs = db$Outcome)
cm_logreg_train
metrics_logreg_train = confusionMatrix(cm_logreg_train)
metrics_logreg_train

#logistic step 
prob_test_visit_stepwise_train = predict(stepwise_model, newdata = db , type= "prob")
pred_test_visit_stepwise_train = ifelse(prob_test_visit_stepwise_train$'1' >= 0.3006133,1,0)
cm_stepwise_train = table(Pred=pred_test_visit_stepwise_train, Obs = db$Outcome)
cm_stepwise_train
metrics_stepwise_train = confusionMatrix(cm_stepwise_train)
metrics_stepwise_train

#logistic regression with lasso: metrics on training set
prob_test_visit_lasso_train = predict(lasso_model, newdata = db , type= "prob")
pred_test_visit_lasso_train = ifelse(prob_test_visit_lasso_train$'1' >= 0.3006824,1,0)
cm_lasso_train = table(Pred=pred_test_visit_lasso_train, Obs = db$Outcome)
cm_lasso_train
metrics_lasso_train = confusionMatrix(cm_lasso_train)
metrics_lasso_train


#classification tree: pruned
tuneGrid = expand.grid(.cp = seq(0.01, 0.5, 0.01)) #grid of cp values
tree_model <- train(x, y, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid)
tree_summary <- summary(tree_model)
print(tree_model$bestTune) #cp: 0.01
library(rpart.plot)
x11()
rpart.plot(tree_model$finalModel, extra = 1)
#classification tree: metrics
prob_train_tree = predict(tree_model, newdata = db, type = "prob")
pred_train_tree = ifelse(prob_train_tree$'1'>=0.5,1,0)
cm_tree_train = table(Pred = pred_train_tree, Obs = db$Outcome)
print(cm_tree_train)
metrics_tree_train = confusionMatrix(cm_tree_train)
print(metrics_tree_train)

#classification tree with randomforest
rf_model <- train(x, y, method = "rf", trControl = ctrl)
rf_summary <- summary(rf_model)
rf_summary
x11()
varImpPlot(rf_model$finalModel)
#predictions and metrics
# Fai predizioni con il tuo modello di Random Forest
prob_test_rf_train = predict(rf_model, newdata = db, type = "prob")
pred_test_rf_train = ifelse(prob_test_rf_train$'1' >= 0.5, 1, 0)
cm_rf_train = table(Pred = pred_test_rf_train, Obs = db$Outcome)
print(cm_rf_train)
metrics_rf_train = confusionMatrix(cm_rf_train)
print(metrics_rf_train)
#on test set (still 1???)
prob_test_rf_train = predict(rf_model, newdata = test_set, type = "prob")
pred_test_rf_train = ifelse(prob_test_rf_train$'1' >= 0.5, 1, 0)
cm_rf_train = table(Pred = pred_test_rf_train, Obs = db$Outcome)
print(cm_rf_train)
metrics_rf_train = confusionMatrix(cm_rf_train)
print(metrics_rf_train)



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
#ok I re-run the code and I got mtry=2 and now 
#the accuracy is 0.9984252...
#did I set the seed wrong? no

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
x11()
plot(svm_linear_tuned)
svm_linear_tuned$bestTune

#predicition on the training set
# linear_svm_tuned_pred_train <- predict(svm_linear_tuned, newdata = select(db, -Outcome)) #error: kernlab class prediction calculations failed; returning NAs
new_db <- rename(db, y = Outcome) 
str(new_db)
str(tr_data)
linear_svm_tuned_pred_train <- predict(svm_linear_tuned, newdata = new_db) #error: kernlab class prediction calculations failed; returning NAs
cm_svm_train <- table(Pred = linear_svm_tuned_pred_train, Obs = db$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
# Obs
# Pred   0   1
#       0 712 189
#       1  94 278
metrics_svm_train = confusionMatrix(cm_svm_train)
metrics_svm_train

#tuning the radial model 
grid_radial <- expand.grid(sigma = c(0.01, 0.02, 0.05, 0.1),
                           C = c(1, 10, 100, 500, 1000))
set.seed(123)
svm_radial_tuned <- train(x,y, method = "svmRadial",
                         trControl=ctrl,
                         tuneGrid = grid_radial)
svm_radial_tuned
plot(svm_radial_tuned)
svm_radial_tuned$bestTune #sigma 0.1 and C=500

#prediction on the training set
radial_svm_tuned_pred_train <- predict(svm_radial_tuned, newdata = select(db,-Outcome)) 
cm_svm_train <- table(Pred = radial_svm_tuned_pred_train, Obs = db$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train


# #BOOTSTRAPPED VERSIONS
# ctrl_boot = trainControl(method = "boot632", number = 200)
# #bootstrapped logistic regression
# log_boot = train(x,y,method="glm",trControl=ctrl_boot)
# log_boot_summary <- summary(log_boot)
# log_boot_summary
# #pred on training
# prob_test_visit_logreg_train_boot = predict(log_boot, newdata = db , type= "prob")
# pred_test_visit_logreg_train_boot = ifelse(prob_test_visit_logreg_train_boot$'1' >= 0.5,1,0)
# cm_logreg_train_boot = table(Pred=pred_test_visit_logreg_train_boot, Obs = db$Outcome)
# cm_logreg_train_boot
# metrics_logreg_train_boot = confusionMatrix(cm_logreg_train)
# metrics_logreg_train_boot
# #logistic regression with stepwise selection
# stepwise_boot <- train(x, y, method = "glmStepAIC", trControl = ctrl_boot)
# stepwise_summary_boot <- summary(stepwise_boot)
# stepwise_summary_boot #AIC: 1234.1 takes long to run
# #prediction on training 
# prob_test_visit_stepwise_train_boot = predict(stepwise_boot, newdata = db , type= "prob")
# pred_test_visit_stepwise_train_boot = ifelse(prob_test_visit_stepwise_train_boot$'1' >= 0.5,1,0)
# cm_stepwise_train_boot = table(Pred=pred_test_visit_stepwise_train_boot, Obs = db$Outcome)
# cm_stepwise_train_boot
# metrics_stepwise_train_boot = confusionMatrix(cm_stepwise_train_boot)
# metrics_stepwise_train_boot




#EVALUATING PERFORMANCES ON THE TEST SET

#let's load the test set
test_set = read.csv("Testing.csv")
nrow(test_set) #308 instances
#let's also clean it from missing values
test_set <- test_set[!(test_set[, 3] == 0 | test_set[, 4] == 0 | test_set[, 5] == 0 | test_set[, 6] == 0), ]
nrow(test_set) #156 test instances
table(test_set$Outcome)
#0:111 and 1:45  
x_test = test_set[,-9]
y_test = test_set[,9]
y_test = as.factor(y_test)
test_set$Outcome = as.factor(test_set$Outcome)
#logistic regression  
prob_test_visit_logreg = predict(logistic_model, newdata = test_set , type= "prob")
pred_test_visit_logreg = ifelse(prob_test_visit_logreg$'1' >= 0.5,1,0)
cm_logreg = table(Pred=pred_test_visit_logreg, Obs = test_set$Outcome)
cm_logreg
# Obs
# Pred  0  1
# 0 99 14
# 1 12 31
library(caret)
cm_logreg = as.table(cm_logreg)
metrics_logreg = confusionMatrix(cm_logreg)
print(metrics_logreg)
#accuracy: 0.8333 
#no information rate: 0.7115
#kappa: 0.5886  
#sensitivity: 0.8919
#specificity: 0.6889
#pos pred value: 0.8761
#neg pred value: 0.7209
#balanced accuracy: 0.7904
#POSITIVE CLASS 0!!!


#logistic regression with stepwise
prob_test_visit_stepwise = predict(stepwise_model, newdata = test_set , type= "prob")
pred_test_visit_stepwise = ifelse(prob_test_visit_stepwise$'1' >= 0.5,1,0)
cm_stepwise = table(Pred=pred_test_visit_stepwise, Obs = test_set$Outcome)
cm_stepwise
# #     Obs
# Pred  0  1
#    0 99 13
#    1 12 32
cm_stepwise = as.table(cm_stepwise)
metrics_stepwise = confusionMatrix(cm_stepwise)
print(metrics_stepwise)

#logistic regression with lasso 
prob_test_visit_lasso = predict(lasso_model, newdata = test_set , type= "prob")
pred_test_visit_lasso = ifelse(prob_test_visit_lasso$'1' >= 0.5,1,0)
cm_lasso = table(Pred=pred_test_visit_lasso, Obs = test_set$Outcome)
cm_lasso
# #    Obs
# Pred  0  1
#     0 99 15
#     1 12 30
cm_lasso = as.table(cm_lasso)
metrics_lasso = confusionMatrix(cm_lasso)
print(metrics_lasso)

#classification tree  
pred <- predict(tree_model, newdata= test_set, type="prob")
pred_class <- ifelse(pred$'1' >= 0.5, 1, 0)
cm_tree = table(Pred = pred_class, Obs = test_set$Outcome)
cm_tree <- as.table(cm_tree)
cm_tree
#       Obs
# Pred  0  1
#     0 84 15
#     1 27 30
metrics_tree <- confusionMatrix(cm_tree)
print(metrics_tree)

#classification tree with random forest ... doesn't run...
pred <- predict(rf_model, newdata= test_set, type="prob")
pred_class <- ifelse(pred$'1' >= 0.5, 1, 0)
cm_rf = table(Pred = pred_class, Obs = test_set$Outcome)
cm_rf <- as.table(cm_rf)
cm_rf
# #    Obs
# Pred  0  1
#     0 98 13
#     1 13 32
metrics_rf <- confusionMatrix(cm_rf)
print(metrics_rf)

#tuned mtry classification tree with random forest rf_model_mtry_tuned
pred <- predict(rf_model_mtry_tuned, newdata= test_set, type="prob")
pred_class <- ifelse(pred$'1' >= 0.5, 1, 0)
cm_rf_tuned = table(Pred = pred_class, Obs = test_set$Outcome)
cm_rf_tuned <- as.table(cm_rf_tuned)
cm_rf_tuned
#       Obs
# Pred  0  1
#     0 99 17
#     1 12 28
metrics_rf_tuned <- confusionMatrix(cm_rf_tuned)
print(metrics_rf_tuned)

#svm linear tuned - it works
library(caret)
linear_svm_tuned_pred <- predict(svm_linear_tuned, newdata = test_set) 
cm_svm <- table(Pred = linear_svm_tuned_pred, Obs = test_set$Outcome)
cm_svm = as.table(cm_svm)
cm_svm
metrics_svm = confusionMatrix(cm_svm)
metrics_svm
#svm radial tuned
#prediction on the training set
radial_svm_tuned_pred_test <- predict(svm_radial_tuned, newdata = select(test_set,-Outcome)) 
str(test_set)
cm_svm_train <- table(Pred = radial_svm_tuned_pred_test, Obs = test_set$Outcome)
cm_svm_train = as.table(cm_svm_train)
cm_svm_train
# Obs
# Pred  0  1
#     0 90 15
#     1 21 30

#caret::twoClassSummary --> to do AUC - ROC
#after i do ROC I can tune the probablity threshold
#tune it on the training set (on logistic regression ok, in the other models??)

#EDA ON THE TEST SET TO SEE WHY 
x11()
boxplot(test_set, col = "red1")

x11()
par(mfrow = c(2,2))
boxplot(test_set$Age~test_set$Outcome, col = "red1")
boxplot(test_set$BMI~test_set$Outcome, col = "red1")
boxplot(test_set$Pregnancies~test_set$Outcome, col = "red1")
boxplot(test_set$Insulin~test_set$Outcome, col = "red1")

x11()
barplot(table(test_set$Outcome), col = "red1",
        main = "Distribution of Diabetes Outcome in the Test Set")



#CLUSTERING 
head(db)
dim(db)
db_clust = db[,-9]
head(db_clust)
db_clust = scale(db_clust)
#hierarchical
library(cluster)
library(factoextra)
dist_mat = dist(db_clust)
hclust_model = hclust(dist_mat)
x11()
plot(hclust_model)
#cut tree into 2 clusters
groups_hclust <- cutree(hclust_model,k=2)

#k-means
kmeans_model = kmeans(db_clust,centers=2)
x11()
fviz_cluster(kmeans_model,data=db_clust)
#let's see how it relates to the outcome
#1. let's add the cluster assignments to the original dataframe
db$cluster = kmeans_model$cluster
table(db$Outcome,db$cluster)
# 1   2
# 0 603 203
# 1 141 326
#cluster 1: 18,94% sick
#cluster 2: 61,62% sick
#so there is an assocition between clusters and diabetes
#and it seems like it grows along the first components
X11()
fviz_nbclust(db_clust, kmeans, method = "wss", k.max = 25, verbose = FALSE)

kmeans_model_best = kmeans(db_clust,centers=4)
x11()
fviz_cluster(kmeans_model_best,data=db_clust)
#let's see how it relates to the outcome
#1. let's add the cluster assignments to the original dataframe
db$cluster = kmeans_model$cluster
table(db$Outcome,db$cluster)