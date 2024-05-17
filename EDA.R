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

# PCA: principal component analysis
# now we will perform PCA on the training set
# boxplot to see if I need to scale
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