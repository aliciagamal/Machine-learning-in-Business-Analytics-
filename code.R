#data description to be put at the beginning

rm(list = ls())
db = read.csv("finalalldata.csv")
#cleaning: we start by removing 2 unecessary columns, used to label the database
#for readibility but that are of no interest for the prediction of the class
db = db[,-c(1,16)]
#let's check if the classes are balanced
table(db$label)
#0: 192, 1:441 --> unbalanced
#univariate analysis
library(summarytools)
dfSummary(db)
library(dplyr)
db_summary <- dfSummary(db, max.distinct.values = 5)
db_summary%>% view()
#bivariate analysis: dfSummary divided by level of "label"
library(magrittr)
df_sum_label <- df %>% group_by(label) %>% dfSummary(max.distinct.values = 5)
df_sum_label %>% view()
#the thing above does not work but i will make it work soon 

# MISSING VALUES 
sum(is.na(db)) #100 missing values overall
sum(is.na(db[,1])) #0
sum(is.na(db[,2])) #0
sum(is.na(db[,3])) #0
sum(is.na(db[,4])) #0
sum(is.na(db[,5])) #0
sum(is.na(db[,6])) #0
sum(is.na(db[,7])) #9
sum(is.na(db[,8])) #5
sum(is.na(db[,9])) #4
sum(is.na(db[,10])) #5
sum(is.na(db[,11])) #4
sum(is.na(db[,12])) #35
sum(is.na(db[,13])) #0
sum(is.na(db[,14])) #38
sum(is.na(db[,15])) #0

#column 12 and 14 are critical because they have a lot of missing values
#we cannot delete these columns, as they are 
#critical medical features that likely heavily impact the outcome 
db_clean = na.omit(db) #we remove the instances with na values
table(db_clean$label)
#0: 171. 1: 382 --> still unbalanced, still a similar proportion than before
#we do not think that the fact that these features are missing in the dataset
#is related to the outcome 0 and 1.
db_summary_clean <- dfSummary(db_clean, max.distinct.values = 5)
db_summary_clean %>% view()
#some univariate statistics on the variables
library(psych)
describe(db_clean, omit=TRUE, skew=FALSE, IQR = TRUE)
str(db_clean)

#bivariate
## Numbers
## Summary statistics per deposit
## Caution: cat are also included
describe(db_clean~label, skew=FALSE, IQR = TRUE)
#bivariate: visualization
#boxplot
x11()
par(mfrow = c(1,2))
boxplot(db_clean$age~db_clean$label) #age is much older for people with label = 1
boxplot(db_clean$bmi~db_clean$label) #not a big difference in bmi
#now for smoker status and sex
# Assuming 'df' is your dataframe
library(ggplot2)
#for sex: 1 is women and 2 is men
db_sex1_clean = db_clean[db_clean$sex==1,]
db_sex2_clean = db_clean[db_clean$sex==2,]
table(db_sex1_clean$label) #0:87, 1:311
table(db_sex2_clean$label) #0:84, 1:71
table(db_clean$sex) #1:398 and 2: 155 
#it is unbalanced by sex 
library(ggplot2)
# Create a bar plot
x11()
ggplot(db_clean, aes(x=factor(sex), fill=factor(label))) +
  geom_bar(position="dodge") +
  scale_fill_discrete(name="Outcome", labels=c("0", "1")) +
  scale_x_discrete(name="Sex", labels=c("1", "2")) +
  theme_minimal() +
  labs(title="Distribution of Outcomes by Sex")

x11()
ggplot(db_clean, aes(x=factor(smoke), fill=factor(label))) +
  geom_bar(position="dodge") +
  scale_fill_discrete(name="Outcome", labels=c("0", "1")) +
  scale_x_discrete(name="Smoke", labels=c("0", "1")) +
  theme_minimal() +
  labs(title="Distribution of Outcomes by smoker status")
#the incidence of the illness is much greater in smokers (smoke==1)


#splitting training and test set 

#logistic regression + variable selection (stepwise and lasso)
#classification tree + pruning 
#support vector machine (linear vs radial) with tuning cost and sigma (if radial) parameters
#model scoring

# eventually PCA later on 

#discussion of results
#goodbye










