rm(list = ls())
finalalldata = finalalldata[,-1]
model = lm(label~., data = finalalldata)
summary(model)
