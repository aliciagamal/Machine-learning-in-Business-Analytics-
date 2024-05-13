graphics.off()
# Open a new graphics window
x11()
# Plot the ROC curve for the Logistic Regression model
plot.roc(roc_obj_logreg, main="ROC Curves", col="#00BFC4")
# Add the ROC curve for the Lasso model to the existing plot
lines.roc(roc_obj_lasso, col="#F8766D")
# Add the ROC curve for the Stepwise Selection model to the existing plot
lines.roc(roc_obj_stepwise, col="#7CAE00")
# Add a legend to the plot
legend("bottomright", legend=c("Logistic Regression", "Lasso", "Stepwise Selection"), col=c("#00BFC4", "#F8766D", "#7CAE00"), lty=1)
