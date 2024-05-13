rm(list = ls())
db = read.csv("Training.csv")
db <- db[!(db[, 3] == 0 | db[, 4] == 0 | db[, 5] == 0 | db[, 6] == 0), ]
db$Outcome = as.factor(db$Outcome)
#CLUSTERING 
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
db$cluster
table(db$Outcome,db$cluster)
#    1   2
# 0 203 603
# 1 326 141
#cluster 1: 18,94% sick
#cluster 2: 61,62% sick
#so there is an association between clusters and diabetes
#and it seems like it grows along the first components
X11()
graphics.off()
par(mfrow = c(3,1))
fviz_nbclust(db_clust, kmeans, method = "wss", k.max = 25, verbose = FALSE)
fviz_nbclust(db_clust, kmeans, method = "silhouette", k.max = 25, verbose = FALSE)
fviz_nbclust(db_clust, kmeans, method = "gap_stat", k.max = 25, verbose = FALSE)
kmeans_model_best = kmeans(db_clust,centers=4)

#silhouette comparison 
library(cluster)
library(factoextra)
x11()
par(mfrow = c(1,2))
sil = silhouette(kmeans_model$cluster, dist(db_clust))
plot(sil, col = c("#00BFC4", "#F8766D"), main = "with k = 2")
sil_best = silhouette(kmeans_model_best$cluster, dist(db_clust))
plot(sil_best, col = c("#00BFC4", "#F8766D", "#7CAE00", "#C77CFF"), main = "with k=4")


x11()
fviz_cluster(kmeans_model_best,data=db_clust)
#let's see how it relates to the outcome
#1. let's add the cluster assignments to the original dataframe
db$cluster = kmeans_model$cluster
table(db$Outcome,db$cluster)

#boxplots for each cluster
x11()
par(mfrow = c(1,2))
for(i in 1:2){
  cluster_data = db[db$cluster == i,]
  boxplot(cluster_data[,1:(ncol(cluster_data)-1)], main = paste("Cluster", i), las = 2,
          col = "red1", ylim = c(0,800))
}

#clustering no outliers
# Assuming 'df' is your data frame
indexes_to_remove <- c(4, 5, 112, 121, 137, 208, 295, 348, 351, 371, 380, 488, 512, 527, 640, 723, 773, 825, 826, 837, 876, 879, 973, 982, 1013, 1033, 1112, 1119, 1138)
db_clust <- db_clust[-indexes_to_remove, ] #db_clust is not scaled here, i had not run the scale()
scale(db_clust)
kmeans_noout = kmeans(db_clust,centers=2)
x11()
fviz_cluster(kmeans_noout,data=db_clust)
