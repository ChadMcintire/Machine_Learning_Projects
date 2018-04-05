library(datasets)
myData = state.x77

getwd()
setwd("C:/Users/Chad/Desktop/450/Week 11")
write.csv(myData, file = "myData.csv")

library(tidyverse)
View(myData)

library(dplyr)
library(caret)
#for scaling
# Assuming goal class is column 10
distance = dist(as.matrix(myData))
hc = hclust(distance)
plot(hc)

preObj <- preProcess(myData, method=c("center", "scale"))
newData <- predict(preObj, myData)
distance = dist(as.matrix(newData))
hc = hclust(distance)
plot(hc)

myDataminArea <- newData[,1:7]
distance = dist(as.matrix(myDataminArea))
hc = hclust(distance)
plot(hc)

frost <- newData[,7]
distance = dist(as.matrix(frost))
hc = hclust(distance)
plot(hc)

#or this for scaling

data_scaled = scale(myData)

datCluster <- kmeans(data_scaled, 3, nstart = 20)

datCluster$centers

table(datCluster$cluster)

summary(datCluster)

vector <- integer(length(1:25))
vector2 <- integer(length(1:25))
for (i in 1:25) {
  datCluster <- kmeans(data_scaled, i, nstart = 20)
  vector[i] <- datCluster$tot.withinss
  vector2[i] <- i
}

df = data.frame(vector2,vector)

g <- ggplot( df, aes(vector2,vector))
g + geom_jitter()+
  geom_vline(xintercept = 9)

clusplot(data_scaled, datCluster$cluster, color = TRUE, shade = TRUE, labels = 2, lines = 0)


summary(datCluster$cluster)

sort(datCluster$cluster)
datCluster$centers

datCluster$cluster

datCluster$withinss
datCluster$tot.withinss

