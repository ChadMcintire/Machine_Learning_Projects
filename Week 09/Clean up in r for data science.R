#install.packages('e1071', dependencies = TRUE)
library(e1071)
iris
plot(iris)

MyData <- read.csv(file="c:/TheDataIWantToReadIn.csv", header=TRUE, sep=",")

s<- sample(150,40)
s

col <- c("Petal.Length" , "Petal.Width", "Species")
iris_train <- iris[s, col]


svmfit <- svm(Species ~., data = iris_train, kernel = "linear", cost = 100, scale = FALSE)

print(svmfit)

summary(svmfit)
plot(svmfit, iris_train[,col])

tuned <- tune(svm, Species~., data = iris_train, kernel = "linear", ranges = list(cost=c(.001,.01,.1,1,10,100)))

summary(tuned)

p <- predict(svmfit, iris_test[,col], type = "class")
plot(p)
table(p, iris_test[,3])
mean(p == iris_test[,3])
