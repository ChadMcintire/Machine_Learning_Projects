#install.packages('e1071', dependencies = TRUE)
library(e1071)
iris
plot(iris)

letters <- read.csv(file="C:/Users/Chad/Desktop/450/Week 09/letters.csv", header=TRUE, sep=",")
vowel <- read.csv(file="C:/Users/Chad/Desktop/450/Week 09/vowel.csv", header=TRUE, sep=",")
?vowel
View(vowel)
typeof(letters$y_box)
letters <- scale(letters$letter)

s <- sample(length(letters$letter),length(letters$letter)*.75)
s

?svm()

letter_train <- letters[s, colnames(letters)]
letter_test <- letters[-s, colnames(letters)]
yeah <- c(1,.1, .01, .001)


for (i in yeah){
  for(j in yeah){    
    print(i)
    print(j)
    svmfit <- svm(letter ~., data = letters, kernel = "linear", cost = i, gamma = j)

    prediction <- predict(svmfit, letter_test[,-1])
    
    confusionMatrix <- table(pred = prediction, true = letter_test$letter)
    
    agreement <- prediction == letter_test$letter
    accuracy <- prop.table(table(agreement))
    
    print(confusionMatrix)
    print(accuracy)
    }
}


#print(svmfit)

#summary(svmfit)
#plot(svmfit, letter_train[,col])

prediction <- predict(svmfit, letter_test[,-1])

confusionMatrix <- table(pred = prediction, true = letter_test$letter)

agreement <- prediction == letter_test$letter
accuracy <- prop.table(table(agreement))


print(confusionMatrix)
print(accuracy)
#tuned <- tune(svm, letter~., data = letter_train, kernel = "linear", ranges = list(cost=c(.001,.01,.1,1,10,100)))

