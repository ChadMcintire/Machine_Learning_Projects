letters <- read.csv(file="C:/Users/Chad/Desktop/450/Week 09/letters.csv", header=TRUE, sep=",")
vowel <- read.csv(file="C:/Users/Chad/Desktop/450/Week 09/vowel.csv", header=TRUE, sep=",")

vowel <- subset( vowel, select = -c(1:2) )
View(vowel)
vowel <- na.omit(vowel)
s <- sample(length(vowel$Class),length(vowel$Class)*.75)
s

letter_train <- vowel[s, colnames(vowel)]
letter_test <- vowel[-s, colnames(vowel)]
yeah <- c(1,.1, .01, .001)


for (i in yeah){
  for(j in yeah){    
    print(i)
    print(j)
    svmfit <- svm(Class ~., data = vowel, kernel = "radial", cost = i, gamma = j)
    prediction <- predict(svmfit, letter_test[,-11])
    
    confusionMatrix <- table(pred = prediction, true = letter_test$Class)
    
    agreement <- prediction == letter_test$Class
    accuracy <- prop.table(table(agreement))
    print(confusionMatrix)
    print(accuracy)
  }
}


#print(svmfit)

#summary(svmfit)
#plot(svmfit, letter_train[,col])

prediction <- predict(svmfit, letter_test[,-13])

confusionMatrix <- table(pred = prediction, true = letter_test$letter)

agreement <- prediction == letter_test$letter
accuracy <- prop.table(table(agreement))


print(confusionMatrix)
print(accuracy)