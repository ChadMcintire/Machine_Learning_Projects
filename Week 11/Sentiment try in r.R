library(data.table)
library(dplyr)
library(tidytext)
library(magrittr)
library(data.table)
mydata<-fread("C:/Users/Chad/Desktop/450/Week 11/Try3.csv", header = FALSE)
mydata <- mydata[, -c(27:40)]
View(mydata)

tidytext::unnest_tokens(twitfile, V2, urls, token = 'regex', pattern=" ")


#convert from a data.table to a data.frame
dfmydata <- as.data.frame(mydata) 
View(dfmydata)
text_df <- data_frame(line = 1:45275, text = dfmydata[,2])
text_df %>%
  unnest_tokens(word, dfmydata[,2])

str(dfmydata)



twitfile <- read.csv("C:/Users/Chad/Desktop/450/Week 11/Try3.csv", header = FALSE)
twitfile <- twitfile[, -c(27:40)]
twitfiletext <- twitfile[, 2]

length(twitfile[1, 2])
text_df <- data_frame(line = 45276, text = twitfile[, 2])
text_df
View(text_df[,2])

text_df %>%
  unnest_tokens(word, as.character(twitfile[, 2]))
