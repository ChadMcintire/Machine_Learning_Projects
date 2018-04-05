library(arules)
data(Groceries)

rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.2))
rulesbyconfidence <-sort(rules, by="confidence", decreasing=TRUE)
rulesbylift <-sort(rules, by="lift", decreasing=TRUE)
rulesbysupport <-sort(rules, by="support", decreasing=TRUE)

inspect(rulesbyconfidence[1:5])
inspect(rulesbylift[1:5])
inspect(rulesbysupport[1:5])

rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.3))
rulesbyconfidence <-sort(rules, by="confidence", decreasing=TRUE)
rulesbylift <-sort(rules, by="lift", decreasing=TRUE)
rulesbysupport <-sort(rules, by="support", decreasing=TRUE)

inspect(rulesbyconfidence[1:5])
inspect(rulesbylift[1:5])
inspect(rulesbysupport[1:5])

rules <- apriori(Groceries, parameter = list(supp = 0.003, conf = 0.3))
rulesbyconfidence <-sort(rules, by="confidence", decreasing=TRUE)

rulesbysupport <-sort(rules, by="support", decreasing=TRUE)

inspect(rulesbyconfidence[1:5])
inspect(rulesbylift[1:5])
inspect(rulesbysupport[1:5])

inspect(rulesbysupport[1])[4]
inspect(rulesbyconfidence[1])[5]
inspect(rulesbylift[1])[6]
stuff <- inspect(rulesbylift[1])
rules <- apriori(Groceries, parameter = list(supp = 0.07, conf = 0.1))
rulesbylift <-sort(rules, by="lift", decreasing=TRUE)
stuff <- inspect(rulesbylift[1])
stuff$lift

rules <- apriori(Groceries, parameter = list(supp = 0.05, conf = 0.22))
rulesbysupport <- sort(rules,  decreasing = TRUE,by = "support")
highestsupp<- inspect(rulesbysupport[2:6])

rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 1.0))
rulesbysupport <- sort(rules,  decreasing = TRUE,by = "lift")
highestconf <- inspect(rulesbysupport[1:5])

rules <- apriori(Groceries, parameter = list(supp = 0.0003, conf = .008))
rulesbylift <- sort(rules,  decreasing = TRUE,by = "lift")
inspect(rulesbylift[220:230])


