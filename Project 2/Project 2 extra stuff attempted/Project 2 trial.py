import math
from sklearn import datasets
iris = datasets.load_iris()
import random

from sklearn.model_selection import train_test_split

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


def shuffleAndGroup():
    df = iris.data
    a_train, a_test, b_train, b_test = train_test_split(df, iris.target, test_size=.3, random_state=3)
    return a_train, a_test, b_train, b_test

print("start here for get neighbors")

data_train, data_test , targets_train, targets_test = shuffleAndGroup()
dist = []
#length = len(data_test)-1
#print(data_test[0])


data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']

num = random.randint(1,len(data_test))
print(type(num))
print(num)
data_test_val = data_test[random.randint(1,len(data_test))]
print("try this")
#print(dest_val)
print("oh yeah")

for i in range(10):
    #print("i =", i)
    
    print("data_train i =", data_train[i], data_test[i])
    distance = euclideanDistance(data_train[i], data_test_val, 4)
    print("i =", i)
    dist.append([distance,data_train[i], data_test[i]])
    print("try =", i)
    
sorted_Neighbors = sorted(dist, key = lambda flower: flower[0])


print("done")
print(sorted_Neighbors)
print("neighbors")

#give the neighbors the target info
k = 3
neighbors = []
for i in range(k):
    neighbors.append(sorted_Neighbors[i][1])
    
print(neighbors)

print("neighbors done")
    
    
print(dist[0])
print(dist[0][1])
print(dist[1])
print(dist[1][1])
print(dist[2])
print(dist[2][1])
print(dist[3])
print("actual data", data_train[0])
print ('Distance: ' + repr(distance))

print(len(data_test))
print(len(data_train))
print(len(targets_train))
print(targets_test.shape)
#print (random.randint(1,len(data_test)))

#data1 = [2, 2, 2, 'a']
#data2 = [4, 4, 4, 'b']


#print(euclideanDistance(a, b, 3))
#print(euclideanDistance(a, c, 3))
