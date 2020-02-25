import sys,os
import numpy as np
import numpy
import scipy.io.wavfile
from math import sqrt
import matplotlib.pyplot as plt
import random

NUM_ITERATIONS = 6#change
#####MUST CHANGE THE DIR OF THE FILES!!!!!!!!!!!
dir_path = os.path.dirname(os.path.realpath(__file__))

sample, centriods = sys.argv[1], sys.argv[2]
sample = dir_path + "\\" + sample
centriods = dir_path + "\\" + centriods
fs, y = scipy.io.wavfile.read(sample)
x = np.array(y.copy())
centrioads = np.loadtxt(centriods)
print(type(centrioads))
file = open(dir_path + "\\" + "output.txt", "w+")

'''
x1 = random.sample(range(-1000,1000), 500)
y1 = random.sample(range(-1000,1000), 500)
l1 = list(zip(x1,y1))
print(l1)
l2 = numpy.array(l1)
print(l2)
x = l2
'''

'''
c1 = random.sample(range(10), 10)
c2 = random.sample(range(10), 10)
l3 = list(zip(c1,c1))
l4 = numpy.array(l3)
centrioads = l4
print(type(centrioads))
'''

def calc_euclid_dist(p1 ,p2):
    return float(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def sumPoints(list):
    if (len(list) == 0):
        return
    n = len(list)
    listX, listY = zip(*list)
    return (sum(listX) / n, sum(listY) / n)

prevStr = ""

newCents = {k: [] for k in range(len(centrioads))}

total_loss = []

for iter in range(0,NUM_ITERATIONS):
    iter_loss = 0
    for i in x:
        #min_dis =  float("inf")
        closest = centrioads[0]
        index = 0
        min_dis = calc_euclid_dist(i,closest)
        curr = 0
        for j in centrioads[1:centrioads.shape[0]]:
        #curr = 0
        #index = 0
        #for j in centrioads:
            curr = curr + 1
            dist = calc_euclid_dist(i, j)
            if(min_dis > dist):
                min_dis = dist
                closest = j
                index = curr
        newCents.get(index).append(i)
        iter_loss += min_dis
    print("iteration:", iter)
    print("loss:", iter_loss / len(x))
    total_loss.append((iter, iter_loss / len(x)))

    for i, vals in enumerate(newCents.values()):
        c = sumPoints(vals)
        centrioads[i] = c
        centrioads[i] = centrioads[i].round()

    #write to file

    file.write("[iter " + str(iter) + "]:")
    s = ""
    for i in centrioads:
        s += str(i)
        s += ","
    s = s[:-1]
    file.write(s)
    file.write("\n")

    if prevStr == s:
        break
    prevStr = s
    #done

    for j in range(0, len(newCents)):
    #for j in newCents.values():
        newCents[j] = []
        #j = []k

    '''
    for i in x:
        plt.scatter(i[0], i[1], s = 10, color='red')
    for i in centrioads:
        plt.scatter(i[0], i[1], s=30, color='blue')

    plt.show()
'''

res = [[ i for i, j in total_loss ],
       [ j for i, j in total_loss ]]
plt.plot(res[0],res[1])
plt.show()

file.close()