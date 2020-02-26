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
    newX = (sum(listX) / n).round()
    newY = (sum(listY) / n).round()
    #return (sum(listX) / n, sum(listY) / n)
    return newX, newY

prevStr = ""

newCents = {k: [] for k in range(len(centrioads))}

total_loss = []

for iter in range(0, NUM_ITERATIONS):
    iter_loss = 0
    for i in x:
        dists = []
        for cent in centrioads:
            dists.append(calc_euclid_dist(i,cent))

        min_dist = min(dists)
        index = dists.index(min_dist)
        newCents.get(index).append(i)
        iter_loss += min_dist
    print("iteration:", iter)
    print("loss:", iter_loss / len(x), "\n")
    total_loss.append((iter, iter_loss / len(x)))

    for i, vals in enumerate(newCents.values()):
        centrioads[i] = sumPoints(vals)


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

    for l in newCents.values():
        del l[:]

    '''
    for i in x:
        plt.scatter(i[0], i[1], s = 10, color='red')
    for i in centrioads:
        plt.scatter(i[0], i[1], s=30, color='blue')

    plt.show()
'''

x,y = zip(*total_loss)
#res = [[ i for i, j in total_loss ],
#       [ j for i, j in total_loss ]]
#plt.plot(res[0],res[1])
plt.title("Loss Graph:")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(x,y)
plt.show()

file.close()