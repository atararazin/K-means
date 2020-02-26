import numpy as np
import numpy
from math import sqrt
import matplotlib.pyplot as plt
import random

NUM_ITERATIONS = 20

x = random.sample(range(-1000,1000), 500)
y = random.sample(range(-1000,1000), 500)
l1 = list(zip(x,y))
points = numpy.array(l1)

centriods = np.array([[0., 0.], [1., -1.], [2., -2.], [3., -3.], [4., -4.], [5., -5.], [6., -6.], [7., -7.], [8., -8.], [9., -9.]])
newCents = {k: [] for k in range(len(centriods))}
total_loss = []

def k_means():
    for iter in range(0, NUM_ITERATIONS):
        iter_loss = 0
        for i in points:
            dists = []
            for cent in centriods:
                dists.append(calc_euclid_dist(i, cent))

            min_dist = min(dists)
            index = dists.index(min_dist)
            newCents.get(index).append(i)
            iter_loss += min_dist
        total_loss.append((iter, iter_loss / len(points)))

        update_cents(centriods)
        reset_newCents()
        draw_points_in_plane()

    draw_loss_graph()

def calc_euclid_dist(p1 ,p2):
    return float(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def sumPoints(list):
    if (len(list) == 0):
        return
    n = len(list)
    listX, listY = zip(*list)
    newX = (sum(listX) / n).round()
    newY = (sum(listY) / n).round()
    return newX, newY

def update_cents(centrioads):
    for i, vals in enumerate(newCents.values()):
        centrioads[i] = sumPoints(vals)

def reset_newCents():
    for l in newCents.values():
        del l[:]

def draw_points_in_plane():
    for i in points:
        plt.scatter(i[0], i[1], s=10, color='red')
    for i in centriods:
        plt.scatter(i[0], i[1], s=30, color='blue')
    plt.show()


def draw_loss_graph():
    x, y = zip(*total_loss)
    plt.title("Loss Graph:")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(x, y)
    plt.show()

k_means()