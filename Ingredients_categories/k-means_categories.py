import string
from collections import Counter

import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import manifold
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import hypertools as hyp

recipes = pd.read_csv('/home/cezary/Dataset-recpies+ratings/epi_r_Epicurious.csv')

'''recipes = recipes.drop('rating', 1)
recipes = recipes.drop('calories', 1)
recipes = recipes.drop('protein', 1)
recipes = recipes.drop('fat', 1)
recipes = recipes.drop('sodium', 1)
recipes = recipes.drop('title', 1)'''

nutrients = pd.concat([recipes['calories'], recipes['protein'], recipes['fat'], recipes['sodium']], axis=1)


hyp.plot(nutrients, '.', n_clusters=10)

'''categories = []
recipes = np.load('recipes.npy')

for rec in recipes:
    for r in rec.get('categories'):
        categories.append(r)

print Counter(categories)'''

'''recipes = recipes[0:1000].astype('bool')
recipes.drop_duplicates()

S = (pairwise_distances(recipes, metric="jaccard"))
simulations = 7

cmap = plt.get_cmap('Set1')
colors = [cmap(i) for i in np.linspace(0,1, simulations)]

markers = []
labels = [str(n+1) for n in range(simulations)]

dt = [('len', float)]
A = S
A = A.view(dt)

G = nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=50)

plt.tight_layout()
plt.axis('equal')
plt.show()'''


