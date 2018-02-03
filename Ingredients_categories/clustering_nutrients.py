import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import openpyxl
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift, SpectralClustering, MiniBatchKMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import sys
import csv

from sklearn.manifold import TSNE
from wordcloud import WordCloud

import hypertools as hyp

reload(sys)
sys.setdefaultencoding('utf-8')

recipes = np.load('recipes.npy')

recipes_nutrients = []
nutrients_order = ['calories', 'fat', 'protein', 'sodium']
categories = {'Appetizer', 'Breakfast', 'Brunch', 'Buffet', 'Dessert', 'Dinner', 'Lunch', 'Side'}

recipes_final = []
for r in recipes:
    for e in r['categories']:
        if e in categories:
            recipes_final.append(r)

recipes_names = []
recipes_categories = []
for rec in recipes_final:
    temp = []
    temp.append(rec['calories'])
    temp.append(rec['fat'])
    temp.append(rec['protein'])
    temp.append(rec['sodium'])
    if None not in temp:
        recipes_nutrients.append(np.array(temp))
        recipes_names.append(rec['title'])

X = np.array(recipes_nutrients)
X = preprocessing.scale(X)

print type(recipes_nutrients[0])
hyp.plot(recipes_nutrients, '.', n_clusters=10)


#bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ms.fit(X)

#labels = ms.labels_
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)

'''for nc in range(n_clusters_):
    for l, lab in enumerate(labels):
        if lab == nc:
            print 'Cluster ', nc, ': ', recipes_names[l]'''


#clf = KMeans(n_clusters=5)
#clf.fit(X)



#spectral = SpectralClustering(n_clusters=5)
#spectral.fit(X)

#labels = ms.labels_
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)

#for nc in range(n_clusters_):
#    for l, lab in enumerate(labels):
#        if lab == nc:
#            print 'Cluster ', nc, ': ', recipes_names[l], [v for v in recipes_final[l]['categories'] if v in categories]

#two_means = MiniBatchKMeans(n_clusters=4)
#two_means.fit(X)

#print [x for x in two_means.labels_]


#X_embedded = TSNE(n_components=3).fit_transform(X)

#plt.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=labels, cmap=plt.cm.get_cmap("jet", 10))

#plt.show()

'''inertia = []
for nr2 in range(2, 51):
    X = preprocessing.scale(X)
    clf = KMeans(n_clusters=nr2)
    clf.fit(X)

    print 'Cluster number ', nr2, ' : ', clf.inertia_
    inertia_for_save = float(clf.inertia_)
    inertia.append(inertia_for_save)

with open('recipes_inertia_nutrient', 'wb') as f:
    np.savetxt(f, inertia, fmt='%.5f')

print 'Done1'
'''

'''
dpgmm = mixture.BayesianGaussianMixture(n_components=5).fit(X)
clusters = dpgmm.predict(X)

for x in range(0, len(clusters)):
    print clusters[x]'''