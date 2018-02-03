import openpyxl
import numpy as np
from collections import Counter
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import sys
import csv
from wordcloud import WordCloud
from sklearn.metrics.pairwise import pairwise_distances


reload(sys)
sys.setdefaultencoding('utf-8')

recipes = np.load('recipes.npy')
recipes_ingredients = []
recipes_titles = []
ingredients_all_list = []
recipes_categories = []
categories = []
recipes_dietery_consideration = []
columns_index = ['E']

start = 0
nr_clusters = 10
nr_ingredients = 15

top_ingredients = ['sugar', 'salt', 'olive oil', 'garlic', 'unsalted butter']


ingredients = np.load('recipes_only_ingredients.npy')
for ing in ingredients:
    ing = ing.replace('[', '').replace(']', '').replace(" u'", '').replace(" '", '').replace("u'", '')\
        .replace('uEquipment: ', '').replace("'", '')
    ing = ing.split(',')
    recipes_ingredients.append(ing)

wb = openpyxl.load_workbook('epicurious_categories.xlsx')
first_sheet = wb.get_sheet_names()[0]
worksheet = wb.get_sheet_by_name(first_sheet)

for i, column in enumerate(columns_index):
    for row in range(2, len(worksheet[column])):
        cell_name = "{}{}".format(column, row)
        if worksheet[cell_name].value is not None:
            categories.append(str(worksheet[cell_name].value))

for ingr1 in recipes_ingredients:
    for ingr2 in ingr1:
        ingredients_all_list.append(ingr2)

for rec in recipes:
    recipes_titles.append(rec.get('title'))
    recipes_categories.append(rec.get('categories'))

for rcc in recipes_categories:
    recipes_dietery_consideration.append([i for i in rcc if i in categories])

ingr_occurence = Counter(ingredients_all_list)

values_recipes = []
ingredients_for_KMeans = []
for io in ingr_occurence.most_common(nr_ingredients):
    ingredients_for_KMeans.append(io[0])

int_ingredients = list(range(1, len(ingredients_for_KMeans) + 1))
index = list(range(0, len(recipes_titles)))
column_headers = ['ingredient_' + str(i) for i in ingredients_for_KMeans]

for rp in recipes_ingredients:
    value_recipe = [0 for i in ingredients_for_KMeans]
    for ip in rp:
        for k, ifk in enumerate(ingredients_for_KMeans):
            if ip == ifk:
                value_recipe[k] = 1
                break
    values_recipes.append(value_recipe)

values_ingredients_recipes = pd.DataFrame(data=values_recipes, index=index, columns=column_headers,
                                          dtype=float)

# values_ingredients_recipes.to_csv('binary_matrix_ingredients.csv', index=False)

'''values_ingredients_recipes = pd.read_csv('binary_matrix_ingredients.csv')


#X = values_ingredients_recipes.head(10000)
X = np.array(values_ingredients_recipes)

similarity_matrix = pairwise_distances(X, metric='jaccard')


scl = SpectralClustering(n_clusters=8, affinity='precomputed')
scl.fit(similarity_matrix)

print [x for x in scl.labels_]'''


#clf = KMeans(n_clusters=nr_clusters)
#clf.fit(X)
#print clf.labels_
#print 'Inertia for ingredients number: ', nr_ingredients, ' and cluster number ', nr_clusters, ' : ', clf.inertia_

#db = DBSCAN(eps=5, min_samples=10).fit(X)

#print [x for x in db.labels_]

'''Z = linkage(X, 'ward')
c, coph_dists = cophenet(Z, pdist(X))

print c

last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print "clusters:", k'''


X = np.array(values_ingredients_recipes)

clf = KMeans(n_clusters=nr_clusters)
clf.fit(X)

for label_nr in range(0, nr_clusters):
    recipes_with_labels = []
    for n, recipe in enumerate(recipes):
        if clf.labels_[n] == label_nr:
            temp = {}
            temp['title'] = recipe.get('title')
            temp['ingredients'] = recipes_ingredients[n]
            temp['label'] = clf.labels_[n]
            temp['categories'] = recipe.get('categories')
            recipes_with_labels.append(temp)
    
    #with open("recipes_with_labels_" + str(label_nr) + ".csv", "wb") as myfile:
    #    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #    wr.writerow(recipes_with_labels)
    
    to_plot_categories = []
    to_plot_ingredients = []
    for tp in recipes_with_labels:
        for t in tp.get('categories'):
            to_plot_categories.append(t)
        for p in tp.get('ingredients'):
            if p not in ingredients_for_KMeans:
                to_plot_ingredients.append(p)
    
    # Generate a word cloud image
    wordcloud = WordCloud(width=1600, height=800).generate(' '.join(to_plot_ingredients))
    plt.figure( figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    savefig('ingredients_' + str(nr_ingredients) + '_cloud_' + str(label_nr) + '.png', bbox_inches='tight')

    wordcloud = WordCloud(width=1600, height=800).generate(' '.join(to_plot_categories))
    plt.figure( figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    savefig('categories_cloud_' + str(label_nr) + '.png', bbox_inches='tight')

    plt.close('all')

# plot graph
'''values = Counter(to_plot).values()
labels = Counter(to_plot).keys()
indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()'''