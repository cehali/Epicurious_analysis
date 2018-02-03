import json
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing


recipes = np.load('recipes.npy')
recipes_ingredients = []
recipes_titles = []
ingredients_all_list = []
columns_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

start = 0

ingredients = np.load('recipes_only_ingredients.npy')
for ing in ingredients:
    ing = ing.replace('[', '').replace(']', '').replace(" u'", '').replace("'", '').replace(" '", '').replace("u'", '')
    ing = ing.split(',')
    recipes_ingredients.append(ing)

for ingr1 in recipes_ingredients:
    for ingr2 in ingr1:
        ingredients_all_list.append(ingr2)

for rec in recipes:
    recipes_titles.append(rec.get('title'))

ingr_occurence = Counter(ingredients_all_list)


def get_kmeans_inertia(ingredients_concluded):
    inertia = []
    X = np.array(ingredients_concluded)
    X = preprocessing.scale(X)
    for nr2 in range(2, 51):
        clf = KMeans(n_clusters=nr2)
        clf.fit(X)

        print 'Inertia for ingredients number: ', nr1, ' and cluster number ', nr2, ' : ', clf.inertia_
        inertia.append(clf.inertia_)

    return inertia


for nr1 in range(2, 50):
    values_recipes = []
    ingredients_for_KMeans = []
    for io in ingr_occurence.most_common(nr1):
        ingredients_for_KMeans.append(io[0])

    int_ingredients = list(range(1, len(ingredients_for_KMeans) + 1))
    index = list(range(0, len(recipes_titles)))
    column_headers = ['ingredient_' + str(i) for i in int_ingredients]

    for rp in recipes_ingredients:
        value_recipe = [0 for i in int_ingredients]
        for ip in rp:
            for k, ifk in enumerate(ingredients_for_KMeans):
                if ip == ifk:
                    value_recipe[k] = 1
                    break
        values_recipes.append(value_recipe)

    values_ingredients_recipes = pd.DataFrame(data=values_recipes, index=index, columns=column_headers,
                                              dtype=float)

    recipes_inertia = get_kmeans_inertia(values_recipes)

    with open('recipes_inertia_' + str(nr1), 'wb') as f:
        np.savetxt(f, recipes_inertia, fmt='%.5f')


print 'Done2'
