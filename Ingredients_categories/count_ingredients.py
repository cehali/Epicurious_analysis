import difflib
from ingredient_parser.en import parse
import json
from collections import Counter
import numpy as np
import openpyxl
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

ingredients = []
recipes = np.load('recipes.npy')
recipes_ingredients = []
ingredients_excel = []
recipes_titles = []
ingredients_binary = []
ingredients_for_KMeans = []
values_recipes = []
inertia = []
columns_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

with open('ingredients.json') as json_data:
    ingredients_full = json.load(json_data)

for ingr in ingredients_full:
    ingredients.append(ingr.get('name'))

wb = openpyxl.load_workbook('Categories_ingredients.xlsx')
first_sheet = wb.get_sheet_names()[0]
worksheet = wb.get_sheet_by_name(first_sheet)

for i, column in enumerate(columns_index):
    for row in range(2, len(worksheet[column])):
        cell_name = "{}{}".format(column, row)
        if worksheet[cell_name].value is not None:
            ingredients_excel.append(str(worksheet[cell_name].value))

try:

    ingr_occurence = Counter(ingredients)
    start = 0
    for rec in recipes:
        recipe_ingredients = ingredients[start:(start+len(rec.get('ingredients')))]
        for i, rc_in in enumerate(recipe_ingredients):
            if (rc_in in [w for w in ingredients if ingr_occurence[w] == 1]) or (rc_in in [w for w in ingredients if ingr_occurence[w] == 2]):
                nearest_ingredient = difflib.get_close_matches(rc_in, ingredients_excel, cutoff=0.1)
                if len(nearest_ingredient) > 0:
                    recipe_ingredients[i] = nearest_ingredient[0]
        start += len(rec.get('ingredients'))
        recipes_ingredients.append(str(recipe_ingredients))
        recipes_titles.append(rec.get('title'))

    np.save('recipes_only_ingredients', recipes_ingredients)

    ingr_occurence1 = Counter(ingredients)

    for nr1 in range(1, 20000):
        for io in ingr_occurence1.most_common(10*nr1):
            ingredients_for_KMeans.append(io[0])

        int_ingredients = list(range(1, len(ingredients_for_KMeans) + 1))

        index = list(range(0, len(recipes_titles)))

        column_headers = ['ingredient_' + str(i) for i in int_ingredients]

        for rp in recipes_ingredients:
            value_recipe = [0 for i in int_ingredients]
            for ip in rp:
                for k, ifk in enumerate(ingredients_for_KMeans):
                    if ip == ifk:
                        value_recipe[k] = int_ingredients[k]
                        break
            values_recipes.append(value_recipe)

        for nr2 in range(2, 30):
            clf = KMeans(n_clusters=nr2)
            clf.fit(values_recipes)

            print 'Inertia for ingredients number: ', nr1, ' and cluster number ', nr2, ' : ', clf.inertia_
            inertia_for_save = 'ingredients number: ' + str(nr1), ' cluster number: ' + str(nr2) + \
                               ' - ' + str(clf.inertia_)
            inertia.append(inertia_for_save)

    np.save('inertia_recipes', inertia)

    print 'Done1'

except:
    start = 0

    with open('ingredients.json') as json_data:
        ingredients_full = json.load(json_data)

    for ingr in ingredients_full:
        ingredients.append(ingr.get('name'))

    for rec in recipes:
        recipe_ingredients = ingredients[start:(start + len(rec.get('ingredients')))]
        # for i, rc_in in enumerate(recipe_ingredients):
        # if (rc_in in [w for w in ingredients if ingr_occurence[w] == 1]) or (rc_in in [w for w in ingredients if ingr_occurence[w] == 2]):
        # nearest_ingredient = difflib.get_close_matches(rc_in, ingredients_excel, cutoff=0.1)
        # if len(nearest_ingredient) > 0:
        #    recipe_ingredients[i] = nearest_ingredient[0]
        start += len(rec.get('ingredients'))
        recipes_ingredients.append(recipe_ingredients)
        recipes_titles.append(rec.get('title'))

    ingr_occurence = Counter(ingredients)

    for nr1 in range(1, 20000):
        for io in ingr_occurence.most_common(10*nr1):
            ingredients_for_KMeans.append(io[0])

        int_ingredients = list(range(1, len(ingredients_for_KMeans) + 1))

        index = list(range(0, len(recipes_titles)))

        column_headers = ['ingredient_' + str(i) for i in int_ingredients]

        for rp in recipes_ingredients:
            value_recipe = [0 for i in int_ingredients]
            for ip in rp:
                for k, ifk in enumerate(ingredients_for_KMeans):
                    if ip == ifk:
                        value_recipe[k] = int_ingredients[k]
                        break
            values_recipes.append(value_recipe)

        for nr2 in range(2, 30):
            clf = KMeans(n_clusters=nr2)
            clf.fit(values_recipes)

            print 'Inertia for ingredients nubmer: ', nr1, ' and cluster number ', nr2, ' : ', clf.inertia_
            inertia_for_save = 'ingredients nubmer: ' + str(nr1), ' cluster number: ' + str(nr2) + \
                               ' - ' + str(clf.inertia_)
            inertia.append(inertia_for_save)

        np.save('inertia_recipes', inertia)

    print 'Done2'

#For speed of computation, only run on a subset
'''n = 2000
x_data = recipes_ingredients[:n]
recipes_titles_subset = recipes_titles[:n]

ingredients_distinct = list(set(ingredients))

for mat1 in x_data:
    ingredient_binary = [0] * len(ingredients_distinct)
    for mat2 in mat1:
        for r in range(0, len(ingredients_distinct)):
            if mat2 == ingredients_distinct[r]:
                ingredient_binary[r] = 1
    ingredients_binary.append(ingredient_binary)

similarity = cosine_similarity(ingredients_binary)

# perform t-SNE embedding
vis_data = TSNE(n_components=2, metric='cosine').fit_transform(similarity)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y)
#for i, txt in enumerate(recipes_titles_subset):
#    plt.annotate(txt, (vis_x[i], vis_y[i]))
plt.show()'''