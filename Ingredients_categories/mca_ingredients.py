import openpyxl
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import sys
import csv
from wordcloud import WordCloud

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
nr_clusters = 12
nr_ingredients = 5

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

print values_ingredients_recipes