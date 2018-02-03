import openpyxl
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
import sys

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
nr_clusters = 20
nr_ingredients = 6

ingredients = np.load('recipes_only_ingredients.npy')
for ing in ingredients:
    ing = ing.replace('[', '').replace(']', '').replace(" u'", '').replace("'", '').replace(" '", '').replace("u'", '')
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

print ingr_occurence