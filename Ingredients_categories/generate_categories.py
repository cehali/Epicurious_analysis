import numpy as np
import openpyxl
from collections import defaultdict
from openpyxl.utils import coordinate_from_string

recipes = np.load('recipes.npy')

diary = []
vegetables = []
fruits = []
spices = []
meats = []
fish = []
baking_grains = []
oils = []
seafood = []
added_sweeteners = []
seasonings = []
nuts = []
condiments = []
desserts_snacks = []
beverages = []
soup = []
dairy_alternatives = []
legumes = []
sauces = []
alcohol = []

categories = [diary, vegetables, fruits, spices, meats, fish, baking_grains, oils, seafood, added_sweeteners,
              seasonings, nuts, condiments, desserts_snacks, beverages, soup, dairy_alternatives, legumes,
              sauces, alcohol]

categories_str = ['diary', 'vegetables', 'fruits', 'spices', 'meats', 'fish', 'baking_grains', 'oils', 'seafood',
                  'added_sweeteners', 'seasonings', 'nuts', 'condiments', 'desserts_snacks', 'beverages', 'soup',
                  'dairy_alternatives', 'legumes', 'sauces', 'alcohol']

columns_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


wb = openpyxl.load_workbook('Categories_ingredients.xlsx')
first_sheet = wb.get_sheet_names()[0]
worksheet = wb.get_sheet_by_name(first_sheet)

for i, column in enumerate(columns_index):
    for row in range(2, len(worksheet[column])):
        cell_name = "{}{}".format(column, row)
        if worksheet[cell_name].value is not None:
            categories[i].append(str(worksheet[cell_name].value))

recipes_categories = {}
recipes_ingredients_categories = dict.fromkeys(categories_str, [])
ingredients_categories = []
for k, rec in enumerate(recipes):
    for ingr in rec.get('ingredients'):
        for l, cat in enumerate(categories):
            for ingr_cat in cat:
                if ingr_cat in ingr:
                    ingredients_categories.append(ingr_cat)
            for temp in ingredients_categories:
                recipes_ingredients_categories[categories_str[l]].append(temp)
            ingredients_categories = []
    recipes_categories[k] = recipes_ingredients_categories
    recipes_ingredients_categories = dict.fromkeys(categories_str, [])

print recipes[0].get('ingredients')
print recipes_categories[0]