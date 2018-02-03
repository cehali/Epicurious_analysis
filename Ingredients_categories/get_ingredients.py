import difflib
from ingredient_parser.en import parse
import numpy as np
import openpyxl
from collections import defaultdict
from openpyxl.utils import coordinate_from_string

recipes = np.load('recipes.npy')
ingredients = []
nearest_ingredients = []
columns_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

wb = openpyxl.load_workbook('Categories_ingredients.xlsx')
first_sheet = wb.get_sheet_names()[0]
worksheet = wb.get_sheet_by_name(first_sheet)

for i, column in enumerate(columns_index):
    for row in range(2, len(worksheet[column])):
        cell_name = "{}{}".format(column, row)
        if worksheet[cell_name].value is not None:
            ingredients.append(str(worksheet[cell_name].value))

print len(ingredients)

rec = recipes[0]
print rec.get('ingredients')
for ingr_reci in rec.get('ingredients'):
    print ingr_reci
    temp = parse(ingr_reci).get('name')
    nearest_ingredient = difflib.get_close_matches(temp, ingredients, cutoff=0.1)
    print nearest_ingredient
    nearest_ingredients.append(nearest_ingredient)

