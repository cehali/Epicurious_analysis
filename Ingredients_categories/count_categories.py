import numpy as np
from collections import Counter
import openpyxl
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

columns_index = ['G']
categories = []
quantity_categories = []

recipes = np.load('recipes.npy')

recipes_categories = []
for rec in recipes:
    for k, cat in enumerate(rec.get('categories')):
        recipes_categories.append(cat)

num_categories = Counter(recipes_categories)

wb = openpyxl.load_workbook('epicurious_categories.xlsx')
first_sheet = wb.get_sheet_names()[0]
worksheet = wb.get_sheet_by_name(first_sheet)

for i, column in enumerate(columns_index):
    for row in range(2, len(worksheet[column])):
        cell_name = "{}{}".format(column, row)
        if worksheet[cell_name].value is not None:
            categories.append(str(worksheet[cell_name].value))

for cat in categories:
    print cat
    quantity_categories.append(num_categories[cat])

print sum(quantity_categories)
print len(recipes)