import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

recipes = np.load('recipes.npy')
ingredients = []

for rec in recipes:
    ingredients.append(rec.get('ingredients'))


with open("ingredients_epicurious.txt", "w") as text_file:
    for ingr1 in ingredients:
        for ingr2 in ingr1:
            text_file.write(str(ingr2) + '\n')
