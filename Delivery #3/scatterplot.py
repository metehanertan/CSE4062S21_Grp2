import pandas as pd

keys = ['NN', 'JJ', 'IN', 'CD', 'RB', 'MD', 'PRP', 'JJS', 'VBD', 'VB', 'VBG', 'VBP', 'NNS', 'DT', 'RBR', 'VBN', 'VBZ', 'JJR', 'RBS', 'CC', 'RP', 'WP', 'WDT', 'FW', 'UH', 'TO', 'WRB', '$', 'NNP', 'PDT', 'EX', 'WP$', "''"]
values = [4938, 3979, 170, 175, 836, 13, 5, 33, 791, 467, 483, 884, 1221, 17, 45, 471, 361, 62, 3, 19, 24, 6, 2, 60, 2, 1, 6, 2, 28, 1, 2, 1, 1]
dictionary = {}

for key, value in zip(keys, values):
    dictionary[key] = value

print(dictionary)

df = pd.DataFrame(values, columns=keys)
chart = pd.plotting.scatter_matrix(df, alpha=0.2)