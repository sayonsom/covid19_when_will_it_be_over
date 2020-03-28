import pandas as pd
import re
import json

pop_dict = {}
df = pd.read_csv('country_population.csv', index_col=None)
df_health = pd.read_csv('health_care_rank.csv', index_col=None)
pattern_to_delete = r'\[\w+\s+\d\]'
for idx, row in df.iterrows():
    country = row['Country']
    tmp = df_health[df_health['Country'] == country]
    if len(tmp) == 1:
        rank = tmp['Rank'].values[0]
    else:
        rank = "depends heavily on foreign aid"
    country = re.sub(r'\[\w+\s+\d\]', '', country)
    pop_dict[country] = {}
    pop_dict[country]['area'] = row['Area']
    pop_dict[country]['population'] = row['Population']
    pop_dict[country]['pop_density'] = row['Density']
    pop_dict[country]['rank'] = str(rank)
import pprint
pprint.pprint(pop_dict)
with open("country_population.json", "w") as fp:
    json.dump(pop_dict,fp)

