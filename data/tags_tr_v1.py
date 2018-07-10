import pandas as pd

df = pd.read_csv('tags.csv')

names = df['名称']
categories = df['类型']
all_dict = {}
for i in range(len(names)):
    category_list = [] 
    if names[i] in all_dict.keys():
        all_dict[names[i]].append(categories[i])
        category_list = all_dict[names[i]]
        all_dict[names[i]] = list(set(category_list))
    else:
        category_list.append(categories[i])
        all_dict[names[i]] = category_list

df_key = pd.read_csv('tourist_cndbpedia5.csv')

keywords = list(df_key['title'])

final_dict = {}
for keyword in keywords:
    if keyword not in all_dict.keys():
        all_dict[keyword] = ['普通']

#print(all_dict)
save_file = open('tags_dict.txt', 'w')
save_file.write(str(all_dict))
save_file.close()

