# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:44:45 2020

@author: patel
"""

import os
os.system('cmd /k "pip install mlxtend"')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules 


data=pd.read_csv("C:\\Users\\patel\\Downloads\\book.csv")
colnames=data.columns
data.head()
plt.plot(data)

frequent_itemsets = apriori(data, min_support=0.07, use_colnames=True) 
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()


    
###
self=rules['antecedents']
plt.bar(x=self,height=rules['support'],color='rgbkymc')
plt.xlabel("antecedent")
plt.ylabel("support")
###

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)
rules_no_redudancy.dtypes

plt.bar(x=rules_no_redudancy.index,height=rules_no_redudancy.lift,color='c')

plt.xlabel("antecedent")

plt.ylabel("lift")



