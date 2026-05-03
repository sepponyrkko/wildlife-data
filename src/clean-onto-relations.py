#!/usr/bin/env python3
"""
A small utility to compose onto-relations.txt for the ML experiment.

(Run FIRST the ontology importer "collect-onto-rels.py")

Created on Sun Nov 19 10:33:39 2023
@author: seppo
"""

import scipy as sp
import numpy as np
import pandas as pd
from collections import defaultdict


# In[read in onto-relations.tsv]
rel_df = pd.read_csv("onto-relations.tsv",sep="\t").drop_duplicates()


# In[taxonomic levels]
grouping_order = [('#species',''),
                 ('#genus','wo:genus'),
                 ('#tribe','wo:tribe'),
                 ('#family','wo:family'),
                 ('#superfamily','wo:superfamily'),
                 ('#infraorder','wo:infraorder'),
                 ('#suborder','wo:suborder'),
                 ('#order','wo:order'),
                 ('#superorder','wo:superorder'),
                 ('#class','wo:class'),
                 ('#superclass','wo:superclass'),
                 ('#phylum','wo:phylum'),
                 ('#kingdom','wo:kingdom'),
                 ]


func_relgrp = {'wo:livesIn':'location', 'wo:growsIn':'location', 'wo:adaptation':'adaptation'}

x = rel_df[rel_df.p.isin(func_relgrp.keys())]
out_df = pd.DataFrame(zip([func_relgrp[p] for p in x.p], x.p, x.s, x.o),columns=['relation','p','s','o'])
# In[test]

out_df.to_csv("onto-relations.txt",sep="\t",index=False)
print("onto-relations.txt written, data shape" , out_df.shape)

# In[scan the taxonomic levels only upwards direction]

taxon_level = {y:x for (x,y) in enumerate([z for (z,w) in grouping_order])}
taxon_pred = [y for (x,y) in grouping_order if len(y)]
t_df = pd.DataFrame(rel_df[rel_df.p.isin(taxon_pred)])

t_df['s_level'] = [taxon_level[c] if c in taxon_level else -1 for c in t_df.s.str.replace(r'.*#','#',regex=True)]
t_df['o_level'] = [taxon_level[c] if c in taxon_level else 99 for c in t_df.o.str.replace(r'.*#','#',regex=True)]

out2_df = pd.DataFrame(t_df[t_df.s_level<t_df.o_level], columns=out_df.columns)
out2_df['relation']='taxongroup'

out_df = pd.concat([out_df,out2_df]).drop_duplicates()


# In[now for real]

out_df.to_csv("onto-relations.txt",sep="\t",index=False)
print("onto-relations.txt written, data shape" , out_df.shape)

