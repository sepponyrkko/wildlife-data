#!/usr/bin/env python3
"""
A small utility to compose analysis groups out of the taxonomical
data.

Created on Sun Nov 19 10:33:39 2023
@author: seppo
"""

import scipy as sp
import numpy as np
import pandas as pd
from collections import defaultdict


# In[read in onto-relations.tsv]
rel_df = pd.read_csv("onto-relations.tsv",sep="\t").drop_duplicates()





# In[define the taxonomic levels and upward direction]
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

# Taxonomic rank orders checked from https://en.wikipedia.org/wiki/Taxonomic_rank
# https://en.wikipedia.org/wiki/Order_(biology)#Hierarchy_of_ranks
# and https://en.wikipedia.org/wiki/Tribe_(biology)
#  Sun Nov 19 2023


# In[]

import re

glabels = [l[1:] for (l,r) in grouping_order]
glinks = [r for (l,r) in grouping_order if len(r)>0]

# Flip relation types downwards to upwards from genus...kingdom
# 
# for instance Muridae#family	wo:genus	Apodemus#genus
#   >>>   Apodemus#genus Muridae#family	wo:family	

links = pd.DataFrame(rel_df[rel_df.p.isin(glinks)],columns=['o','p','s'])
links['p'] = links.s.str.replace('^.*#','wo:',regex=True)
links.rename(columns={'s':'o','o':'s'},inplace=True)
rel_df = pd.concat([links,rel_df])

#
# Now populate the steps from species up to kingdom
#

kingdom_steps = []
species_list = rel_df.s[rel_df.s.str.match(r"^.*\#species") |
                        rel_df.s.str.match(r"^.*\#genus")].unique()

for st in species_list:

    x = rel_df[(rel_df.s==st) & (rel_df.p.isin(glinks))]
    y = ([x[x.p==g].o.max() for g in glinks if g in list(x.p)])
    for yd in range(1,len(y)):
        y_df = pd.DataFrame(zip(y[:-yd],
                                [re.sub(r'^.*\#','wo:',yy) 
                                        for yy in y[yd:]],
                                y[yd:]),
                            columns=['s','p','o'])
        if (len(y_df)):
            kingdom_steps.append(y_df)

kingdom_steps_df = pd.concat(kingdom_steps)

rel_df = pd.concat([rel_df,kingdom_steps_df]).drop_duplicates().reset_index(drop=True)

# In[fill in rank levels upwards]

taxons = rel_df[
    rel_df.s.str.replace('.*\#','',regex=True).isin(glabels) &
    (rel_df.p=='rdf:type')].s.unique()

g_df = pd.DataFrame(zip(taxons,
                        taxons,
                        [re.sub('^.*\#','',t) for t in taxons]
                        ),columns=['taxon','group','level'])
g_df.set_index('taxon',inplace=True)
g_df['fixed']=False


# In[iterate taxon rank levels]

TARGET_SIZE = 25

for tl_id in range(len(glabels))[1:]:
    
    tl_name = glabels[tl_id]
    tl_coll = glabels[0:tl_id]
    tl_link=glinks[tl_id-1]
    
    print("Grouping %s" % tl_name)
    
    agg_df = g_df[g_df.level.isin(tl_coll) & (g_df.fixed==False)]
    agg_rels = rel_df[rel_df.p==tl_link]
    
    # collect the next grouping data
    
    df=pd.DataFrame(
        pd.merge(agg_df,agg_rels,left_index=True,right_on=['s']),
                    columns=['s','o'])
    
    # lift the taxon groups 
    
    lift_dict = {t:g for t,g in zip(df.s,df.o)}
    newgroups = [(lift_dict[i] if i in lift_dict else g) for i,g in zip (g_df.index, g_df.group)]
        
    g_df.group=newgroups    
    
    g_df.level=g_df.group.str.replace('^.*\#','',regex=True)    
    
    gsize = g_df.group.value_counts()
    
    g_df['fixed']=g_df.group.isin(gsize.index[gsize>TARGET_SIZE])


# In[Collect the rest as one group]

gsize = g_df.group.value_counts()
g_df['fixed']=g_df.group.isin(gsize.index[gsize>5])
out_df=pd.DataFrame(g_df)
out_df.group = [g if f else 'Life#life' for g,f in zip(out_df.group,out_df.fixed)]
out_df.level=out_df.group.str.replace('^.*\#','',regex=True)    
out_df['fixed']=True
out_df.drop(columns=['fixed'],inplace=True)
out_df.to_csv("analysis_groups.tsv",sep="\t")

# In[write groups.txt]

with open("groups.txt","w") as the_file:
    for g,gdf in out_df.groupby('group'):
        the_file.writelines('%s\n-------\n  '%g)
        the_file.writelines('\n  '.join(gdf.index))
        the_file.writelines('\n\n')
     
        
