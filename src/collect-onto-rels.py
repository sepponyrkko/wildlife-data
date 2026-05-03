#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:15:13 2023

Short utility using unix shell pipe to compose a full TSV corpus data

This writes out onto-relations.txt for further processing

@author: seppo
"""

# In[Load the corpus data]:

import pandas as pd
import subprocess
from io import StringIO
import re

datastr = (str(subprocess.check_output(
        ['cat ../../wildlife-corpus/ontology-data/rdf_tsv/*.tsv|sort|uniq']
        , shell=True),'utf-8'))
rawdata = pd.read_csv(StringIO(datastr),
                      sep='\t',
                      header=None,
                      names=['s','p','o'])



# In[from raw to rdf]:

def cleanname(cleanx):
    cleanx = re.sub(r'[\'\"]+','',cleanx)
    cleanx = cleanx.replace("http://www.bbc.co.uk/nature/","n:")

    if ('/' in cleanx) and (':' in cleanx):
        cleanx = re.sub(r'[\(\)\'\",;]+','',cleanx)
        cleanx = re.sub(r'\/+\#','#',cleanx) 
        cleanx = re.sub(r'\#adaptations','#adaptation',cleanx) 
        cleanx = re.sub(r'\#habitats','#habitat',cleanx) 
        cleanx = re.sub(r'\#ecozones','#ecozone',cleanx) 
        cleanx = re.sub(r'.*\/','',cleanx) 
        cleanx = re.sub(r'[^A-Za-z0-9\-\:\#\. ]+','_',cleanx)
    return cleanx
    

# drop rows on subjects without names before anchor

rdfdata = pd.DataFrame(rawdata[~rawdata.s.str.match(r".*\/\#.*")])    

# also discard rows with NA in 's' or 'o'

rdfdata=pd.DataFrame(rdfdata[~(rdfdata.o.isna() | rdfdata.s.isna())])

# sanitize the URI names for our experiment

rdfdata.s = rdfdata.s.apply(cleanname)
rdfdata.o = rdfdata.o.apply(cleanname)

# keep only the Knowledge base related definitions in this dataset

rdfdata = rdfdata[(rdfdata.s.str.match(r'.*\#.+')) | 
                  (rdfdata.s.str.match(r'\w+:'))]


rdfdata.drop_duplicates(inplace=True)
rdfdata.reset_index(drop=True,inplace=True)


rdfdata=pd.DataFrame(rdfdata[~((rdfdata.o.str.len()<3) | (rdfdata.s.str.len()<3))])


# In[now some rewriting in the anchor URI format]:

taxon_ids = rdfdata[rdfdata.s.isin(rdfdata[rdfdata.p=='wo:name'].s) &
                    (rdfdata.p=='rdf:type')].o.unique()    

taxon_types = rdfdata[(rdfdata.p=='rdf:type') & 
                (rdfdata.o.isin(taxon_ids))].drop_duplicates()


conceptid={c:re.sub(r"\#.*","#",c)+t[3:].lower() for (c,t) in zip(taxon_types.s,taxon_types.o)}

rdfdata.s = [conceptid[x] if x in conceptid else x for x in rdfdata.s]

rdfdata.o = [conceptid[x] if x in conceptid else x for x in rdfdata.o]



# In[write clean rdf data]:

  
rdfdata.to_csv("onto-rdfdata.tsv",sep="\t",index=False)



# In[filter usable predicate rows]:
    
wo_refs = pd.DataFrame(rdfdata[rdfdata.p.str.startswith('wo:') | 
                               (rdfdata.p.str.startswith('rdf:type') &
                                rdfdata.o.str.startswith('wo:'))])

wo_refs['sc']=[('[%s]'%re.sub('^.*#','',x)) if '#' in x else x for x in wo_refs.s]
wo_refs['oc']=[('[%s]'%re.sub('^.*#','',x)) if '#' in x else x for x in wo_refs.o]
wo_refs = wo_refs[(wo_refs.oc.str.match('.*[\[:]')) & 
                  (wo_refs.p != 'wo:collection') &
                  (~wo_refs.s.str.match(r'.*\#collection'))]

relations = pd.DataFrame(wo_refs,columns=['s','p','o'])

   
# In[write a clean ontology relations table]:

relations.to_csv("onto-relations.tsv",sep="\t",index=False)


# In[write out some dot graphs for exploring the references]:

import numpy as np    

def dotfile(groups,prefix):
    for name,g in groups:
        strbuf = 'digraph g { \n  edge [lblstyle="above, sloped"];\n'
        for i,x in (enumerate(g.iloc)):
            strbuf += '  "%s" -> "%s" [label="%s (%s)"; weight=%2.1f; penwidth=%2.1f ; color=gray]\n' % \
            (x[0],x[1],x[2],x[3],x[3],1+np.log(x[3]))
        strbuf += '}'
        
        name2 = re.sub(r'[^A-Za-z]+','_',name)
        fn='dots/%s_%s.dot' % (prefix,name2)
        with open(fn, 'w') as the_file:
            the_file.write(strbuf)
            print("%s written"%fn)

do_dots = False

if do_dots:
    
    ref_stats = wo_refs.groupby(['sc','oc']).p.value_counts()
    gg = ref_stats.reset_index()
    dotfile(gg.groupby('oc'),'to')
    dotfile(gg.groupby('sc'),'fr')
    dotfile(gg.groupby('p'),'pp')
    dotfile([('refs',gg)],'all')





