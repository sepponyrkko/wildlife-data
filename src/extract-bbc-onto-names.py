#!/usr/bin/env python

#
# Produce a syntactically labeled term dictionary for the BBC wildlife ontology database
#
# onto-rdfdata.tsv -->> en_pos_termtokens.tsv
#
# NB!! Run FIRST collect-onto-rels.py
# >>> THEN this 
#
# Seppo Nyrkkö May 20, 2023
#


import pandas as pd

# In[Load the ontology data]:

rdfdata = pd.read_csv('onto-rdfdata.tsv',sep="\t")

# In[Pick the spcecies and other imporant]:

# Pick the spcecies and other imporant concept IDs introduced in the WO ontology:

c_classes = [x for x in rdfdata[rdfdata.p=='rdf:type'].o.unique() 
                if x.startswith('wo:')]

concept_ids = rdfdata[(rdfdata.p=='rdf:type') & (rdfdata.o.isin(c_classes))].s

# pick the statements with predicate wo:name
# these are typically species and genus with pointers to the taxon labels

taxon_names = pd.DataFrame(
        rdfdata[(rdfdata.s.isin(concept_ids)) 
        & rdfdata.p.isin(['wo:name'])])

# pick the species nodes' primary label (the representative headline)

taxon_labels = pd.DataFrame(
        rdfdata[(rdfdata.s.isin(taxon_names.s)) 
        & rdfdata.p.isin(['rdfs:label'])])

# In[Pick primary labels on taxon nodes]:

# pick the primary labels (e.g. species name)

c_names1 = pd.DataFrame(
        rdfdata[(rdfdata.s.isin(taxon_names.o)) & 
        rdfdata.p.isin(['rdfs:label'])])

# pick other labels (other common names used)

c_names2 = pd.DataFrame(rdfdata[
        (rdfdata.s.isin(taxon_names.o)) &
        rdfdata.p.isin(['wo:commonName'])])




# In[iterate over concept ids]:

termlist = []
termcolumns=['concept','lang','token','priority']

for cid in concept_ids.unique():
    if not cid.endswith("#name"):
        cidlabels = rdfdata[(rdfdata.s==cid) & (rdfdata.p=='rdfs:label')].o
    
        nametags = taxon_names[taxon_names.s==cid].o
    
        termlist+= [[cid,'en',t,0] for t in cidlabels]
        termlist+= [[cid,'en',t,1] for t in c_names1[c_names1.s.isin(nametags)].o]
        termlist+= [[cid,'en',t,2] for t in c_names2[c_names2.s.isin(nametags)].o]
    

en_termtoken_df = pd.DataFrame(termlist,columns=termcolumns)

print(en_termtoken_df.sample(5))

# for debugging only
# en_termtoken_df.to_csv('en_termtokens.tsv',sep='\t',index=False)
 




# In[POS]: part of speech + dependent structure analysis
#
#
# For dependency arcs parsed term tagging (nicknamed tag pattern)
# this analyzes multi-word tokens (in English with UDPIPE EWT 2.5)
#

import corpy.udpipe
ud_en = corpy.udpipe.Model("english-ewt-ud-2.5-191206.udpipe")


def tagpattern(udmodel, phrase):
    head_word = [phrase+' -']
    dep_words = []
    if ud_en==udmodel:
        phrase = 'the '+phrase
    sent = list(udmodel.process(phrase))
    if (len(sent)):
        s0 = sent[len(sent)-1]
        wordpos = lambda x: x.lemma+' '+x.upostag
        head_word = [wordpos(w) for w in s0.words if w.head==0]
        dep_words = [wordpos(w)+' '+w.deprel+'> '+wordpos(s0.words[w.head])  
                     for w in s0.words if w.head!=0 and w.id>0 and (w.upostag!='DET')]
    return (head_word,dep_words)

#  testing

#tagpattern(ud_fi,'Amerikan mustakarhu')
#tagpattern(ud_en,'African bush elephant')
tagpattern(ud_en,'Antarctic minke whale')
#tagpattern(ud_en,'Huon tree-kangaroo')

# write the POS + dependent analyzed terms into en_pos_termtokens.tsv

# In[POS2]: 
df = en_termtoken_df
tagpatterns = [tagpattern(ud_en,token) for token in df.token]
df['head_lemma'] = [' | '.join(x) for (x,y) in tagpatterns]
df['parts'] = [' | '.join(y) for (x,y) in tagpatterns]
# In[POS3]: 
df.drop_duplicates(inplace=True)
print(df.sample(2).T)
df.to_csv('en_pos_termtokens.tsv',sep='\t',index=False)



