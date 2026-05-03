#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:27:37 2024

@author: seppo
"""

import pandas as pd
import re

def load_termtokens():
    '''read the term tokens termtokens.tsv'''
    
    termtok_df=pd.read_csv("en_case_termtokens.tsv",sep="\t")    
    termtok_df["tag_w"] = 100/(1+termtok_df.priority)
        
    term_df = pd.DataFrame(termtok_df.rename(
                    columns={'concept':'concept','token':'term'}),
                    columns=['term','concept','lang','tag_w','head_lemma','parts'])
    
    term_df["term_id"]=term_df.index
    term_df["concept_uri"]=list(term_df.concept)
    term_df["concept"]=term_df.concept.str.replace('#.*$','',regex=True)
    term_df=pd.DataFrame.drop_duplicates(term_df,subset=['term','concept_uri','lang'])           
    term_df.parts.fillna('',inplace=True)    
    term_df['upos']=term_df.head_lemma.apply(lambda x: re.sub("^.* ","",x))
    term_df['lemma']=term_df.head_lemma.apply(lambda x: re.sub(" [A-Z]+$","",x))    
    
    return term_df
    

