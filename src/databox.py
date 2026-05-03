#!/usr/bin/env python
# coding: utf-8
#
"""
Data (tool)box for my wildlife experiment.
This contains methods for data loading and accessing the tokens and tags.
(This replaces the old toolbox library.)

Use the "init_tags.py" to initialize the data first.

@author: seppo

"""
# import libraries

import os
import pandas as pd
import dask.dataframe as ddf
import itertools
import scipy as sp
import skipgram

# In[Init]

_sent_df = None    # sentence id, text
onto_df = None    # triplets
term_df = None    # concept and terms

# define the data frames which are read and filled in 
# the data import and encoding process.

D_wordvec=0   # fastvec preprocessing    
D_input = 90   # the context encoding space     



def read_onto():
    '''
        Read the ontology db from onto-relations.txt
        with tab-separated columns relation,pred,subj,obj
    '''
    global onto_df
    onto_df = pd.read_csv("onto-relations.txt",sep="\t")
    onto_df.columns=['relation', 'pred', 'subj', 'obj']
    print ("OK -- read the ontology db",onto_df.shape)


import skipgram as sgm

def set_D_wordvec(dim_w : int):
    global skipgram_model, D_wordvec 
    D_wordvec = dim_w
    skipgram_model = sgm.build_skipgram_model(D_wordvec)
  

import termlib

def read_termtokens():
    '''read the term tokens termtokens.tsv'''
    
    global term_df
    term_df = termlib.load_termtokens()
    

def get_sent_df():
    '''
        A global sentence dataframe sent_df is read in with the 
        dask dataframe utilities from multiple sentence tables
        
        Returns
        -------
        _sent_df : pd.DataFrame
            sentence dataframe sent_df 
       
    '''
    global _sent_df 
    if _sent_df:
        return _sent_df
    
    print("Dataframe _sent_df loading.")

    sentscorpus = ddf.read_csv("../corpus/wiki_articles/tsv/*_sents.tsv",
                               sep="\t",
                               dtype=str)
    
    _sent_df = sentscorpus.compute()
    _sent_df.set_index(['document','sent_id'],inplace=True)
    
    return _sent_df

def read_token_tags():
    '''
    Loads the saved state of token tags (location) and concept tags (links) 
    into their 'home' data frames (token_df and tag_df)
    '''
    global token_df, tag_df
    if (os.path.exists("tokens.tsv") & os.path.exists("token_tags.tsv")):
        token_df = pd.read_csv('tokens.tsv',sep='\t').set_index('token_id')
        tag_df   = pd.read_csv('token_tags.tsv',sep='\t').set_index('tag_id')
        print("Restored %d tags and %d tokens. Run update_corpus_tagdata() to update." % (len(tag_df),len(token_df)))        
    else:
        raise "=== Run update_corpus_tagdata in init_tags.py to build token_tags.tsv and tokens.tsv==="



read_onto()
read_termtokens()
read_token_tags()
set_D_wordvec(8)



# In[Getters]

from collections import OrderedDict
cache_conllu_doc = OrderedDict()
cache_conllu_doc_limit = 64

def get_document_conllu(docid : str, debug=False):
    '''
    Get document conll-u data frame by referred document id
    or read it from cache dictionary.

    Parameters
    ----------
    d : str
        document id
    Returns
    -------
    corpus_df : pd.DataFrame
        Corpus data frame
    '''
    
    global cache_conllu_doc
    
    if docid in cache_conllu_doc.keys():
        cache_conllu_doc.move_to_end(docid)
        if debug:
            print("cache_conllu_doc HIT "+docid)
        return cache_conllu_doc[docid] 
    
    if (len(cache_conllu_doc.keys()) > cache_conllu_doc_limit):
        cache_conllu_doc.popitem(False)
    
    corpus = ddf.read_csv("../corpus/wiki_articles/tsv/"+docid+"_words.tsv",
                      sep="\t",
                      dtype=str
                      )
    if (debug):
        print("Reading document %s" % docid)
    cache_conllu_doc[docid] = corpus.compute()

    return cache_conllu_doc[docid]


def filter_conllu_by_sent_id(conllu_df: pd.DataFrame = None, 
                    sent_id: str = '', 
                    include_punct : bool = False):
    '''   
    Parameters
    ----------
    conllu_df : pd.DataFrame
        DataFrame of the corpus in the conll-u format
        (can be empty if document given in doc_id)
    doc_id : str
        Document identifier (if multiple documents in conllu_df dataframe)
    sent_id : str
        Sentence id number within the document

    Returns
    -------
    sdf : pd.DataFrame
        DataFrame of the corpus sentence in the conll-u format, with index reset

    '''

    sent_id = str(sent_id)
    sdf = pd.DataFrame(
        conllu_df if include_punct else conllu_df[(conllu_df.upos != 'PUNCT')])

    if sent_id != '':
        sdf = sdf[(sdf.sent_id == sent_id)]

    sdf.reset_index(inplace=True, drop=True)
    sdf.index = sdf.id
    return sdf


def get_sent_wordarcs(sent_conllu: pd.DataFrame, word_id : str, depth=3, upwards=3, no_follow=""):
    '''
    build a word arc dataframe.    

    Parameters
    ----------
    sent_conllu : pd.DataFrame
        Parsed sentence as a conll-u dataframe.
    word_id : str
        word id in the parsed sentence dataframe. For instance '2'
    depth : int, optional
        max hop count. The default is 3.
    upwards : int, optional
        steps available to ascend the tree. The default is 3.
    no_follow : str, optional
        do not follow to this node id. The default is "".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns (arc_chain,arc_lemma,arc_upos).

    '''
    arcs = list()
    if (depth > 0):

        word_id = str(word_id)
        wdf = sent_conllu.loc[word_id]  # single-word dataframe

        rdep = wdf.deprel
        rlemma, rupos, rwidx = (None, None, None)
        if wdf['head'] in sent_conllu.index:
            rlemma, rupos, rwidx = (sent_conllu.lemma.loc[wdf['head']],
                                    sent_conllu.upos.loc[wdf['head']],
                                    wdf['head'])
        # Also append root arcs, if upwards counter > 0

        if (upwards > 0):
            if (not pd.isna(rlemma)):
                rdep = str(rdep)+"<"
                rlemma = str(rlemma)
                rwidx = str(rwidx)
                arcs.append((rdep, rlemma, rupos, rwidx))
                if upwards > 0:
                    for (arc_type, arc_lemma, arc_upos, arc_id) in \
                            get_sent_wordarcs(sent_conllu, rwidx, depth-1, upwards-1, no_follow=word_id):

                        arc_chain = rdep+arc_type
                        arcs.append((arc_chain, arc_lemma, arc_upos, arc_id))

        # Descend the dependent word arcs

        depword_df = sent_conllu[sent_conllu['head'] == word_id]
        for i in depword_df.index:
            if depword_df.id[i] != no_follow:
                deprel_name = depword_df.deprel[i] + ">"
                arcs.append(
                    (deprel_name, depword_df.lemma[i], depword_df.upos[i], depword_df.id[i]))

                for (arc_type, arc_lemma, arc_upos, arc_id) in get_sent_wordarcs(sent_conllu, depword_df.id[i], depth-1, upwards=0, no_follow=word_id):
                    arc_chain = deprel_name+arc_type
                    arcs.append((arc_chain, arc_lemma, arc_upos, arc_id))
    return arcs


# In[Token_Getters]

def token_sent(token_id: str, include_punct: bool = False):
    '''
    get token sentence conllu DataFrame by referred token_id

    Parameters
    ----------
    token_df : pd.DataFrame
        The token store
        
    token_id : str    
        The token of interest

    Returns
    -------
    df : pd.DataFrame
        DataFrame with sentence conll-u data

    '''
    token = token_df.loc[token_id]
    doc_df = get_document_conllu(token.document)
    df = filter_conllu_by_sent_id(doc_df, token.sent_id, include_punct)
    return df



from collections import OrderedDict
cache_token_arcs = OrderedDict()
cache_token_arcs_limit = 20480

def token_arcs(token_id: str, debug=False, rebuild_file=False):
    '''
    load disk-cached token arcs by referred token_id

    Parameters
    ----------
    token_id : str    
        The token of interest

    Returns
    -------
    arcs : pd.DataFrame
        DataFrame with columns (arc_chain,arc_lemma,arc_upos).

    '''
    
    # cache
    
    global cache_token_arcs
    
    if token_id in cache_token_arcs.keys():
        cache_token_arcs.move_to_end(token_id)
        if debug:
            print("cache_token_arcs HIT "+token_id)
        return cache_token_arcs[token_id]
    
    if (len(cache_token_arcs.keys()) > cache_token_arcs_limit):
        cache_token_arcs.popitem(False)
        
    # cache miss
    
    token = token_df.loc[token_id]
    doc_id = token.document
    sent_id = token.sent_id
    word_id = token.word_id
    arc_df = pd.DataFrame(
        columns=['token_id', 'arcpath', 'lemma', 'upos', 'widx'])
    arcpath_cache_fn = "../corpus/arcs/%s.tsv" % (doc_id)
    
    if (not rebuild_file) and os.path.isfile(arcpath_cache_fn):
        arc_df = pd.read_csv(arcpath_cache_fn, sep='\t')
        if debug:
            print("found %s of size %d" % (arcpath_cache_fn, len(arc_df)))

    warc_df = arc_df[arc_df.token_id == token_id]

    if len(warc_df) < 1:
        if debug:
            print("updating %s" % (arcpath_cache_fn))

        df = get_document_conllu(doc_id)
        arcs = get_sent_wordarcs(filter_conllu_by_sent_id(df, sent_id), word_id)

        warc_df = pd.DataFrame(
            arcs, columns=['arcpath', 'lemma', 'upos', 'widx'])
        warc_df['token_id'] = token_id
        warc_df.widx = warc_df.widx.astype(int)

        arc_df = pd.concat([arc_df, warc_df])
        arc_df.drop_duplicates(inplace=True)
        arc_df.to_csv(arcpath_cache_fn, sep='\t', index=False)
        if debug:
            print("written %s of size %d" % (arcpath_cache_fn, len(arc_df)))
            
    df = warc_df.drop(columns=['token_id']).sort_values('widx')
    cache_token_arcs[token_id] = df
    return df


tokentag_df = []

# convenience df to get best scored tag data by token_id
def tokentags():
    '''
    convenience data frame - best scored tag data by token_id

    Returns
    -------
    tokentag_df : pd.DataFrame
        get best scored tag data per each token_id
   '''
    global tokentag_df 
    if len(tokentag_df)<1:
        tokentag_df = tag_df.sort_values('score',ascending=False).groupby('token_id').head(1).set_index('token_id')
        tokentag_df['term']=list(term_df.iloc[tokentag_df.term_id].term)
        
    return tokentag_df


def token_data(token_id: str):
    '''
    Shorthand for `token_df.loc[token_id]. Returns pd.Series of token`
    '''
    return token_df.loc[token_id]


def token_termpart_exclusion(token_id: str):

    termmatches = tag_df[tag_df.token_id == token_id].sort_values(
        'score').tail(2).term_id.unique()
    
    patterns = []
    for termmatch in termmatches:
        for x in term_df.iloc[termmatch].parts.split(' | '):
            patterns.extend([' '.join(x.split()[0:2])])

    return list(set(patterns))

def token_arcs_ex_term(token_id : str):
    '''
    Gets token arcs, but excluding the compound term parts, such as:
            'Arizona black-tailed prairie dog',
            'brown-throated three-toed sloth', 
            'straw-coloured fruit bat',
            'crown-of-thorns starfish', 
            'white-fronted bee-eater',
            'East African long-eared elephant shrew',
            'Black-flanked rock-wallaby'

    Parameters
    ----------
    token_id : str
        The token id

    Returns
    -------
    pd.Dataframe
        Dataframe of the context arcs, excluding arcs belonging to the term

    '''
    a = token_arcs(token_id)
    
    ex_list = token_termpart_exclusion(token_id)
    
    t_widx = int(token_data(token_id).word_id)    
    
    aa = a[(t_widx-a.widx).abs() <= len(ex_list)]
    
    # excluded arcs (because in compound term)
    
    ex_a_index = aa[(aa.lemma+' '+aa.upos).isin(ex_list)].index
    
    return a.drop(ex_a_index).sort_values('widx')
    
    
    

def token_tags(token_id: str):
    '''
    A function that simplifies the process of fetching
    assigned tags for a token (which map to term entries).
    
    Parameters
    ----------
    token_id : str
        The token id, for instance 'Til-b8a5_18_rugosa-2'

    Returns
    -------
    pd.DataDrame
        assigned tags for a token (with term_id, score, concept_uri etc.) 
    '''
    return pd.DataFrame(tag_df[tag_df.token_id==token_id]).sort_values('score',ascending=False)


# In[Context_Encodings]

import hashlib

def arc_proj_idx(arcname : str):
    '''
    Parameters
    ----------
    arcname : the n-arc name to be hashed in 

    Returns
    -------
    positions : list
      List of integer indices to store the encoded vector of dependent word
    '''   
    return [x%(D_input) for x in (hashlib.md5(str.encode(arcname)).digest() +
                                  hashlib.md5(str.encode(arcname+'X')).digest() +
                                  hashlib.md5(str.encode(arcname+'Y')).digest() +
                                  hashlib.md5(str.encode(arcname+'Z')).digest()
                                  )[0:(D_wordvec)]]

import numpy as np

def token_link_matrix(token_id: str, emb_mode: int = 0, debug=False):
    '''
    Create the Contextual encoding for ML.

    Parameters
    ----------
    token_id : str
        Token id to analyze, for instance 'Aar-5e58_102_dogs-18'

    Returns
    -------
    synt_ctx_enc_df : pd.DataFrame
        The context encoding array per row
    
    synt_ctx_array : TYPE
        The context encoding array per row

    '''
    
    token_arcpath_df = token_arcs_ex_term(token_id)
    n_arcs = len(token_arcpath_df)

    lemma_embs = [skipgram_model[x] for x in token_arcpath_df.lemma] 

    ce_poss = [arc_proj_idx(x) for x in token_arcpath_df.arcpath]

    cols = list(itertools.chain.from_iterable(ce_poss))  # concatenate entries
    rows = [i for i in range(n_arcs)
            for j in range(D_wordvec)]  # 0,0,0...n-1,n-1
    

    vals = list(itertools.chain.from_iterable(lemma_embs))


    synt_ctx_array = sp.sparse.csr_matrix((vals, (rows, cols)),
                                            shape=(n_arcs, D_input)
                                            ).toarray().round(3)
    
    if (emb_mode % 2) == 1:

        synt_ctx_array = (synt_ctx_array.T / 
                          np.linalg.norm(synt_ctx_array,axis=1)).T
        
    synt_ctx_enc_df = token_arcpath_df.reset_index(drop=True).copy()

    if debug:
        position_arr = sp.sparse.csr_matrix(([1]*len(vals), (rows, cols)),
                                            shape=(n_arcs, D_input)
                                            ).toarray().round(3)
        return synt_ctx_enc_df, synt_ctx_array, position_arr, lemma_embs
    

    return synt_ctx_enc_df, synt_ctx_array


# In[token context encoding vector]

from collections import OrderedDict

cache_token_ctx_enc = OrderedDict()
cache_token_ctx_enc_limit = 64000 # sanity limits - one item is around 1 kb

# No sense in local disk storing

# def store_cache_token_ctx_enc(fn = 'token_ctx_enc_cache.pkl'):
#     import joblib    
#     joblib.dump(cache_token_ctx_enc,fn)
#     print("Stored %d cached token encodings in %s" % (len(cache_token_ctx_enc.keys()),fn))
    
# def restore_cache_token_ctx_enc(fn = 'token_ctx_enc_cache.pkl'):
#     global cache_token_ctx_enc
#     import joblib
#     if os.path.exists(fn):
#         cache_token_ctx_enc = joblib.load(fn)
#         print("Restored %d cached token encodings in %s" % (len(cache_token_ctx_enc.keys()),fn))
#     else:
#         print("No cached token encodings in %s" % (fn))

# restore_cache_token_ctx_enc()

tokentags() # refresh the lookup table

def token_ctx_enc(token_id: str, emb_mode, debug=False):
    '''
    This composes a context encoding vector for a token

    Parameters
    ----------
    token_id : str
        Token id of the word.

    Returns
    -------
    ctx_enc_v : np.array
        A vector in the input space
    '''
    
    # hash id for this request

    vec_id = token_id + ':' + str(emb_mode) + \
        ':' + str(D_wordvec) + ':' + str(D_input)

    # cache

    global cache_token_ctx_enc

    if vec_id in cache_token_ctx_enc.keys():
        cache_token_ctx_enc.move_to_end(vec_id)
        if debug:
            print("cache_token_ctx_enc HIT "+vec_id)
        return cache_token_ctx_enc[vec_id]

    if (len(cache_token_ctx_enc.keys()) > cache_token_ctx_enc_limit):
        for i in range(1000):
            cache_token_ctx_enc.popitem(False)
            
    # calculate

    link_df, arc_nv = token_link_matrix(token_id, emb_mode)

    ctx_enc_v = arc_nv.sum(axis=0)
    
    # cache

    cache_token_ctx_enc[vec_id] = ctx_enc_v   
    
    # if (len(cache_token_ctx_enc) % 1000 == 0):
    #     store_cache_token_ctx_enc()
        
    return ctx_enc_v


# In[x]

# set_D_wordvec(21) # experimental only
