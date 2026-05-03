#!/usr/bin/env python
# coding: utf-8

'''
Initialize experiment data sets:
 - tagging terms in documents
 - build context encoding vectors
 - fasttext skipgram model
 
NB Run this interactively in e.g. notebook or ipython:
  update_corpus_tagdata()  
  build_ctx_enc_df(token_df)
 
'''

# import libraries


import pandas as pd
import os
import dask.dataframe as ddf

sent_df = None    # sentence id, text
term_df = None    # concept and terms

# define the data frames which are read and filled in 
# the data import and encoding process.

#    initialize sane values
    


# In[Fasttext word vector model]:
   
import skipgram

D_wordvec = 8
skipgram_model = skipgram.build_skipgram_model(D_wordvec)

# In[termtokens]:

import termlib
def read_termtokens():
    '''read the term tokens from en_case_termtokens.tsv'''
    
    global term_df
    term_df = termlib.load_termtokens()
    
read_termtokens()
print("read %d termtokens"%len(term_df))


# In[document reader helper by document id]:

def read_document(d : str):
    '''
    Get document conll-u data by referred document id

    Parameters
    ----------
    d : str
        document id

    Returns
    -------
    corpus_df : pd.DataFrame
        Corpus data frame

    '''
    
    corpus = ddf.read_csv("../corpus/wiki_articles/tsv/"+d+"_words.tsv",
                      sep="\t",
                      dtype=str
                      )
    print("Reading document %s" % d)        
    return corpus.compute()





# In[Text Load]

def refresh_sent_df():
    '''
        A global sentence dataframe **sent_df** is read in with the 
        dask dataframe utilities from multiple sentence tables
    '''
    global sent_df 
    print("Dataframe sent_df loading.")

    sentscorpus = ddf.read_csv("../corpus/wiki_articles/tsv/*_sents.tsv",
                               sep="\t",
                               dtype=str)
    
    sent_df = sentscorpus.compute()
    sent_df.set_index(['document','sent_id'],inplace=True)
    
    print("Dataframe sent_df loaded.")
    print("Sample:\n%s\n\n" % str(sent_df.sample(2)))

refresh_sent_df()



# In[Matching the Multitoken terms]

def sent_arcs_list(sdf : pd.DataFrame()):
    '''
    Get a list of dependency arcs in the sentence
    for compound-word term tagging evaluation

    Parameters
    ----------
    sdf : pd.DataFrame()
        conll-u formatted sentence data frame

    Returns
    -------
    str_arcs : list
        List of string-formatted arcs

    '''

    head_lemmas = pd.Series([sdf.lemma[sdf.id==h].max() for h in sdf['head']]).fillna('')
    head_uposs = pd.Series([sdf.upos[sdf.id==h].max() for h in sdf['head']]).fillna('')
    s_arcs = pd.DataFrame(zip(sdf.lemma,sdf.upos,sdf.deprel+'>',head_lemmas,head_uposs))
    str_arcs = [(list(s_arcs.loc[i])) for i in s_arcs.index]
    return str_arcs


def evaluate_term_form(sdf : 'pd.DataFrame', cand_token_id : 'str', sent_arcs : list = []):
    '''
   
    Evaluates the given term pointer in a sentence data frame.

    Parameters
    ----------
    sdf : pd.DataFrame
        sentence data frame.
    cand_token_id : str
        candidate wordtoken id
    sent_arcs : list
        list of sentence arcs - can be empty

    Returns
    -------
    results : list of tuples (term_id, score, startpos, endpos)
        - **concept URI** for this
        - ID which term is found at this location
        - **coverage score** for this term 
        - Concept tag confidence * hit cvg = tag score
        - Token surface form
        - Number of matched term token parts (including head)
        
    Note
    ----
    this still needs to be tested

    '''
    if (len(sent_arcs)<1):
        sent_arcs=sent_arcs_list(sdf)
        
    filt_lemma = sdf[sdf.id==str(cand_token_id)].lemma.max()
    filt_upos = sdf[sdf.id==str(cand_token_id)].upos.max()
    token_form = sdf[sdf.id==str(cand_token_id)].form.max()
    termcands = term_df[(term_df.lemma==filt_lemma) & (term_df.upos==filt_upos)]
    results=[]
    
    if (len(termcands)):
    
        # collect a dataframe of sentence inter-token relations
    
        global sentence_arcstr_list # for debug purp
        sentence_arcstr_list=[' '.join(sa) for sa in sent_arcs]
        
        for tci in termcands.index:
            tc = termcands.loc[tci]
            #print("checking "+tc.concept)
            hits = 1
            checks = 1
            
            # iterate over argument definitions for multitoken term, e.g.
            # ['Dall PROPN compound> sheep NOUN' , ...]

            parts = [p for p in tc.parts.split(' | ') if len(p)>0]
            
            for req_arg_part in parts:
                (arg_lemma,arg_upos,arg_dep,head_lemma,head_upos) = req_arg_part.split(' ')
                checks += 1
                if req_arg_part in sentence_arcstr_list:
                    hits += 1
        
            score = int(100 * hits / checks)
            
            tag_w = (300 + tc.tag_w) * 0.25 # [scale the term weight to 75...100%]
            
            results.append((tc.concept_uri, tci, score, int( 0.01*tag_w*score), token_form, hits, checks))
    
    return results

# In[Tagging the tokens]

import hashlib
def hash_prefix(in_str):
    return ''.join(["%02x"%i for i in (hashlib.md5(str.encode(in_str)).digest()[0:2])])

def tag_tokens_loop(c_df : 'pd.DataFrame'):
    '''
    Iterates over 1) sentences in documents 2) words in these sentences
    
    Evaluates each term candidate.
    
    Parameters
    ----------
    c_df : 'pd.DataFrame'
        corpus collection where to find term occurrences

    Returns
    -------
    'pd.DataFrame'
        DF with columns:
            
         - document, sent_id, word_id
         - term_id
         - coverage
         - concept_uri
         - score
         - found_parts, req_parts
         - token_id
    '''
    term_coll = []
        
    all_sents = c_df.groupby(by=['document','sent_id'])
    for (g,df) in all_sents:
            (docid,sentid)=g
            sdf=df
            
            # Term head lemma and UPOS must match the given term in order to evaluate hit
            
            lemma_cand_mask = (sdf.lemma.str.lower() + sdf.upos).isin(term_df.lemma.str.lower() + term_df.upos)
            print(sdf.lemma[lemma_cand_mask])
            arcs = sent_arcs_list(sdf)
            
            for id in sdf[lemma_cand_mask].id:
                termcands = evaluate_term_form(sdf,id,arcs)
                term_coll.extend([(docid,sentid,id)+t for t in termcands])
                
    tags = pd.DataFrame(term_coll,columns=['document','sent_id','word_id','concept_uri','term_id','coverage','score','token_form','found_parts','req_parts'])
    hashpart = tags.document.apply(hash_prefix)
    tags['token_id']=(tags.document.str[0:3]+ hashpart +
                     "_"+tags.sent_id+"_"+tags.token_form+"-"+tags.word_id)
    tags.index = list(tags['token_id'] + '_' + tags['term_id'].astype(str))
    tags.index.name = 'tag_id'

    return tags 


# In[Tagging data]


def batch_tag_articles(testrun_flag=False,suffixstart=0,xsuffix=None):
    '''
    This scans the corpus directory with hexadecimal suffixes 00..ff
    and writes out raw tag candidates to corpus/tags/raw_tags_(suffix).tsv
    

    Returns
    -------
    None.

    '''
    
    suffixes = ['%02x'%x for x in range(suffixstart, (suffixstart+1) if testrun_flag else 256)]
    suffixes = suffixes + ['cx']
    if (xsuffix):
        suffixes = xsuffix
    
    for suffix in suffixes:
        tsv_name  = "../corpus/tags/raw_tags_"+suffix+".tsv"
        corp_name = "../corpus/wiki_articles/tsv/*"+suffix+"_words.tsv"
        
        if not os.path.exists(tsv_name):

            print("Reading %s" % corp_name)
            
            corpus = ddf.read_csv(corp_name,
                              sep="\t",
                              dtype=str
                              )
        
            c_df = corpus.compute()
            
            print("Read CONLL-U << %s" % c_df.document.unique())
        
            tag_df = tag_tokens_loop(c_df)
    
            print("Tag tokens >> %s" % tsv_name)
            tag_df.to_csv(tsv_name,sep='\t')

            print("done %s" % tsv_name)
        else:
            print("found %s" % tsv_name)



# In[compose tagdata]

def compose_tagdata():
    '''
    Concatenate the raw tagging data from 
    corpus/tags/raw_tags_(suffix).tsv
    
    clean up and write proper tag candidates to token_tags.tsv
    and then write tokens.tsv containing the token_id    
    '''
    global tag_df, token_df
   
    print("compose_tagdata(): reading raw_tags ....")
    
    raw_tag_dask = ddf.read_csv("../corpus/tags/raw_tags_*.tsv",sep="\t").set_index('tag_id')
    raw_tag_df = raw_tag_dask.compute().drop_duplicates()

    tag_df = pd.DataFrame(raw_tag_df)

    concept_medians = tag_df.groupby('concept_uri').coverage.median()
    tag_concept_median=list(concept_medians.loc[tag_df.concept_uri])
    
    token_medians = tag_df.groupby('token_id').coverage.median()
    tag_token_median = list(token_medians.loc[tag_df.token_id])
    
    tag_df['approved_c'] = tag_df['coverage']>=tag_concept_median    
    tag_df['approved_t'] = tag_df['coverage']>=tag_token_median    
    coverage_min = 50
    
    
    tag_df = tag_df[tag_df.approved_t &
                    tag_df.approved_c & 
                    (tag_df.coverage>coverage_min) ]
    
    tag_df.drop(columns=['approved_c','approved_t'],inplace=True)    

    tag_df.to_csv('token_tags.tsv',sep='\t')
    
    token_df = pd.DataFrame(tag_df,
                            columns=['token_id','document',
                                     'sent_id','word_id',
                                     'token_form']).\
                        drop_duplicates().set_index('token_id')    
                        
    token_df.to_csv('tokens.tsv',sep='\t')
    
    print("Composed %d raw tags to %d tags and %d tokens" % (len(raw_tag_df),len(tag_df),len(token_df)))

    print("== Remember to run restore_token_tags() !!! ==")

# In[Compose tags and token files now]

def update_corpus_tagdata(testrun_flag=False,suffixstart=0):
    '''
    build missing token and tags table in the corpus directory
    '''    
    global tag_df, token_df    
    batch_tag_articles(testrun_flag,suffixstart)
    compose_tagdata()

def restore_token_tags():
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
        print("=== Run update_corpus_tagdata(debug_flag,suffix=...) to update tokens.tsv. ===")

restore_token_tags()

