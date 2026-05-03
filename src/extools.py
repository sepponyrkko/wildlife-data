#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment helper tools

Created on Tue Jul  9 14:30:52 2024

@author: seppo
"""

# In[1]

# Test area.

import numpy as np
import pandas as pd
import databox as db
import re


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def md_df(df_in: pd.DataFrame, fname='', p_index=True):
    '''
    Create markdown safe for pandoc

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    fname : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    df = pd.DataFrame(df_in)
    for c,col in df.items():
        orig = col.astype(str)
        fixed = col.astype(str).str.replace('<', '< ').str.replace('>', '> ')
        if (str(orig) != str(fixed)):
            df[c]=fixed
        
    s = df.to_markdown(tablefmt='github', index=p_index)
    if fname == '':
        return s
    with open(fname, 'w') as f:
        f.write(s)


def clipstr(instr,maxlen=40,end=False):
    '''
    Clip printable string from head or end to max length at whitespace.

    Parameters
    ----------
    instr : str
        string in
    maxlen : TYPE, optional
        desired max length The default is 40.
    end : TYPE, optional
        trim at and. The default is False.

    Returns
    -------
    str
        the trimmed (clipped) string

    '''
    if len(instr)<maxlen:
        return instr
    import re
    return re.sub(r'^\S+\s+','',instr[-maxlen:]) if not end else \
        re.sub(r'\s+\S+$','',instr[:maxlen]) 


# In[data helper]


def onto_pairs(concept_pattern: str = '.*',
               rel_name: str = '.*',
               direction: str = ''):
    '''
    Get defined ontology pairs.
    These are the "knowledge base statements" stored in the ontology.

    Parameters
    ----------
    concept_pattern : str, optional
        Regex for concept pattern. The default is '.*'.
    rel_name : str, optional
        match pattern for relation. The default is '.*'.
    direction : str, optional
        [left, right, any] The default is ''.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    oo = db.onto_df[db.onto_df.relation.str.match(rel_name)]

    if direction == 'right':
        return oo[oo.subj.str.match(concept_pattern)]

    if direction == 'left':
        return oo[oo.obj.str.match(concept_pattern)]

    return oo[oo.subj.str.match(concept_pattern) |
              oo.obj.str.match(concept_pattern)]


# In[plotting helper ICA]

def plot_ica_kde(p0: pd.DataFrame,
                 p1: pd.DataFrame,
                 pR: pd.DataFrame,
                 rel_type='location',
                 icx=0, icy=1):

    import seaborn as sns

    df0 = pd.DataFrame(p0, columns=[icx, icy])
    df0 = df0[0:len(p1)] if len(p0) > len(p1) else df0
    df1 = pd.DataFrame(p1, columns=[icx, icy])
    dfR = pd.DataFrame(pR, columns=[icx, icy])
    df0['dataset'] = 'term without reference'
    df1['dataset'] = 'term with ' + rel_type
    dfR['dataset'] = rel_type + ' reference'
    trdata = pd.concat([df0, df1, dfR])
    trdata.columns = ['IC_%d' % icx, 'IC_%d' % icy, 'dataset']

    g = sns.PairGrid(trdata, hue="dataset", palette=sns.color_palette(
        "hls", len(trdata.dataset.unique())))
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.histplot)
    g.map_upper(sns.kdeplot, warn_singular=False)
    g.add_legend()

    print('dataset sizes', trdata['dataset'].value_counts())
    return {'trdata': trdata}


# In[plotting helper GMM]


def plot_gmm_kde(p1: pd.DataFrame,
                 gmm_b,
                 icx=0, icy=1):

    import seaborn as sns
    trdata = pd.DataFrame(p1, columns=[icx, icy])
    trdata.columns = ['IC_%d' % i for i in trdata.columns]
    trdata['cluster'] = gmm_b.predict(p1)
    trdata.sort_values('cluster', ascending=True, inplace=True)

    g = sns.PairGrid(trdata, hue='cluster', palette=sns.color_palette(
        "hls", len(trdata.cluster.unique())))
    g.map_lower(sns.scatterplot)
    g.map_upper(sns.kdeplot,warn_singular=False)
    g.map_diag(sns.histplot)
    g.add_legend()

    return {'trdata': trdata}

# In[pretty token]


def pretty_token(t: str, maxlen=10, punct=True):
    '''
    Token sentence context explaining function.
    Useful for data exploration.

    Parameters
    ----------
    t : str
        token id, for instance 'Til-b8a5_18_rugosa-2'
        
    maxlen : TYPE, optional
        Length of sentence context window. The default is 20.
        
    punct : TYPE, optional
        Flag to include punctuation. The default is False.

    Returns
    -------
    dictionary:
        
        * 'word_id': word id in the sentence,
        * 't_arcs': arcpahts DF,
        * 't_synt': DF of the sentence word forms with arcpaths,
        * 'ctx': left and right sentence context
        * 's_ctx': left and right sentence context with arcpath annotation
    '''

    t_conllu_df = db.token_sent(t, punct).rename(columns={'id': 'widx'})
    t_arcs_df = db.token_arcs(t)
    t_arcs_ext_df = db.token_arcs_ex_term(t)
    t_conllu_df.widx = t_conllu_df.widx.fillna('0').astype(int)
    t_word_id = db.token_data(t).word_id
    t_synt = t_conllu_df.merge(t_arcs_df, on='widx', how='left').sort_values('widx')
    t_synt = t_synt[((t_synt.widx - t_word_id) <= maxlen) &
                    ((t_synt.widx - t_word_id) >= -maxlen)]
    t_synt.index = t_synt.widx
    t_synt = pd.DataFrame(t_synt, columns=['form', 'arcpath']).fillna('')
    f_col = t_synt.form
    
    # Context with word forms only

    f_col = t_synt.form
    ctx = [' '.join(f_col.loc[f_col.index < t_word_id]),
           '[%s]' % f_col.loc[t_word_id],
           ' '.join(f_col.loc[f_col.index > t_word_id])]

    # Context + arc paths


    f_col = t_synt.form+'('+t_synt.arcpath+')'
    s_ctx = [' '.join(f_col.loc[f_col.index < t_word_id]),
             '[%s]' % f_col.loc[t_word_id],
             ' '.join(f_col.loc[f_col.index > t_word_id])]

    t_arcs_df = t_arcs_df.copy()
#    t_arcs_df['widx'] = t_arcs_df['widx'].astype(int)
    t_arcs_df.sort_values('widx',inplace=True)
    t_arcs_df.reset_index(drop=True,inplace=True)
    
    result = {'word_id': t_word_id,
                    't_arcs': t_arcs_df,
                    't_synt': t_synt,
                    'ctx': ctx,
                    's_ctx': s_ctx,
                    't_arcs_ex_term': t_arcs_ext_df}
    
    tag = dict(db.tokentag_df.loc[t])
    term = dict(db.term_df.loc[tag['term_id']])
    
    result = {**result, **tag, **term}
    
    return dotdict(result)




# In[CE vector DF for tokens]

import scipy as sp

def compute_feat_vec(arcpath:str, lemma:str, ica_trans):
        lemma_emb = db.skipgram_model[lemma]
        ce_pos = db.arc_proj_idx(arcpath)
        rows = [0]*db.D_wordvec
        
        arr = sp.sparse.csr_matrix((lemma_emb,(rows,ce_pos)),shape=(1,db.D_input)).toarray().round(3)
        arr /= np.linalg.norm(arr)
        #print("lemma %s arcpath %s -> %s" % (lemma,arcpath,str(arr.shape)))
        return ica_trans.transform(arr + ica_trans.mean_)[0]
    

def get_ce(tokenids : list):
    
    rows = []
    
    for t in tokenids:
        rows.append(db.token_ctx_enc(t,1))
        
        # add a nice progress bar
        
        count = len(rows)
        progress = int(20.0 * count / len(tokenids))
        print(('Reading %d vectors: %20s:'%(len(tokenids),t[-20:]))
           + '|'*progress + '.'*(20-progress),end='\r')
    
    print('OK!'+' '*50,end='\r')

    df = pd.DataFrame(
        rows,
        index=tokenids)
    df.index.name='token_id'
    return df


def select_datapoints(concepts: list = [],
                      relation: str = '',
                      direction: str = 'both',
                      postprocess: bool = True):

    '''
    Extract a concept-based sample set from the corpus.

    Parameters
    ----------
    concepts : list, optional
        Filter expressions to pick concepts. The default is [].
        For instance, [r'Aardvark#species', r'Antarctic.*#species']
        
    relation : str, optional
        Filter expression to pick ontology. 
        The default is '' which matches all.
        Can be e.g. 'location' or 'adaptation'

    direction: str, optional
        Direction of the relation ['right','left','both'].
        The default is 'both'.

    postprocess: bool, optional
        Postprocess thresholding = keep the better half of hits as pairs
        The default is True.

    Returns
    -------
    tuple
        paired, unpaired, related, xdf
          - paired = datapoints matched with related concept
          - unpaired = datapoints without "proper" match
          - related = datapoints on the other end of the relation
          - xdf = cross link table of possible related tag pairs
    '''

    wordlist = concepts
    assert wordlist.__class__ == list

    xdf_columns = ['btag', 'bloc', 'rtag', 'rloc', 'rdist',
                   'btoken', 'rtoken', 'bterm', 'rterm', 'tagscore']
    pairs = [pd.DataFrame(columns=xdf_columns)]
    base_tokens = []

    conceptlist = []
    for w in wordlist:
        conceptlist.extend(
            db.tag_df.concept_uri[db.tag_df.concept_uri.str.match(w)].unique())

    count = 0
    size = len(list(set(conceptlist)))+1
    
    for w in list(set(conceptlist)):
        count += 1
        progress = int(20.0 * count / size)
        print(('%d concept tags: %40s:'%(size,w[0:40]))
           + '|'*progress + '.'*(20-progress),end='\r')
    
        
        o_df = onto_pairs(w, relation, 'right')
        base_tag_df = db.tag_df[db.tag_df.concept_uri.str.match(w)]
        base_tokens.append(base_tag_df.token_id)

        rel_tag_df = db.tag_df[db.tag_df.concept_uri.isin(
            set(o_df.subj).union(o_df.obj))]

        # exclude self-references
        rel_tag_df = rel_tag_df[~rel_tag_df.index.isin(base_tag_df.index)]

        # exclude non-relevant tags
        rel_tag_df = rel_tag_df[rel_tag_df.document.isin(base_tag_df.document)]

        # group by document id for speedup
        tag_by_doc_related = rel_tag_df.groupby('document')  # related
        tag_by_doc_baseref = base_tag_df.groupby('document')  # base
        
        max_pairs = 1000
        
        max_pairs_per_doc = max_pairs / (1+len(rel_tag_df.document.unique()))

        for doc in tag_by_doc_related.groups.keys():

            # evaluate the relation between tags base_tag --> rel_tag

            related_tags = tag_by_doc_related.get_group(doc)
            base_tags = tag_by_doc_baseref.get_group(doc)

            base_locus = base_tags.sent_id.astype(int)
            rel_locus = related_tags.sent_id.astype(int)
            xdf = pd.DataFrame(zip(base_locus.index, base_locus)).\
                merge(pd.DataFrame(zip(rel_locus.index, rel_locus)), how='cross')
            xdf.columns = xdf_columns[0:4]
            xdf['rdist'] = abs(xdf.bloc-xdf.rloc)
            xdf['btoken'] = list(db.tag_df.loc[xdf.btag].token_id)
            xdf['rtoken'] = list(db.tag_df.loc[xdf.rtag].token_id)
            xdf['bterm'] = list(
                db.term_df.loc[db.tag_df.loc[xdf.btag].term_id].term)
            xdf['rterm'] = list(
                db.term_df.loc[db.tag_df.loc[xdf.rtag].term_id].term)
            xdf['tagscore'] = np.array(db.tag_df.loc[xdf.rtag].score) * np.array(
                db.tag_df.loc[xdf.btag].score) * .01 / (1+xdf['rdist'])
            
            xdf = xdf[xdf['tagscore'] >= 1.0] # require at least score 1.0
            xdf = xdf.sort_values('tagscore', ascending=False)
            
            # limit the number of topic-comment pairs per document
            
            tries = 3
            while (len(xdf)>max_pairs_per_doc) and tries:            
                xdf2 = xdf[xdf['tagscore'] > xdf['tagscore'].median()]
                xdf = xdf2 if len(xdf2) else xdf
                tries -= 1

            pairs.append(xdf)

    xdf = pd.concat(pairs)
    
    # Truncate the results at max_pairs with maximum tag score
    
    xdf = xdf.sort_values('tagscore', ascending=False).head(max_pairs)
    
    base_token_ids = pd.concat(base_tokens).unique()
    
    print('OK!'+' '*50)

    # post-process thresholding:
    # globally in the corpus, between each term pair, only keep the better half scored as 'hits'

    if postprocess:
        xx = []
        for gn, gg in xdf.sort_values('tagscore', ascending=False).groupby(['bterm', 'rterm']):
            g = pd.DataFrame(gg)
            tries = 3
            while (len(g)>10) and tries:
                g = g[g.tagscore >= g.tagscore.mean()]
                tries -= 1
            xx.append(g)
            
        xdf = pd.concat(xx) if len(xx) > 0 else xdf

    # add the referring base tokens to paired token dataframe

    paired = get_ce(db.tag_df.loc[xdf.btag].token_id.unique())
    pairs = get_ce(db.tag_df.loc[xdf.rtag].token_id.unique())

    # add the remaining base tokens, to unpaired token dataframe
    
    unpaired_tokens = list(set(base_token_ids).difference(set(xdf.btoken)))
    
    unpaired_tokens_df = pd.DataFrame(unpaired_tokens, columns=['token_id'])
    unpaired_tokens_df['doc']=unpaired_tokens_df.token_id.str[0:8]
    unpaired_tokens = list(unpaired_tokens_df.sort_values('token_id').
                           groupby('doc').head(5).token_id.head(1000))

    unpaired = get_ce(unpaired_tokens)

    return paired, unpaired, pairs, xdf

# In[token dictionary]

def build_token_dictionary(tokenset : list, 
                           punct : bool = True, 
                           maxlen : int = 20,
                           dot_dict : bool = False):
    tokendict = dict()
    count = 0
    for t in tokenset:
        pt = pretty_token(t, punct=punct, maxlen=maxlen)
        tokendict[t] = pt if dot_dict else dict(pt)
        count += 1
        progress = int(20.0 * count / len(tokenset))
        print(('Reading %d tokens: %20s:'%(len(tokenset),t[-20:]))
               + '|'*progress + '.'*(20-progress),end='\r')
    print('-'*50+' '*10)
    print('OK')
    return tokendict
    
        

def short_token_ctx(c,ll=30,lc=11):
    '''
    Space-Pad the context block from pretty token output.

    Parameters
    ----------
    c : list
        the ctx snippet from pretty token output.
    ll : int, optional
        length of first and third column. The default is 30.
    lc : int, optional
        center column length = token width. The default is 11.

    Returns
    -------
    s : str
        Padded example of the context block.

    '''
    ctok = c[1][1:-1].upper()
    ctok += ' '*((lc-len(ctok))//2) if len(ctok)<lc else ''
    ctok = (' '*(lc-len(ctok)) +ctok) if len(ctok)<lc else ctok
    
    c0 = c[0]
    c0 = '.'*(ll-len(c0))+c0 if len(c0)<ll else c0
    
    s = (("%"+str(ll)+"s")% c0[-ll:]) + (" %s "%ctok) + ("%-"+str(ll)+"s")%c[2][0:ll]
    
    if len(c0)>ll:
        ss = re.sub(r'^\S+','.',s)
        s = '.'*(len(s)-len(ss)) + ss
        
    if not s.endswith(' '):
        s = s[:-1]+'='
    return s
    

# In[heatmap]
    
import matplotlib.pyplot as plt
import matplotlib
    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts