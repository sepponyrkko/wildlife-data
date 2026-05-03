#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 09:25:38 2025

@author: seppo
"""


import pandas as pd
import joblib
import numpy as np

import dask.dataframe as ddf

from sample_group import SG
from som_lattice import SomLattice

import databox as db

import extools as ex
import syn_tree

def nrm(x):
    '''
    normalize vector x by dividing with the sum(x)
    '''
    return x/(0.000001 + sum(x))

# In[class]

class SomSpace:

    def get_som_model(self, n_dim : int = 24):
        
        idata_df = self.idata_df

        my_som = SomLattice(n_dim)
        fn = 'data/full_SOM_%d.pkl' % n_dim

        try:
            my_som.SOM = joblib.load(fn)
            print("restored SOM model from %s" % fn)
            
        except:
            print("train SOM")
            print(idata_df.shape)
            
            my_som.train_SOM(np.array(idata_df))
            joblib.dump(my_som.SOM,fn)
            print("written SOM model to %s" % fn)

        self.my_som = my_som

    def load_ce_data(self):

        try:
            full_ce_data_df = pd.read_parquet('data/full_ce_data.pq')
            print("full_ce_data_df restored")
            print(full_ce_data_df.shape)
        except:

            xs = []
            for g in SG._get_group_list():
                gg = SG(g)
                xs.append(gg.ce_data_df.copy().assign(group=g))

            full_ce_data_df = pd.concat(xs)
            full_ce_data_df.set_index('group', append=True, inplace=True)

            xs = None
            full_ce_data_df.to_parquet('data/full_ce_data.pq')
            print("full_ce_data_df.pq written")
            print(full_ce_data_df.shape)
        return full_ce_data_df
    
    def get_ICA_transformer(self, ce_data_df : pd.DataFrame, n_dim : int):
        
        fn = 'data/ICA_transformer_%d-dim.pkl' % n_dim
        
        try:
            ica_tr = joblib.load(fn)
            print("Restored FastICA from "+fn)

        except:        
            from sklearn.decomposition import FastICA
            print("Building FastICA and fitting with context enc data")
            print(ce_data_df.shape)
            
            ica_tr = FastICA(n_components=n_dim,max_iter=2048,random_state=42)
            ica_tr.fit(ce_data_df)
            joblib.dump(ica_tr, fn)
            print("written "+fn)
        
        return ica_tr
        

    def init_ica_datapoints(self, ce_data_df):
        
        tf = self.get_ICA_transformer(ce_data_df, 24)

        idata_df = pd.DataFrame(tf.transform(ce_data_df),
                                index=ce_data_df.index)

        idata_df = idata_df.reset_index().set_index(
            ['rel_role', 'group', 'token_id'])
        
        print(idata_df.shape)
        print(idata_df.sample(5))
        
        self.ica_tr = tf
        self.idata_df = idata_df

    def get_cxcy_df(self):
        width, height, n_comps = self.my_som.SOM.shape
        cxcy_df = pd.DataFrame(
            [dict(cx=cx, cy=cy, cell="%02d_%02d" % (cx, cy),
                  dx=cx+(0.5 * (cy % 2)), dy=cy*0.85)
                for cx in range(width) for cy in range(height)])
        return cxcy_df

    def annotate_lattice(self, label_df = None, color=None, textoff = True):
        from matplotlib import figure, pyplot as plt
        import matplotlib as mpl

        my_som = self.my_som

        fig = figure.Figure(figsize=(8, 5), dpi=150)
        fig.tight_layout(pad=5.0)

        axs = fig.subplots(1, 1, sharey=False)
        ax = axs

        legends = []



        cxcy_df = self.get_cxcy_df()
        
        rgbs = {}

        for cell, c in cxcy_df.iterrows():
            
            if color is None:
                rgb = my_som.SOM[int(c.cx), int(c.cy), 0:3]*.5+.5
                
            else:
                val = 0                
                if c.cell in color.index:
                    val = 1.5+np.log10(color.loc[c.cell]+.001)
                    val = np.clip(val,0,1)
                rgb = [np.sin(val),np.sin(val+0.5),np.cos(val-0.5)]
                
            col = np.clip(rgb, 0, 1)
            rgbs[int(c.cx), int(c.cy)] = col

            ax.plot(c.dx, c.cy, marker='h', markersize=24, alpha=.9, color=col)
            
        if label_df is not None:
            cells = sspc.get_cxcy_df().set_index('cell')
            lab_df = pd.merge(label_df,cells,left_index=True,right_index=True)

            for cell, c in lab_df.iterrows():
    
                rgb = rgbs[int(c.cx), int(c.cy)]
                l_color = 'white' if sum(rgb[0:2]) < 0.9 else 'black'
    
                l_text = '\n'.join([s[0:5] for s in c.label.split('_')])
                dy = c.cy
                dx = c.cx if (c.cy % 2 == 0) else c.cx+.5
                dx = dx - .5
                dy = dy - .5
                
                # place the text a bit better
                
                n_textlines = len(l_text.split('\n'))
                y_off = (3-n_textlines)*.05
                dy += y_off
                dx += .08
    
                ax.text(dx, dy, l_text,
                        color=l_color,
                        fontsize=5)

        #legends.append('Voronoi cell')
        
        if not textoff:
            fig.suptitle("SOM Voronoi cell coordinates (IC 0,1,2) in RGB color values" +
                         "\n with each of relation-role density peaks labeled.", fontsize=8)
            ax.set_title('Relations: [adaptation, location, taxongroup, free]\n' +
                         'Roles: [topic, comment]', fontsize=7)
    
        ax.legend(legends, loc='center', bbox_to_anchor=(1.2, .8),
                  ncol=2, fancybox=True)

        return fig, axs

    def __init__(self):
        ce_data_df = self.load_ce_data()
        self.init_ica_datapoints(ce_data_df)
        self.get_som_model()
        
    def compute_idata_bmus(self):
        ica_data = self.idata_df
        n_dim = ica_data.shape[1]
        
        fn = 'data/SOM_%d_idata_BMUs.pkl' % n_dim

        try:        
            self.idata_bmus = joblib.load(fn)
            print("restored "+fn)
            
        except:
            print("Finding BMUs "+fn)
            self.idata_bmus = pd.Series(
                [self.my_som.find_BMU(ica_data.iloc[n].array)
                 for n in range(len(ica_data))],index=ica_data.index)
            
            joblib.dump(self.idata_bmus, fn)
            print("stored "+fn)
            
        return self.idata_bmus
        
    def som_token_bmus(self):
        data_bmus = self.compute_idata_bmus()
        ica_data = self.idata_df
       
        som_tokens_df = pd.DataFrame([{
            'cell' : "%02d_%02d" % (cx, cy), 
            'rel_role' : rel_role, 
            'tokenid' : tokenid, 
            'cx' : cx, 'cy' : cy, 
            'group' : sgroup}
            for (cx, cy), (rel_role, sgroup, tokenid)
            in zip(
                data_bmus, ica_data.index)]
        ).set_index('tokenid')
        
        return som_tokens_df
        
    def category_population_analysis(self,som_tokens_df : pd.DataFrame,cat_spec : str):
            '''
                SOM population statistics 
                Unit cell rel_role population             
                KL divergence per relation-role category
                Token binning in a lattice
            '''
            import scipy.special as sp

            print("category_population_analysis (%s)"%cat_spec)
                        
            xdf = pd.crosstab(som_tokens_df.cell,
                              som_tokens_df[cat_spec])

            x_tot = nrm(xdf.sum(axis=1))
            som_cat_pop_H = pd.DataFrame(
                {"kldiv_"+rr: sp.rel_entr(nrm(xdf[rr]), x_tot)
                 for rr in xdf.columns},
                index=list(xdf.index))

            som_cat_pop_H.index.name = 'cell'

            #lw, lh, ld = somv.SOM.shape

            cxcy_df = self.get_cxcy_df()
            

            som_cat_pop_df = (cxcy_df.merge(xdf, on='cell', how='left').
                         set_index('cell').fillna(0))
            print("stats: som_cell_popdist ready!")

            return som_cat_pop_df, som_cat_pop_H, som_tokens_df

        
# In[test]

sspc = SomSpace()
        

# In[helper functions]

def make_label_df(kldiv_df: pd.DataFrame):
    data = [
        pd.DataFrame({
            'score': kldiv_df[ss].sort_values(ascending=False).head(20),
            'label': ss.replace('kldiv_','')})
        for ss in kldiv_df.columns]

    label_df = pd.concat(data, axis=0).sort_index()
    label_df.sort_values('score', ascending=False, inplace=True)
    label_df = label_df[~label_df.index.duplicated(keep='first')]
    return label_df

def make_density(occ_df : pd.DataFrame):
    density = np.array(occ_df)[:,4:].sum(axis=1)
    density = density-np.min(density)
    return pd.Series(density / (np.max(density)+.01),index=occ_df.index)



# In[SOM for relation-role distribution]
# relation-role SOM graph

bmus = sspc.som_token_bmus()

n_occs,kl_divs,toks=sspc.category_population_analysis(bmus,'rel_role')

rr_kl_divs = kl_divs

lab=make_label_df(kl_divs)
occ=make_density(n_occs)


fig,ax=sspc.annotate_lattice(label_df=lab,textoff=False)
fig.savefig('plots/som/som-all-relrole.png')
fig

# In[SOM for comments]
# SOM graph

bmu_c = bmus[bmus.rel_role.str.contains('comment')].copy()
bmu_c['rel']=bmu_c.rel_role.str.replace('_comment','_comm.')

n_occs,kl_divs,toks=sspc.category_population_analysis(bmu_c,'rel')

lab=make_label_df(kl_divs)
occ=make_density(n_occs)

fig,ax=sspc.annotate_lattice(color=occ,label_df=lab)
fig.savefig('plots/som/som-comment-rel.png')
fig

# In[SOM for topics]
# SOM graph

bmu_t = bmus[bmus.rel_role.str.contains('topic')].copy()

n_occs,kl_divs,toks=sspc.category_population_analysis(bmu_t,'rel_role')

lab=make_label_df(kl_divs)
occ=make_density(n_occs)

fig,ax=sspc.annotate_lattice(color=occ,label_df=lab,textoff=True)
fig.savefig('plots/som/som-topic-rel.png')
fig

# In[SOM for location comment]
# SOM graph

bmu_lc = bmus[bmus.rel_role=='location_comment'].copy()

terms=db.tokentag_df.term.loc[bmu_lc.index]

bmu_lc['terms']=[t[0:5]+'_'+t[5:10]+'_'+t[10:15] for t in terms]

n_occs,kl_divs,toks=sspc.category_population_analysis(bmu_lc,'terms')

lab=make_label_df(kl_divs)
occ=make_density(n_occs)

fig,ax=sspc.annotate_lattice(color=occ,label_df=lab,textoff=True)
fig.savefig('plots/som/som-location_comment.png')
fig

# In[DENDROGRAM]

def make_dendrogram(token_list : list, sspc : SomSpace):

    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib.figure import Figure


    fig = Figure(figsize=(2, 2), tight_layout=True)
    fig.suptitle('linkage of token cluster')
    axs = fig.subplots(subplot_kw=dict(yticks=[]))

    try:


        fig = Figure(figsize=(12, 4),
                     dpi=90, tight_layout=True)
        axs = fig.subplots(subplot_kw=dict(yticks=[]))
        fig.suptitle('Sample Similarity dendrogram',
                     fontsize=24)

        ax0 = axs
        
        cx_df = pd.DataFrame() # TODO ADD DATA!!!!!!!!

        clust_x = np.array(cx_df)[:, :-1]
        clust_x_lnk = linkage(clust_x, 'average', metric='cityblock')
        labeltext = [ex.short_token_ctx(
            ex.pretty_token(x).ctx) for x in cx_df.index]

        dn = dendrogram(clust_x_lnk, orientation='left', labels=(
            labeltext), ax=ax0, leaf_font_size=14)
    except:
        print("---make_dendrogram error---")
    return fig




# In[arcpath margin distribution]

_ap_margin_frq = None

def compute_arcpath_margin(ap_df : pd.DataFrame, normalize : bool = True):
    '''
    Compute the Q margin distribution of arcpaths

    Parameters
    ----------
    normalize : bool, optional
        Should the result be normalized to sum to 1. The default is True.

    Returns
    -------
    Series
        the value counts of arcpaths, normalized if asked
    '''
    global _ap_margin_frq 
    if _ap_margin_frq is None:
        ap_margin_frq = ap_df.arcpath.value_counts()
        _ap_margin_frq  = ap_margin_frq 
        
    return nrm(_ap_margin_frq) if normalize else _ap_margin_frq
    


def fetch_token_arcs():
    '''
    Read all the token arcs in separate sample groups.
    If not prepared yet, then run the refresh_token_arcs_tsv method
    

    Returns
    -------
    token_arcs_df : DataFrame
        DataFrame of columns: token relrole arcpath lemma upos widx

    '''
    
    token_arcs_df = ddf.read_csv('plots/*/token_arcs.tsv',sep='\t',assume_missing=True).compute()
    token_arcs_df.drop(columns=['token.1'],inplace=True)
    return token_arcs_df

# In[KL-div cell arcpath analysis]


token_arcs_df = fetch_token_arcs()
ap_margin_q = compute_arcpath_margin(token_arcs_df )
ap_margin_frq = compute_arcpath_margin(token_arcs_df,False)

# In[KL-div cell analysis]


def top_cell_for_cat(kldivs, catkey='location_comment'):
    return kldivs['kldiv_%s' % catkey].sort_values(ascending=False).head(5)


def cell_ap_analysis1(token_arcs_df: pd.DataFrame,
                     ap_margin_q: pd.Series,
                     bmus: pd.DataFrame,
                     cellid: str,
                     top_n=5):

    import scipy.special as spspec

    bmu_tokenids = bmus[bmus.cell == cellid].index

    global bmu_arcs

    bmu_arcs = token_arcs_df[token_arcs_df.token.isin(bmu_tokenids)]

    if (len(bmu_arcs) < 10):
        print("Warning! Only %d BMU arcs - KL Div analysis unstable" %
              len(bmu_arcs))

    p_dist = nrm(bmu_arcs.arcpath.value_counts())

    df = pd.DataFrame({'p': p_dist, 'q': ap_margin_q}).fillna(0)

    el_kldiv = spspec.kl_div(df.p, df.q)
    df = pd.DataFrame({'N': bmu_arcs.arcpath.value_counts(),
                       'el_kldiv': el_kldiv.round(5)}).\
        sort_values('N', ascending=False).\
            dropna()
    return df


def cell_ap_analysis(token_arcs_df: pd.DataFrame,
                     relrole : str,
                     ap_margin_q: pd.Series,
                     bmus: pd.DataFrame,
                     cellid: str,
                     top_n=5):
    a_df = token_arcs_df[token_arcs_df.relrole==relrole]
    df1 = cell_ap_analysis1(a_df, ap_margin_q, bmus, cellid)
    dfx = cell_ap_analysis1(token_arcs_df, ap_margin_q, bmus, cellid)
    df1.columns = ['cat_N','Df_cat']
    dfx.columns = ['cell_N','Df_cell']
    return pd.concat([df1,dfx],axis=1).head(top_n)
    

# In[compose a top 3]    

top_cells = top_cell_for_cat(rr_kl_divs, 'location_comment')
dfs = []
for cellid in top_cells.index[0:3]:

    cell_anal_df = cell_ap_analysis(token_arcs_df,'location_comment',ap_margin_q,bmus,cellid,5)
    dfs.append(cell_anal_df.assign(cell=cellid))
    
anal_df = pd.concat(dfs)
anal_df = anal_df.reset_index().set_index(['cell','arcpath']).reset_index()

# anal_df.to_csv('plots/som/arcpath_loc_com_anal.csv',index=False)

# Run for your interest:
relrole = 'location_comment'
a_df = token_arcs_df[token_arcs_df.relrole==relrole]
cat_bmus = bmus[bmus.index.isin(a_df.token)]

db.tokentag_df.loc[cat_bmus[cat_bmus.cell==top_cells.index[0]].index].term.value_counts().head()
db.tokentag_df.loc[cat_bmus[cat_bmus.cell==top_cells.index[1]].index].term.value_counts().head()
db.tokentag_df.loc[cat_bmus[cat_bmus.cell==top_cells.index[2]].index].term.value_counts().head()


























