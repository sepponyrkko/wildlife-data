#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 08:41:38 2025

@author: seppo
"""

# Some useful libraries

import pandas as pd
import numpy as np
import dask.dataframe as ddf

import matplotlib.pyplot as plt
from matplotlib import figure

from sample_group import SG

import extools as ex
import databox as db
import syn_tree
import sample_group_stats as sgstat

import joblib

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import scipy.special as special

#from scipy.spatial import distance_matrix


# In[ds]

def nrm(x):
    '''
    normalize vector x by dividing with the sum(x)
    '''
    return x/(0.000001 + sum(x))

class DelexSpace:
    
    # These are the methods for updating the data sets.
    # We will load the prepared data below (faster).

    @staticmethod
    def refresh_token_arcs_tsv():
        '''
        This block computes all the token arcs in separate sample groups.
        Run only once after updating the data set.
        '''
        
        s_groups = SG._get_group_list()
        # s_groups = ['Life#life'] # for updating individual groups
    
        for sg_id in s_groups:
    
            print (sg_id)
            my_sg = SG(sg_id)
            b_td = my_sg.get_tokendict()
            zzzs = []
            rrtokendict = my_sg.get_rr_token_dict()
    
            for rr,c_df in rrtokendict.items():
                tokens = c_df.index.droplevel(0)
                zz=[b_td[tok_id]['t_arcs_ex_term'].assign(token=tok_id) for tok_id in tokens]
                zz_df = pd.concat(zz).reset_index(drop=True).assign(relrole=rr)        
                zzzs.append(zz_df)
    
            f_path = 'plots/'+my_sg.group_name+'/token_arcs.tsv'
            zzz_df = pd.concat(zzzs)
            zzz_df = pd.DataFrame(zzz_df,columns=list(zzz_df.columns[-2:])+list(zzz_df.columns[:-2]))    
            zzz_df.to_csv(f_path,sep='\t',index=False)
            print("Written %s" % f_path)
        
    @staticmethod
    def preprocess_feature_stat_pq():
        '''
        Preprocess the per-category feature frequency distribution.
        Store the result in a parquet file.
        '''
        
        feat_d_df = pd.read_csv('plots/feature_Df_n_per_cat.tsv',sep='\t')
        feat_d_df.drop(columns=['Unnamed: 0'],inplace=True)
        feat_d_df.set_index('feat',inplace=True)
        feat_d_df['arcpath']=feat_d_df.index.str.replace('\s.*','',regex=True)
        feat_d_df.to_parquet('plots/feature_Df_n_per_cat.pq')
    
    
    @staticmethod
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
    
    
    #token_arcs_df = fetch_token_arcs()
    #token_arcs_df.sample(5)        
    
    
    def get_rr_plain_arcpaths(self,n_lim=30):
        '''
        This selects the "most specific" arcpaths, 
        coined as "de-lexical features"
        by their computed element-wise KL divergence **Dx**
        '''

        feat_d_df = self.feat_d_df
    
        ap_df = feat_d_df[feat_d_df.index.str.contains('...',regex=False)]
        ap_df1 = ap_df[ap_df.relrole.isin(self.rel_role_set)]
        
        p1 = nrm(ap_df1.groupby('arcpath').n.sum())
        q1 = nrm(ap_df.groupby('arcpath').n.sum())
        ap_d = pd.DataFrame({'p':p1,'q':q1}).fillna(0).sort_values('q')
        ap_d['Df'] = special.rel_entr(ap_d.p,ap_d.q)
        
        aps = ap_d.sort_values('Df',ascending=False).head(n_lim)
    
        # Create a series delex_D of the feature-elementwise divergence Df
        # Limit to top 30 arcpaths with highest Df
        
        self.delex_D = aps.Df
        
        
    def get_xtab_dfs(self,ap_list):
        '''
        cross-tabulation between arcpath-features and tokens.
        '''
        
        # cross-tab of all token-arcpaths (in ap_list)
        
        token_arcs_df = self.token_arcs_df
        rel_role_set = self.rel_role_set
    
        arcs = token_arcs_df[token_arcs_df.arcpath.isin(ap_list)]    
        ap_cross_ref_df = pd.crosstab(arcs.token,arcs.arcpath)
    
        # cross-tab of RR token-arcpaths (in ap_list)
        
        # rr_tok_arc_df = token_arcs_df.groupby("relrole").get_group(my_rel_role)
        
        rr_tok_arc_df = token_arcs_df[token_arcs_df.relrole.isin(rel_role_set)]
        rr_tok_arc_df = rr_tok_arc_df[rr_tok_arc_df.arcpath.isin(ap_list)]
        
        ap_cross_df = pd.crosstab(rr_tok_arc_df.token,rr_tok_arc_df.arcpath)
    
        self.ap_cross_df, self.ap_cross_ref_df = ap_cross_df, ap_cross_ref_df
        
    
    def fit_ica(self):
        '''
        Create an ICA projection from the frequency table.

        Returns
        -------
        None.

        '''        

        from sklearn.decomposition import FastICA
        
        print("FastICA fit to cross tabulation matrix ap_cross_df")
        
        ica_tr = FastICA(n_components=5,max_iter=2048,random_state=42)
        ica_tr.fit(self.ap_cross_df)

        print("FastICA ready")
        
        self.ica_tr = ica_tr
        
        plot_df = pd.DataFrame(ica_tr.transform(self.ap_cross_df)[:,:])
        #plot_df.rename(columns={plot_df.columns[-1]:'e'},inplace=True)
        
        self.plot_df = plot_df
        
        comps_df = pd.DataFrame(ica_tr.components_.round(2),columns=self.ap_cross_df.columns).T
        comps_df.sort_values(0,inplace=True)
        self.comps_df = comps_df
        
        print("== ICA components sample ==")
        print(pd.concat([comps_df.head(5),comps_df.tail(5)]))
        print("===========================")
        
    def ica_grid(self,compact=False):
        ica_tr = self.ica_tr
        plot_df = self.plot_df
        comps_df = self.comps_df

        from matplotlib import figure

        fig = figure.Figure(figsize=(16 if compact else 16, 4 if compact else 12), dpi=90)
        fig.set_tight_layout(20)

        DD = ica_tr.n_components-1  # -1 for cross projection
        
        DD = 3 if compact else DD

        axs = fig.subplots(1 if compact else DD, DD, sharey=False, sharex=False)

        legends = []
        colors = ['orange', 'green', 'red', 'magenta', 'brown',
                  'blue', 'maroon', 'pink', 'brown', 'purple']

        for Dy in range(DD if compact else 1, DD+1):
            for Dx in range(0, Dy):
                ax = axs[Dx] if compact else axs[Dy-1, Dx]
                
                Dy = Dx + 1 if compact else Dy

                if Dx == 0 or compact:
                    ax.set_ylabel("IC %d" % Dy)

                if Dy == DD or compact:
                    ax.set_xlabel("IC %d" % Dx)

                xx = np.array(plot_df[[Dx, Dy]])
                ax.plot(xx[:, 0], xx[:, 1], '.', alpha=0.4, color='lightblue')


                coli = 0
                arrows_n = 5
                arrow_df = (comps_df[[Dx, Dy]])*10
                arrow_df.columns = ['x', 'y']
                show_arr_list = ((arrow_df**2).sum(axis=1).
                                 sort_values(ascending=False).head(arrows_n).index)
                arrow_df = arrow_df.loc[show_arr_list].sort_values('y')
                
                legends = ['tokens']


                for arr in list(arrow_df.index):
                    (x, y) = arrow_df.loc[arr]
                    ax.annotate('', (x, y), xytext=(0, 0), arrowprops={
                                'width': 0.02, 'headwidth': 2, 'color': colors[coli]})
                    legends.append(arr)
                    
                    (x, y) = 1.2*x, 1.2*y # texts far away
                    
                    xoff = 0.5*len(arr)
                    yoff = (coli-2)*1.25
                    
                    ycorr , xcorr = y+yoff , x-xoff                    
                    
                    ax.text(xcorr, ycorr, arr, fontsize=10)
                    
                    ax.plot(1.25*xcorr+2*np.sign(xcorr), 
                            1.25*ycorr, 
                            '-', color=colors[coli])
                    coli += 1
                ax.legend(legends)
                ax.plot([-5, 5, 0, 0], [0, 0, -5, 5], ' ')

            for Dx in range(Dy, DD):
                ax = axs[Dy-1, Dx]
                ax.set_axis_off()

        if compact:
            fig.suptitle('Delexical feature ICA model of the Wildlife corpus tokens. Category: '+self.rel_role)
        else:
            fig.suptitle('Relation-role: '+self.rel_role +
                          '\n De-lexicalized token distribution in ICA space\n' +
                          'Bi-plot of features with highest vector norm.')

        return fig, axs

    def stats_comps_w_Df(self):
        '''
        Compose a comparison table on how the features' weights "w" 
        align with significance estimates "Df".
        
        The Df is the element-wise KL divergence.        
        The "w" are ICA component norms.
        
        The result dataframe goes in comps_w_Df.
        
        '''
        comps_df = self.comps_df
        ap_cross_df = self.ap_cross_df
        delex_D = self.delex_D       
    

        compsw = np.sqrt((comps_df**2).sum(axis=1))
        compsw.name = 'weight'
        compsw.sort_values(ascending=False,inplace=True)
        
        comps_w_Df = pd.concat([compsw,delex_D.loc[compsw.index]],axis=1)
        comps_w_Df['Nf']=ap_cross_df.sum().loc[comps_w_Df.index]
        
        print(comps_w_Df.head(10))
        
        self.comps_w_Df = comps_w_Df
        
    def chart_comps_w_Df(self,top_N = 20):
        
        maxcomps_w_Df_r = self.comps_w_Df.sort_values('weight',ascending=True)
        maxcomps_w_Df_r = maxcomps_w_Df_r.tail(top_N)
        
        fig_ = figure.Figure(figsize=(6,top_N*.25),dpi=90)
        fig_.tight_layout()
        
        axs_ = fig_.subplots(1,1,sharey=False,sharex=False)
        ax = axs_#[0]
        
        ax.barh(width=maxcomps_w_Df_r.weight,
                 y=maxcomps_w_Df_r.index,
                 height=.4)
        
        ax.barh(width=maxcomps_w_Df_r.Df*100,
                 y=pd.Series(range(len(maxcomps_w_Df_r.index)))-.2,
                 height=.4)
        
        # 
        # Move this to the other chart
        # 
        # for i in range(len(maxcomps_w_Df_r.index)):
        #     ax.text(0.8,0.35+i,'N = %d'%maxcomps_w_Df_r.Nf.iloc[i],
        #            fontsize=8)
        
        ax.legend(['ICA component\nweight','elementwise\nKL div x100'],
                 loc = 'lower right',
                 bbox_to_anchor=(1,.1))
        
        ax.set_xticks(np.arange(0,3,.5))
        ax.set_title("ICA delexicalized features\n in the "+self.rel_role+" relation-role category\n"+
                     "sorted by feature norm (weight)" )
        
        # Also show with a table
        
        
        stats_df = self.comps_w_Df[['weight','Df','Nf']].reset_index().round(4).head(top_N)
        # stats_arr = np.array(stats_df)
        # ax = axs_[1]
        
        # table = ax.table(stats_arr,
        #          loc='center',
        #          colLabels=stats_df.columns,
        #          fontsize=22#,
        #          #rowLabels=stats_df.index#,
        #          #edges='vertical'
        #          )
        
        tbl_title = ('The component vector norm (weight) \n and the frequency-based divergence in'+
                     self.rel_role+' relation-role cateogory,\n'+
                     str(len(stats_df))+' features with highest vector norms (weights)')
        
        print(tbl_title)
        print(stats_df.head(10).to_markdown())
        fn = 'plots/delex/comps_w_Df-'+self.rel_role+'.csv'
        stats_df.to_csv(fn, index=False)
        print("CSV WRITTEN TO "+fn)        
        
                     
        return fig_,ax
    
    def build_dendro_comps(self):
        '''
        make normalized vectors for cosine similarity dendrogram
        '''    
        comps_df = self.comps_df
        comps_w = np.sqrt((comps_df**2).sum(axis=1))
        comps_w.name = 'weight'
        
        comps_nrm = pd.DataFrame(np.divide(np.array(comps_df).T,np.array(comps_w)).T,index=comps_w.index)
        
        dendro_comps_df = comps_nrm.loc[self.comps_w_Df.index] # selected
        self.dendro_comps = dendro_comps_df
        return dendro_comps_df
    
    def plot_dendro(self):
        
        dendro_comps = self.dendro_comps # init at the build_dendro_comps 
        
        dists2= pdist(np.array(dendro_comps),metric='cosine')
        Z = hierarchy.linkage(dists2, 'ward')
        
        fig = figure.Figure(figsize=(4,7),dpi=90)
        axs = fig.subplots(1,1,sharey=False)
        ax = axs
        
        Z = hierarchy.linkage(dists2+.05, 'median')
        #Z = hierarchy.linkage(dists2+.05, 'complete')
        #Z = hierarchy.linkage(dists2, 'single')
        
        #plt.figure()
        dn = hierarchy.dendrogram(Z,ax=ax,
                                  orientation='left',
                                  labels=dendro_comps.index,
                                  color_threshold=.5)
        ax.axes.set_xticks(np.arange(2.0,0,-.5))
        ax.axes.set_title('Hierarchical clustering of delexicalized features \n'+
                         self.rel_role+'\n'
                         '(based on cosine distance of feature vectors)')
        ax.axes.set_xlabel('Cosine distance component vectors [0.0...2.0]\n d(X,Y)= 1 - [sum_i (Xi Yi) / (|X||Y|)]')
        
        self.dn = dn        
        self.dn_leaves = pd.DataFrame(dn['leaves_color_list'],dn['ivl'],columns=['leaf_group'])
        
        return fig,ax
        


        
        

    
        
    
    
    def __init__(self,rel_role):
        '''
        Init the object. 

        Parameters
        ----------
        rel_role : str
            The relation-role category to build upon

        Returns
        -------
        None.
        '''
        
        
        
        self.token_arcs_df = DelexSpace.fetch_token_arcs()
        self.feat_d_df=pd.read_parquet('plots/clusters/feature_Df_n_per_cat.pq')

        all_rel_roles = list(self.token_arcs_df.relrole.unique())
        

        if ',' in rel_role:
            self.rel_role_set = rel_role.split(',')
        else:
            self.rel_role_set = [r for r in all_rel_roles 
                                 if (rel_role in r) 
                                 and not r.startswith('free_')]
        
        self.rel_role = rel_role.replace(',','_and_')

        
        self.get_rr_plain_arcpaths() # load self.delex_D        
        ap_list = self.delex_D.index

        self.get_xtab_dfs(ap_list)
        self.fit_ica()
        
        
        print("===================")
        print("sample of token_arcs_df")
        print(self.token_arcs_df.sample(5))
        
        print("===================")
        print("sample of feat_d_df")
        print(self.feat_d_df.groupby('relrole').sample(2))
        print("===================")
        
        print("'Most significant' features in %s" % self.rel_role)
        print(self.delex_D.head(5)) # with D
        print("===================")
        
        self.stats_comps_w_Df()
        self.build_dendro_comps()

        pass
    
    
    def combo_poisson_test(self, ap_a : str, ap_b : str):
        '''
        Compute the Poisson mean E-test for the given feature pair a,b

        Parameters
        ----------
        ap_a : str
            feature a. E.g. 'obj<'.
        ap_b : TYPE, optional
            feature b. E.g. 'nmod>conj>'.

        Returns
        -------
        p_stat : float
            E-test statistic.
        p_pval : float
            E-test p-value.
        k1 : TYPE
            The number of a,b events in this category.
        k2 : TYPE
            The total number of a,b events in all categories.

        '''
        
        a_df,b_df = self.pt_a_df, self.pt_b_df
        n1,n2= self.pt_n1, self.pt_n2
    
        k1,k2 = np.sum(a_df[ap_a] * a_df[ap_b]), np.sum(b_df[ap_a] * b_df[ap_b])
    
        import scipy.stats as sst 
        p_stat,p_pval=sst.poisson_means_test(k1,n1,k2,n2)
        
        if p_pval<0.1:
            print(("%25s + %-25s (%4d %-4d)" % (ap_a, ap_b, k1, k2)) + 
                  (" E-test: %1.3f p-val: %1.5f "%(p_stat,p_pval)))
        
        return p_stat,p_pval,k1,k2
    
    def init_poisson(self):
        ap_pairs = []
        my_leaves = self.dn_leaves
        
        self.pt_a_df = (self.ap_cross_df>0)*1
        self.pt_b_df = (self.ap_cross_ref_df>0)*1
        
#        self.pt_n2 = len(self.token_arcs_df.token.unique())
        self.pt_n1 = len(self.token_arcs_df[self.token_arcs_df.relrole.isin(self.rel_role_set)].token.unique())
        self.pt_n2 = len(self.token_arcs_df[~self.token_arcs_df.relrole.isin(self.rel_role_set)].token.unique())
        
        print("sample group sizes (n1, n2): %d , %d"%(self.pt_n1,self.pt_n2))
        
        
        for lg,leaves in my_leaves.groupby(my_leaves.leaf_group):
            my_aps = list(leaves.index.unique())
            for i,ap_a in enumerate(my_aps):
                for ap_b in my_aps[:i]:
                    if not (ap_a.startswith(ap_b) or ap_b.startswith(ap_a)):
                        ap_pairs.append((ap_a,ap_b,lg))
                        
        self.ap_pairs = ap_pairs
        

# In[pair measurement]
    

def poisson_testing(ds):
    fig,axs = ds.plot_dendro()
    ds.init_poisson()
    
    x=[]
    for ap_pair in ds.ap_pairs:
        x.append(ds.combo_poisson_test(ap_pair[0],ap_pair[1]))
    x_df = pd.DataFrame(x,columns=['stat','p_val','k1','k2'],index=ds.ap_pairs)    
    x_df.p_val = x_df.p_val.round(5)
    
    # sane index
    
    ix_df = pd.DataFrame([[a,b,c] for (a,b,c) in x_df.index],columns=['a','b','leaf_group'],index=x_df.index)
    x_df = pd.concat([x_df,ix_df],axis=1).reset_index(drop=True)
    
    # fresh sort    
    x_df = x_df[x_df.stat != 0].copy()
    x_df.sort_values('p_val',inplace=True)    
    return x_df.reset_index(drop=True)

def poisson_table(ds):
   
    x_df = poisson_testing(ds)
    x_tbl = x_df.groupby('leaf_group').head(3)
    x_tbl = x_tbl[['a','b','k1','k2','p_val']]
    x_tbl.p_val = list(x_tbl.p_val.apply(lambda x:'%0.05f %s'%(x,'*'*int(-np.log10(x/.5+.001)))))
    x_tbl.columns=['feature a','feature b',
                   'occ (N1=%d)'%ds.pt_n1,
                   'ref (N2=%d)'%ds.pt_n2,
                   'E-test p-value']
    x_tbl.to_csv('plots/delex/poisson_pairs_'+ds.rel_role+'.csv',index=False)
    return x_tbl


# In[Run]


# Now test it works

if __name__ == '__main__':
    
    rr = 'topic'
    rr = 'adaptation_comment'
    rr = 'location_comment'
    rr = 'taxongroup_comment'

    ds = DelexSpace(rr)

# In[Run all]

# Now test it works

if __name__ == '__main__':
    
    for rr in ['topic', 'adaptation_comment', 'location_comment', 'taxongroup_comment']:

        ds = DelexSpace(rr)
        fig,axs=ds.ica_grid(True)
        fig.savefig('plots/delex/ica_grid_v2_'+ds.rel_role+'.png',
                    bbox_inches='tight',transparent=None)
        



# In[demo ica grid]
if __name__ == '__main__':

    fig,axs=ds.ica_grid(True)
    fig.savefig('plots/delex/ica_grid_v2_'+ds.rel_role+'.png',
                bbox_inches='tight',transparent=None)
    fig


# In[demo weight bar plot]

if __name__ == '__main__':
    fig,axs=ds.chart_comps_w_Df()
    fig.savefig('plots/delex/comps_w_Df_'+ds.rel_role+'.png',
                bbox_inches='tight',transparent=None)
    fig

# Alternative:
    
# plt.plot(list(ds.comps_w_Df.weight),list(ds.comps_w_Df.Df),'o')


# In[demo dendro plot]

if __name__ == '__main__':
    fig,axs=ds.plot_dendro()
    fig.savefig('plots/delex/dendro_'+ds.rel_role+'.png',
                bbox_inches='tight',transparent=None)
    fig



# In[tabular combo with poisson mean testing]

    
if __name__ == '__main__':
    x_tbl=poisson_table(ds)
    print(x_tbl.to_markdown(index=False))

# Relation-Role count demo

if __name__ == '__main__':
    tdf = ds.token_arcs_df
    token_relrole = tdf[['token','relrole']].drop_duplicates()
    token_relrole.relrole.value_counts()

