#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample group operations
Created on Sat Dec 28 10:36:21 2024

@author: seppo
"""

import os
import re

import numpy as np
import pandas as pd
import scipy.stats as st
from collections import defaultdict


# import databox as db
# import importlib
# ex.get_ce(['Fro-2392_536_predator-9'])

# In[vec_ops]

def nrm(x):
    return x/sum(x)

def nrm2(x):
    return np.round(x / (0.000001 + np.sqrt(sum(x*x))),5)


def cos_dst(x,y):
    return np.inner(nrm2(x),nrm2(y)).round(3)
# In[xx]

class SG:
    '''
    Sample group operations    
    '''
    
    sample_groups = []
    
    @staticmethod
    def _get_group_list():
        
        if len(SG.sample_groups) < 1:
            SG.sample_groups_df = pd.read_csv("analysis_groups.tsv", sep='\t')
            SG.sample_groups = list(SG.sample_groups_df.group.sort_values().unique())
        return SG.sample_groups
    
    
    # Initializer

    def __init__(self, g: str):
        '''
        Init the (SG) sample group object

        Parameters
        ----------
        g : str
            the ID of the group, for instance 'Life#life'
        '''
        
        assert (g in SG._get_group_list())
        
        self.group_uri = g
        self.group_name = re.sub(r'\#', '__', self.group_uri)
        self.g_path = "data/%s/" % self.group_name
        self.tokendict = None
        
        print("My URI: %s" % self.group_uri)
        print("My path: %s" % self.g_path)
        
        g_df = SG.sample_groups_df[SG.sample_groups_df.group == self.group_uri]
        self.taxons = list(g_df.taxon)
        print("My taxons: (%d) %s ..." %(len(self.taxons),self.taxons[0:3]))
        os.makedirs(self.g_path, exist_ok=True)
        
        self.restore_datapoints()
        
        self.build_ica()
        

    def build_dataset(self):
        import extools as ex

        self.ce_data_df = None
        self.pair_df = None

        concept_uris = self.taxons

        enc_dfs = []
        self.pair_dfs = dict()

        for rel in ['location', 'adaptation', 'taxongroup']:
            base_enc1, null_enc1, ref_enc1, self.pair_dfs[rel] = \
                ex.select_datapoints(concept_uris, rel)

            null_enc1['rel_role'] = 'free_topic'
            base_enc1['rel_role'] = '%s_topic' % rel
            ref_enc1['rel_role'] = '%s_comment' % rel
            enc_dfs.extend([null_enc1.reset_index(),
                            base_enc1.reset_index(),
                            ref_enc1.reset_index()])

        self.ce_data_df = pd.concat(enc_dfs).drop_duplicates()
        self.ce_data_df.set_index(['rel_role', 'token_id'], inplace=True)

        # remove excessive tokens from group 'free_topic'

        tokens = self.ce_data_df.index.get_level_values(1)
        tokens = tokens[tokens.duplicated()]
        tokens = set(
            self.ce_data_df.loc['free_topic'].index).intersection(tokens)
        tokens = [('free_topic', x) for x in tokens]

        print('exessive free_topic tokens : %d' % len(tokens))
        self.ce_data_df.drop(index=tokens, inplace=True)

        # build self.pair_df

        xs = []
        for k, i in self.pair_dfs.items():
            x = i.sort_values('tagscore', ascending=False)
            x['relation'] = k
            xs.append(x)
        self.pair_df = pd.concat(xs).set_index(
            ['relation', 'btoken', 'rtoken'])
        self.pair_df = self.pair_df.groupby(self.pair_df.index.names).head(1)
        
        
        import joblib      
        
        os.makedirs(self.g_path, exist_ok=True)
        joblib.dump(self.ce_data_df, self.g_path+'/ce_data_df.pkl')
        joblib.dump(self.pair_df, self.g_path+'/pair_df.pkl')

        self.ce_data_df.round(3).to_csv(self.g_path+"/data_points.tsv", sep='\t')
        self.pair_df.round(3).to_csv(self.g_path+"/pair_table.tsv", sep='\t')

        print("datapoint statistics:")
        for (rrid, rrdf) in self.ce_data_df.groupby('rel_role'):
            print(rrid, len(rrdf))

        print(self.ce_data_df.groupby('rel_role').head(2))

    def get_tokendict(self):
        if self.tokendict:
            return self.tokendict
        
        import joblib
       
        try:
            self.tokendict = joblib.load(self.g_path+'/token_dict.pkl')
            print("=== get_tokendict Restored SG token dictionary "+self.g_path+'/token_dict.pkl')
        except:
            import extools as ex            
            self.tokendict = ex.build_token_dictionary(self.ce_data_df.index.droplevel(0))
            joblib.dump(self.tokendict, self.g_path+'/token_dict.pkl')
            
        return self.tokendict
    
    def restore_datapoints(self):        
        import joblib
        try:
            self.ce_data_df = joblib.load(self.g_path+'/ce_data_df.pkl')
            self.pair_df = joblib.load(self.g_path+'/pair_df.pkl')
            print("Loaded " + self.g_path+'/pair_df.pkl')
        except:
            self.build_dataset()
            
        #self.pair_dict = {g:d for (g,d) in self.pair_df.groupby('relation')}
        return self.pair_df

            

    def build_ica(self, D_space : int = 8):
        '''
        Fit the ICA dimension reduction model. 
    
        Parameters
        ----------
        D_space : int, optional
            dimension of the ML space. The default is 8.
        '''
        from sklearn.decomposition import FastICA
        
        import joblib
        try:
            self.ica_tf = joblib.load(self.g_path+"/ica_tf.pkl")
            
            print("ica_tf.pkl restored")
        
        except:
    
            self.ica_tf = FastICA(n_components=D_space, 
                                      random_state=42,
                                      tol=0.1,
                                      max_iter=200)
        
            self.ica_tf.fit(self.ce_data_df)
        
            joblib.dump(self.ica_tf,self.g_path+"/ica_tf.pkl")
            
            print("ica_tf.pkl saved")
            
        
        x_df = self.ce_data_df
        data = self.ica_tf.transform(x_df)
        
        icx = pd.DataFrame(data,index=x_df.index)
        
        self.ic_rr_token_df = icx
        
        x = icx.droplevel(0)
        self.ic_token_df = x[~x.index.duplicated(keep='first')].copy()        
        
    def get_rr_token_dict(self):
        return {g:d for (g,d) in self.ic_rr_token_df.groupby('rel_role')}
        
    def build_gmm_dict(self):
        '''
        initialize GMMs with token context encodings
        also form a GMM cluster data frame
        '''
        import joblib, os
        filename=self.g_path+"/gmm_dict.pkl"
        try:
            if os.path.exists(filename):
                self.gmm_dict = joblib.load(filename)            
                print("restored " + filename)
                return self.gmm_dict
            
        except:
            print("restoring failed: " + filename)
            
        from sklearn.mixture import BayesianGaussianMixture
        self.gmm_dict = {
            role: BayesianGaussianMixture(n_components=min(
                8, 1+int(len(df)/2)), random_state=42).fit(df)
            for role, df in self.get_rr_token_dict().items() if len(df) > 2
        }
        joblib.dump(self.gmm_dict, filename)
        print("stored " + filename)
        
        return self.gmm_dict
        
    def get_gmm_dict(self):
        '''
        initialize GMMs with token context encodings
        also form a GMM cluster data frame
        '''        
        if not 'gmm_dict' in self.__dict__:
            self.build_gmm_dict()            
        return self.gmm_dict

    # def get_rr_gmm(self,role='free_topic'):
    #     '''
    #     initialize GMMs with token context encodings
    #     also form a GMM cluster data frame
    #     '''        
    #     if not 'gmm_dict' in self.__dict__:
    #         self.build_gmm_dict()
            
    #     return self.gmm_dict[role]            


    @staticmethod
    def make_tokenfeat_df(d: dict) -> pd.DataFrame:
        '''
        Parameters
        ----------
        token_dict : dict
            The token dictionary
    
        Returns
        -------
        tokenfeat_df : pd.DataFrame
            feature stats dataframe.
    
        '''
        print("==make_tokenfeat_df==")
        feats_list = [pd.DataFrame({'token': t, 'token_form': d[t]['token_form'],
                                   'concept_uri':d[t]['concept_uri'], **dict(d[t]['t_arcs'])}) for t in d.keys()]
        tokenfeat_df = pd.concat(feats_list)
        print("==make_tokenfeat_df done==")
        return tokenfeat_df
    
    def get_tokenfeat_df(self):
        if not '_tokenfeat_df' in self.__dict__:
            self._tokenfeat_df = SG.make_tokenfeat_df(self.get_tokendict())

        return self._tokenfeat_df
    
    def get_som_data(self):
        if 'som_data' in self.__dict__:
            return self.som_data
        
        import joblib
        filename=self.g_path+"/som.pkl"
        try:
            self.som_data = joblib.load(filename)            
            print(filename+" restored")
        
        except:
            from som_lattice import SomLattice
            sm=SomLattice(self.ica_tf.n_components)
            print("building SOM of data "+str(self.ic_rr_token_df.shape))
            sm.train_SOM(np.array(self.ic_rr_token_df)[:,0:8],epochs=20,learn_rate=.2)
            self.som_data = sm.SOM
            joblib.dump(self.som_data, filename)
            print(filename+" saved")
            
        return self.som_data
    
    def get_som_lattice(self):
        from som_lattice import SomLattice
        som=SomLattice(self.ica_tf.n_components)
        som.SOM=self.get_som_data()
        return som
        

# In[delex_ica]

def delexicalized(dddf : dict, dim_feats=90) -> pd.DataFrame():
    stats = pd.crosstab(dddf.token,dddf.arcpath)
    neat_stats = pd.DataFrame(stats[stats.sum().sort_values(ascending=False).head(dim_feats).index])
    return neat_stats
    

def delex_ica_df(stats: pd.DataFrame, n_comps=16):
    from sklearn.decomposition import FastICA
    d_ica = FastICA(n_components=n_comps)
    ic_names = ['IC%d' % i for i in range(d_ica.n_components)]

    d_ica.fit(stats)
    # d_comps = pd.DataFrame(d_ica.components_.T, index=stats.columns)
    # d_comps[0]
    # d_comps[0].sort_values().head()
    # d_comps[0].sort_values().tail()
    delex_df = pd.DataFrame(d_ica.transform(stats),
                            index=stats.index,
                            columns=ic_names).sort_values('IC0')
    return delex_df



# In[scoring]


# In[main]

def group_test():
    global x
    
    # for i in SG._get_group_list():
    for i in ['Life#life']:
        
        x = SG(i)
        x.get_tokendict()


if __name__ == '__main__':
    group_test()


