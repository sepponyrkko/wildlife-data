#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample group operations
Created on Sat Dec 28 10:36:21 2024

@author: seppo
"""

import re
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as sp

from sample_group import SG
import extools as ex

import databox as db

# In[numeric shortcuts]

def nrm(x):
    '''
    normalize vector x by dividing with the sum(x)
    '''
    return x/(0.000001 + sum(x))


def nrm2(x):
    '''
    normalize vector x by dividing with the np.sqrt(sum(x^2))
    '''
    return np.round(x / (0.000001 + np.sqrt(sum(x*x))), 5)


def nrm_scale(x, tot=1.0):
    '''
    normalize vector x to a scale of 0...tot (default 1.0)
    '''
    return nrm(x-np.min(x))*tot


def cos_dst(x, y):
    '''
    The cosine distance, ie. inner product of nrm2(x) and nrm2(y)
    '''
    return np.inner(nrm2(x), nrm2(y)).round(3)

def i_rad(x1, x2):
    '''
    The information radius, also the symmetrical version of KL divergence 
    (D_KL) of two distribution vectors (x1 and x2).
    The distributions are normalized to P1 and P2 before computation.
    The reference distribution Q is computed from the average of the P1 and P2.

    Returns
    -------
    float
        The IRad = 0.5 ( D(P1||Q) + D(P2||Q) ) where Q is the average of P's.

    '''
    mx = (nrm(x1)+nrm(x2))*0.5
    r1 = sp.rel_entr(nrm(x1), mx)
    r2 = sp.rel_entr(nrm(x2), mx)
    return 0.5*(r1+r2).sum()



def shorten_relrole(relrole: str):
    '''
    Shorten a relrole name
    '''
    return relrole[0:3]+'-'+re.sub(r'.*\_', '', relrole)[0:3]


def get_comp_vecs(tokenfeat_df: pd.DataFrame, ica_tf,
                  topN: int = 10, minF: int = 2, minAP: int = 2,
                  always_avg=True):

    # arcpath count must be at least the limit

    feat_df = tokenfeat_df.copy()
    # print(len(feat_df))
    apc = feat_df.arcpath.value_counts()
    feat_df = feat_df[feat_df.arcpath.isin(apc[apc >= minAP].index)]
    # print(len(feat_df),apc.head())

    # Feature vector components (top N plus average component)
    # For unique lemma component, require min F occurrences

    feat_comp_vec_pool = []
    for feat_arcpath, arcpath_feats_df in feat_df.groupby('arcpath'):
        lemma_freq = arcpath_feats_df.lemma.value_counts()

        lemmas_selected = lemma_freq[lemma_freq >=
                                     minF].head(topN).index  # top N=10

        feat_v = dict([(l, ex.compute_feat_vec(feat_arcpath, l, ica_tf))
                       for l in lemmas_selected])

        feat_comps = pd.DataFrame(
            [feat_v[l] for l in lemmas_selected], index=list(lemmas_selected))

        # add averaged vector if list was truncated
        if always_avg or (len(lemma_freq) > len(lemmas_selected)):

            feat_sample = [ex.compute_feat_vec(feat_arcpath, l, ica_tf)
                           for l in lemma_freq.head(10).index]

            ap_mean_v = np.array(feat_sample).mean(axis=0)

            feat_avg = pd.DataFrame([ap_mean_v], index=['...'])
            feat_comps = pd.concat([feat_comps, feat_avg])

        feat_comps.index = feat_arcpath+" "+feat_comps.index
        feat_comps.index.name = 'arcpath_lemma'

        feat_comp_vec_pool.append(feat_comps)

    feat_compvec_df = \
        pd.concat(feat_comp_vec_pool) if (
            len(feat_comp_vec_pool) > 0) else pd.DataFrame()
    return feat_compvec_df


# In[Clustering Analysis part]

class GMMScore:
    '''
    GMM test class
    '''
    
    def __init__(self, sample_group, relrole='adaptation_comment'):
        self.sg = sample_group
        self.relrole = relrole
        self.scoring_init()

    def scoring_init(self):

        tokenfeat_df = self.sg.get_tokenfeat_df()
        relrole = self.relrole
        idata = self.sg.ic_rr_token_df.loc[relrole].copy()
        self.idata = idata

        # filter tokenfeat_df rows where token is in relrole group

        relrole_feat_df = tokenfeat_df[tokenfeat_df.token.isin(idata.index)]

        # a minimum of 3 in this GMM dataset
        MIN_ARCPATH_OCC = 3

        arcpath_cnt = relrole_feat_df.arcpath.value_counts()
        stat_arcpaths = arcpath_cnt[arcpath_cnt >= MIN_ARCPATH_OCC].index
        filter_b = relrole_feat_df.arcpath.isin(stat_arcpaths)

        relrole_feat_df = relrole_feat_df[filter_b].copy()

        relrole_feat_df['feat'] = (
            relrole_feat_df.arcpath +
            " " +
            relrole_feat_df.lemma)

        # Initialize the relation-role gmm estimator

        self.gmm = self.sg.get_gmm_dict()[relrole]
        self.feat_compvec_df = get_comp_vecs(relrole_feat_df, self.sg.ica_tf)

        relrole_feat_df.feat = [(f if f in self.feat_compvec_df.index
                                 else re.sub(r' .*', ' ...', f))
                                for f in relrole_feat_df.feat]

        self.relrole_feat_df = relrole_feat_df

    def compute_feature_prediction(self):

        # true distributions per gmm cluster

        t_clustid = pd.Series(self.gmm.predict(self.sg.ic_token_df),
                              index=self.sg.ic_token_df.index)

        self.feat_clust_xtab = pd.crosstab(
            self.relrole_feat_df.feat,
            list(t_clustid.loc[self.relrole_feat_df.token]))

        self.feats_distr = self.feat_clust_xtab.sum(axis=1)

        # compute predicted token feature hits

        predict_scores = []

        feats = self.feats_distr.index

        for f in feats:

            feature_vec = self.feat_compvec_df.loc[f]

            # the correlation hypothesis test:
            # compute the b-cosine (cluster mean and feature vec)

            hits = np.array(
                [np.inner(nrm2(self.gmm.means_[i]),
                          nrm2(np.array(feature_vec)))
                 for i in range(self.gmm.n_components)]
            )

            predict_scores.append(nrm_scale(hits, self.feats_distr.loc[f]))

        self.feat_clust_predicted_df = pd.DataFrame(np.array(predict_scores),
                                                    index=feats)

        self.feat_clust_xtab = self.feat_clust_xtab.reindex(
            self.feat_clust_predicted_df.index).\
            reindex(columns=range(self.feat_clust_predicted_df.shape[1])).\
            fillna(0)

    def compute_entropy(self):
        feat_xtab = self.feat_clust_xtab

        print("=== compute_entropy %s %s ===" %
              (self.sg.group_name, self.relrole))

        feats_nrm_distr = nrm(self.feats_distr)

        clust_entropy = pd.Series(
            st.entropy(np.array(feat_xtab),
                       np.array(feats_nrm_distr).reshape(-1, 1),
                       axis=0), index=feat_xtab.columns)

        clust_pw_entropy = pd.DataFrame(
            {col:
             sp.rel_entr(
                 nrm(feat_xtab[col]),
                 feats_nrm_distr)
             for col in feat_xtab.columns})

        clust_feat_n = pd.DataFrame(
            {col:
             feat_xtab[col]             
             for col in feat_xtab.columns})
            
        self.clust_entropy = clust_entropy
        self.clust_pw_entropy = clust_pw_entropy
        self.clust_feat_n = clust_feat_n

        print("=== compute_entropy done ===")

    @staticmethod
    def corr_block(pred_df, true_df):
        a_pred = np.array(pred_df)
        a_true = np.array(true_df)

        print(a_true.shape, a_pred.shape)

        # if (a_true.shape[1]==5):
        #     print(a_true, a_pred)

        spearmanr_corr = pd.DataFrame(
            [st.spearmanr(a_pred[:, i], a_true[:, i])
             for i in range(a_pred.shape[1])],
            index=true_df.columns)

        pearsonr_corr = pd.DataFrame(
            [st.pearsonr(a_pred[:, i], a_true[:, i])
             for i in range(a_pred.shape[1])],
            index=true_df.columns)

        pdf = pearsonr_corr.sort_values('pvalue')
        sdf = spearmanr_corr.sort_values('pvalue')
        pdf.columns = ['pearson', 'p_pvalue']
        sdf.columns = ['spearman', 's_pvalue']

        pred_corr_df = pd.concat([pdf, sdf], axis=1)
        pred_corr_df['N'] = true_df.sum(axis=0).loc[pred_corr_df.index]
        return pred_corr_df

    def get_pred_correlation(self):

        # self.compute_feature_prediction()
        # self.compute_entropy()

        # Compute the Pearson r and Spearman correlation
        # Correlation tests, from features to clusters

        # Try reading these but beware of false positives

        # a_pred = np.array(self.feat_clust_predicted_df)
        # a_true = np.array(self.feat_clust_xtab)

        if len(self.feat_clust_xtab) > 2:
            self.pred_corr_df = GMMScore.corr_block(self.feat_clust_predicted_df,
                                                    self.feat_clust_xtab)
        else:
            self.pred_corr_df = None

        if len(self.feat_clust_xtab.T) > 2:
            self.feat_corr_df = GMMScore.corr_block(self.feat_clust_predicted_df.T,
                                                    self.feat_clust_xtab.T)
        else:
            self.feat_corr_df = None

        return self.feat_corr_df

    def save_tables(self):
        '''
        # Write out some data to disk

        Returns
        -------
        pw_entropy : DataFrame
            point-wise entropy for features in GMM components
        corr_df : DataFrame
            feature-wise correlation of feature vectors with cluster means
        '''

        pw_entropy = pd.DataFrame(self.clust_pw_entropy).round(3)
        feat_n = self.clust_feat_n
        
        corr_df = self.feat_corr_df.round(3)

        # output folder

        group_name = self.sg.group_name
        plot_path = "plots/"+group_name+"/"
        suff = shorten_relrole(self.relrole)
        os.makedirs(plot_path, exist_ok=True)

        clust_freqs = self.feat_clust_xtab.sum(axis=0)

        corr_df['group'] = group_name
        corr_df['relrole'] = self.relrole

        pw_entropy = pw_entropy.reset_index().melt(
            id_vars=['feat'], value_name='pw_entr', var_name='cluster')
        
        feat_n = feat_n.reset_index().melt(
            id_vars=['feat'], value_name='feat_n', var_name='cluster')
        
        pw_entropy = pw_entropy.merge(feat_n,on=['feat','cluster'])
        
        pw_entropy = pw_entropy[pw_entropy.feat_n > 0].copy()
        pw_entropy['N'] = list(clust_freqs.loc[pw_entropy.cluster])

        pw_entropy['group'] = group_name
        pw_entropy['relrole'] = self.relrole

        entr_fn = plot_path + "gmm-relentr-"+group_name+"-"+suff+".csv"
        corr_fn = plot_path+"gmm-corrtbl-"+group_name+"-"+suff+".csv"

        pw_entropy.to_csv(entr_fn)
        corr_df.to_csv(corr_fn)

        print("saved: ", corr_fn, entr_fn)

        # xtab_fn = plot_path+"gmm-clustxtab-"+group_name+"-"+suff+".csv"
        # self.feat_clust_xtab.to_csv(xtab_fn)

        return pw_entropy, corr_df

    def get_comp_eigvec(self, comp_id):

        covariances = self.gmm.covariances_[comp_id]
        eigVal, eigVec = np.linalg.eigh(covariances)
        return eigVal[::-1], eigVec[::-1]

    def gmm_comp_eigvec(self, comp_id, max_F=5):
        # TODO save these at somewhere for further analysis

        # compute the elongation axis
        v, w = self.get_comp_eigvec(comp_id)

        # select the most elongation-effecting features

        if len(self.feat_compvec_df) == 0:
            return None, None, None, None, v, w

        feats_cxy = pd.DataFrame(
            np.matmul(self.feat_compvec_df, w.T), index=self.feat_compvec_df.index)
        feats_c0 = feats_cxy[0].sort_values()

        # select the data points that are in this component
        # transform data points into the relative component coordinates

        my_x = self.idata[self.gmm.predict(self.idata) == comp_id]
        my_compx = np.matmul((my_x-self.gmm.means_[comp_id]), w.T)
        my_x = pd.DataFrame(my_compx, index=my_x.index)

        # save this for scatter

        # resolve which feats are available in the GMM component
        comp_feat_df = self.relrole_feat_df[self.relrole_feat_df.token.isin(
            my_x.index)]
        comp_feats = list(comp_feat_df.apply(
            lambda x: x.arcpath+" "+x.lemma, axis=1))
        comp_feats += list(comp_feat_df.apply(lambda x: x.arcpath+" ...", axis=1))

        feats_hilite = feats_c0[feats_c0.index.isin(comp_feats)]
        feats_hilite = feats_hilite.loc[(
            feats_hilite**2).sort_values().tail(max_F).index]
        arcpaths_hilite = feats_hilite.index.str.replace(" .*", "", regex=True)

        # find examples of the tokens with these and close features
        comp_ex_feat = comp_feat_df[comp_feat_df.arcpath.isin(
            arcpaths_hilite)].copy()

        feat_vecs = [ex.compute_feat_vec(comp_ex_feat.arcpath.iloc[i], comp_ex_feat.lemma.iloc[i],
                                         self.sg.ica_tf)
                     for i in range(len(comp_ex_feat))]

        feat_vecs = np.array(feat_vecs)

        # print("FEAT_VECS" + str(feat_vecs))
        # print("w.T" + str(w.T))

        if len(feat_vecs):

            elong_x = np.matmul(feat_vecs, w.T)[:, 0].reshape(-1)
            elong_y = np.matmul(feat_vecs, w.T)[:, 1].reshape(-1)
            comp_ex_feat['elongation_x'] = elong_x
            comp_ex_feat['elongation_y'] = elong_y

        # prepare for plotting

        my_x = my_x[my_x.index.isin(comp_ex_feat.token)]
        feats_hilite = feats_cxy.loc[feats_hilite.index].copy()

        # TODO save tables and this seems fine!

        return comp_ex_feat, feats_hilite, my_x, feats_cxy, v, w

    def gmm_llh_matrix(self, simulated_gen=False):
        '''
        GMM models cross-tested with likelihood analysis

        Returns
        -------
        llh_xtab,d : DataFrame, list
            Cross tabulation and scoring source.

        '''
        gg = self.sg

        if simulated_gen == False:
            d = [[x, y, gy.score(gg.ic_rr_token_df.loc[x])] for x, gx in gg.gmm_dict.items()
                 for y, gy in gg.gmm_dict.items()]

        else:
            d = [[x, y, gy.score(gx.sample(100)[0])] for x, gx in gg.gmm_dict.items()
                 for y, gy in gg.gmm_dict.items()]

        x = pd.DataFrame(d, columns=['source', 'scoring_gmm', 'llh'])

        llh_xtab = pd.crosstab(x.source, x.scoring_gmm,
                               values=x.llh, aggfunc=np.mean).round(3)

        gmm_lhood = np.exp(llh_xtab.apply(lambda x: x-np.max(x), axis=0))
        gmm_lhood = gmm_lhood.apply(
            lambda x: x/(np.sum(x)), axis=1).round(4)*100.0
        return gmm_lhood, d

    def make_dendrogram(self, c_id, topN=10):
        my_grid = self

        from scipy.cluster.hierarchy import dendrogram, linkage
        from matplotlib.figure import Figure

        rel_role = my_grid.relrole

        gg = my_grid.sg
        x = gg.ic_rr_token_df.loc[my_grid.relrole]
        x_df = x.copy()
        x_df['loglhood'] = my_grid.gmm.score_samples(x)
        x_df['c'] = my_grid.gmm.predict(x)
        x_df = x_df.reset_index().set_index(['token_id', 'c'])

        fig = Figure(figsize=(2, 2), tight_layout=True)
        fig.suptitle('%s %s linkage of token cluster "%s"' %
                     (gg.group_uri, rel_role, c_id))
        axs = fig.subplots(subplot_kw=dict(yticks=[]))

        try:

            c_df = x_df.groupby('c').get_group(c_id)

            cx_df = c_df.sort_values('loglhood', ascending=False).head(topN)
            cx_df.index = cx_df.index.droplevel(1)

            fig = Figure(figsize=(12, 2+len(cx_df)*.25),
                         dpi=90, tight_layout=True)
            axs = fig.subplots(subplot_kw=dict(yticks=[]))
            fig.suptitle('Sample group %s\n Relation-role category %s\n'\
                         ' Similarity dendrogram of \n'\
                         'component %s'
                         % (gg.group_uri, rel_role, c_id),
                         fontsize=24)

            ax0 = axs

            if (len(cx_df) >= 2):
                clust_x = np.array(cx_df)[:, :-1]
                clust_x_lnk = linkage(clust_x, 'average', metric='cityblock')
                labeltext = [ex.short_token_ctx(
                    ex.pretty_token(x).ctx) for x in cx_df.index]

                dn = dendrogram(clust_x_lnk, orientation='left', labels=(
                    labeltext), ax=ax0, leaf_font_size=14)
        except:
            print("---make_dendrogram error---")
        return fig

# In[GMM tests]


def demo_gmm():

    my_grid = GMMScore(SG('Life#life'), 'adaptation_comment')

    my_grid.compute_feature_prediction()
    my_grid.compute_entropy()
    my_grid.get_pred_correlation()
    my_grid.save_tables()

    comp_ex_feat, feats_hilite, my_x, feats_cxy, v, w =\
        my_grid.gmm_comp_eigvec(1)

    # TODO make nice plots of the my_x and comp_ex_feat examples

    gmm_lhood, raw_loglh_list = my_grid.gmm_llh_matrix()


# --- Test loops ---




def run_gmm_eigv_tests():
    '''
    Run the GMM elongation stat tests for all sample groups

    '''
    for n in SG.sample_groups:
        sg = SG(n)
        gmms = sg.get_gmm_dict()
        for rr in gmms.keys():
            my_grid = GMMScore(sg, rr)

            compids = pd.Series(my_grid.gmm.predict(
                my_grid.idata)).value_counts()
            compids = list(compids[compids > 1].index)

            elongation = []

            for compi in compids:
                comp_ex_feat, feats_hilite, my_x, feats_cxy, v, w =\
                    my_grid.gmm_comp_eigvec(compi)
                if (not comp_ex_feat is None) and len(comp_ex_feat) > 0:
                    comp_ex_feat['group'] = sg.group_name
                    comp_ex_feat['relrole'] = rr
                    comp_ex_feat['comp_id'] = compi
                    comp_ex_feat['N_tokens'] = len(my_x)
                    elongation.append(comp_ex_feat)

            if len(elongation):
                elongation_df = pd.concat(elongation)
                group_name = sg.group_name
                plot_path = "plots/"+group_name+"/"
                os.makedirs(plot_path, exist_ok=True)
                f_name = plot_path+"gmm-elongation-%s.csv" % rr
                elongation_df.to_csv(f_name)
                print("%s written (%s)" % (f_name, elongation_df.shape))
                print(elongation_df.sample(3, replace=True))

# run_gmm_eigv_tests()



def run_gmm_feat_binning_tests():
    '''
    Run the GMM feature binning tests for all sample groups
    '''
    for n in SG.sample_groups[0:]:
        sg = SG(n)
        gmms = sg.get_gmm_dict()
        for rr in gmms.keys():
            my_grid = GMMScore(sg, rr)
            my_grid.compute_feature_prediction()
            my_grid.compute_entropy()
            my_grid.get_pred_correlation()
            if my_grid.feat_corr_df is not None:
                my_grid.save_tables()


# run_gmm_feat_binning_tests()

# sample tokens

# print(ex.pretty_token('Fos-19c4_3_badgers-6').s_ctx)

# ['Examples of fossorial vertebrates are',
#  '[badgers]',
#  ', naked mole - rats , meerkats , armadillos ,']

# print(ex.pretty_token('Fos-19c4_9_burrowers-4').s_ctx)
# ['Other notable early',
#  '[burrowers]',
#  'include Eocaecilia and possibly Dinilysia .']

# See: Gelfand and Yaglom, Calculation of amount of information
#      about a random function contained in another such function


# In[Component Space Analysis part]

class SGStats:
    def __init__(self, sg: SG):
        self.sg = sg
        self.plot_path = "plots/"+sg.group_name+"/"
        os.makedirs(self.plot_path, exist_ok=True)

    def get_feat_vec_stat(self):
        '''
        1b proto-feature vector selection and stats

        Returns
        -------
        proj_featcomps : TYPE
            DESCRIPTION.
        w_featcomps : TYPE
            DESCRIPTION.
        '''

        if '_proj_featcomps' in self.__dict__:
            print("=== get_feat_vec_stat %s ===> CACHED " % self.sg.group_name)
            return self._proj_featcomps, self._w_featcomps

        gg = self.sg
        self.token_feats_df = gg.get_tokenfeat_df()
        tfdf = self.token_feats_df  # shorthand

        # frequency statistics

        print("=== get_feat_vec_stat - %s frequency statistics ===" %
              self.sg.group_name)

        rrtdf = gg.ic_rr_token_df.reset_index()  # tokens and rel-roles

        self.rr_tokenids = {rr: df.token_id
                            for rr, df
                            in rrtdf.groupby('rel_role')}
        self.rr_tokenids['ALL'] = tfdf.token.unique()

        self.rr_featstats = dict()

        for rr, tokens in self.rr_tokenids.items():

            rr_token_feats_df = tfdf[tfdf.token.isin(tokens)]
            if len(rr_token_feats_df):

                featstats = pd.concat(
                    [rr_token_feats_df.arcpath+" "+rr_token_feats_df.lemma,
                     rr_token_feats_df.arcpath+" ..."])
                self.rr_featstats[rr] = featstats.value_counts()

        self.featstats = self.rr_featstats['ALL']

        # feature vector cache at disk

        found = False
        import os
        import joblib
        fv_pkl_fn = self.plot_path+"feat_vecs.pkl"
        if os.path.exists(fv_pkl_fn):
            try:
                proj_featcomps = joblib.load(fv_pkl_fn)
                print(fv_pkl_fn, " restored OK")
                found = True
            except:
                print(fv_pkl_fn, " failed to restore")

        else:

            proj_featcomps = get_comp_vecs(self.token_feats_df, gg.ica_tf)
            joblib.dump(proj_featcomps, fv_pkl_fn)

        # feature vector 2-norms = weights

        w_featcomps = np.sqrt(np.sum(np.square(proj_featcomps), axis=1))
        w_featcomps.name = 'weight'

        # Also, save stats to disk, if needed

        if not found:

            ffs = []

            for rr, s in self.rr_featstats.items():
                ff = pd.concat([s, w_featcomps], axis=1).dropna(axis=0)
                ff['rel_role'] = rr
                ff['group_name'] = gg.group_name
                ffs.append(ff)

            rr_featcomps_df = pd.concat(ffs).\
                reset_index().\
                rename(columns={'index': 'feat'})

            rr_featcomps_df.round(3).to_csv(
                self.plot_path+"rr_feat_w.tsv", sep="\t", index=False)
            print("=== get_feat_vec_stat SAVED ===")

        self._proj_featcomps = proj_featcomps
        self._w_featcomps = w_featcomps
        print("=== get_feat_vec_stat DONE ===")

        return self._proj_featcomps, self._w_featcomps

    def top_relrole_feature_relentr(self, topN=10):
        '''
        Relative entropy of feature distributions
        per relation-role categories

        Returns
        -------
        tbl_ap : TYPE
            DESCRIPTION.
        tbl_feat : TYPE
            DESCRIPTION.

        '''
        stats = self
        aa, bb = stats.get_feat_vec_stat()

        tbl_ap = dict()
        tbl_feat = dict()

        for rr, rrdf in stats.sg.ic_rr_token_df.groupby('rel_role'):
            rr_tokens = rrdf.index.droplevel(0)
            rr_feats = stats.sg.get_tokenfeat_df()
            rr_feats = rr_feats[rr_feats.token.isin(rr_tokens)]

            rr_featlist = pd.concat(
                [rr_feats.arcpath + " " + rr_feats.lemma, rr_feats.arcpath + " ..."])
            sg_featstat = stats.featstats[stats.featstats.index.isin(
                rr_featlist)]

            rr_featstat = rr_featlist.value_counts()
            rr_featstat = rr_featstat.loc[sg_featstat.index]

            feat_stat_df = pd.concat(
                {'rr_feat_n': rr_featstat, 'feat_n': sg_featstat}, axis=1)
            feat_el_entr = sp.rel_entr(
                nrm(feat_stat_df.rr_feat_n), nrm(feat_stat_df.feat_n))
            feat_stat_df['rr_elem_entr'] = feat_el_entr

            # sorted top 10 tables

            top_10_rr_sign_feat = feat_stat_df[~feat_stat_df.index.str.contains(
                ' ...', regex=False)].sort_values('rr_elem_entr', ascending=False).head(topN)
            top_10_rr_sign_aps = feat_stat_df[feat_stat_df.index.str.contains(
                ' ...', regex=False)].sort_values('rr_elem_entr', ascending=False).head(topN)

            tbl_ap[rr] = top_10_rr_sign_aps
            tbl_feat[rr] = top_10_rr_sign_feat
        return tbl_ap, tbl_feat

    # > top_10_rr_sign_feat EXAMPLE:
    # Out[487]:
    #                                rr_feat_n  feat_n  rr_elem_entr
    # nsubj<nmod>case> of                    5       7      0.006952
    # case> of                               9      21      0.005429
    # nmod<det> the                          6      12      0.005045
    # nmod<nmod<det> the                     3       4      0.004397
    # obj< produce                           3       4      0.004397
    # nsubj<nmod>det> the                    3       4      0.004397
    # nsubj<det> the                         3       4      0.004397
    # compound<compound> Lancashire          2       2      0.003818
    # nmod<nmod<amod> overall                2       2      0.003818
    # compound< Group                        2       2      0.003818

    def evaluate_ica_u_scores(self):
        '''
        Compute Mann-Whitney U scores for ICA projection
        per relation-role (in level 0 in i_df)

        Parameters
        ----------
        i_df : DataFrame
            IC projected data, component data in columns, level 0 is the catgory

        Returns
        -------
        DataFrame, DataFrame
            2 tables of u-scoring data: p-value table AND detail with columns:
              ['rel_role', 'ic_comp', 'u_score', 'p_value']

        '''

        i_df = self.sg.ic_rr_token_df

        roles = i_df.index.levels[0]

        # role-to-role utests

        utests = [[lev, lev2, c, *st.mannwhitneyu(i_df.loc[lev][c], i_df.loc[lev2][c])]
                  for c in i_df.columns for lev in roles for lev2 in roles if lev != lev2]

        # role-to-all utests

        utest = [[lev, c, *st.mannwhitneyu(i_df.loc[lev][c], i_df[c])]
                 for c in i_df.columns for lev in roles]

        u = pd.DataFrame(
            utests,
            columns=['rel_role', 'rel_role_b', 'ic_comp', 'u_score', 'p_value'])

        u1 = pd.DataFrame(
            utest,
            columns=['rel_role', 'ic_comp', 'u_score', 'p_value'])

        utest_p_value_df = pd.crosstab(
            u1.rel_role, u1.ic_comp, u1.p_value, aggfunc=np.min)
        utest_score_df = pd.crosstab(
            u1.rel_role, u1.ic_comp, u1.u_score, aggfunc=np.max)

        utest_x_p_value_df = pd.crosstab(
            u.rel_role, u.rel_role_b, u.p_value, aggfunc=np.min).round(3).fillna(1)

        utest_p_value_df['N'] = i_df.groupby(level=0).count()[0]
        utest_score_df['N'] = i_df.groupby(level=0).count()[0]

        # --- finally save to disk! ---

        group_name = self.sg.group_name
        plot_path = "plots/"+group_name+"/"
        f_name = plot_path+"axis_cross_u_score_pvalue.tsv"
        df = utest_x_p_value_df
        df['group'] = group_name
        df['N'] = i_df.groupby(level=0).count()[0]
        df.to_csv(f_name, sep="\t")

        plot_path = "plots/"+group_name+"/"
        f_name = plot_path+"axis_u_score_pvalue.tsv"
        df = utest_p_value_df.round(3).fillna(1)
        df['group'] = group_name
        df['N'] = i_df.groupby(level=0).count()[0]
        df.to_csv(f_name, sep="\t")

        return utest_p_value_df, utest_score_df, u, utest_x_p_value_df

    def som_cell_popdist(self):
        '''
            SOM population statistics 
            Unit cell rel_role population             
            KL divergence per relation-role category
            Token binning in a lattice
        '''
        print("stats: som_cell_popdist")
        
        somv = self.sg.get_som_lattice()
        ica_data = self.sg.ic_rr_token_df
        data_cells = [somv.find_BMU(ica_data.iloc[n].array)
                      for n in range(len(ica_data))]

        som_tokens_df = pd.DataFrame([
            ["%02d_%02d" % (cx, cy), rel_role, tokenid, cx, cy]
            for (cx, cy), (rel_role, tokenid)
            in zip(
                data_cells, ica_data.index)
        ],
            columns=['cell', 'rel_role', 'tokenid', 'cx', 'cy']
        ).set_index('tokenid')

        xdf = pd.crosstab(som_tokens_df.cell,
                          som_tokens_df.rel_role.str.replace("-", "_"))

        x_tot = nrm(xdf.sum(axis=1))
        som_rr_H = pd.DataFrame(
            {"kldiv_"+rr: sp.rel_entr(nrm(xdf[rr]), x_tot)
             for rr in xdf.columns},
            index=list(xdf.index))

        som_rr_H.index.name = 'cell'

        lw, lh, ld = somv.SOM.shape

        cxcy_df = pd.DataFrame(
            [dict(cx=cx, cy=cy, cell="%02d_%02d" % (cx, cy),
                  dx=cx+(0.5 * (cy % 2)), dy=cy*0.85)
                for cx in range(lw) for cy in range(lh)])

        som_rr_df = (cxcy_df.merge(xdf, on='cell', how='left').
                     set_index('cell').fillna(0))
        print("stats: som_cell_popdist ready!")

        return som_rr_df, som_rr_H, som_tokens_df




# In[Space test]

# --- Single SG test ---

def demo_space():
    # my_grid = GMMScore(SG('Life#life'), 'adaptation_comment')
    
    s_grp = SG('Life#life')
    
    stats = SGStats(s_grp )
    
    utest_a_p, uscore_df, u, utest_p = stats.evaluate_ica_u_scores()
    tbl_aps, tbl_feats = stats.top_relrole_feature_relentr(10)
    som_rr_df, som_rr_H, som_tokens_df = stats.som_cell_popdist()
    print(som_rr_H.sum(axis=0))


# --- Space tests loop over sample groups ---



def get_som_irad_xtab(sombin_df):
    '''
    Compute the Helper for the som_kldiv_tests function.
    Returns dataframe for pairwise IRad between rel_role groups 
    in a SOM binning sombin_df .
    '''
    xsom = pd.crosstab(sombin_df.cell, sombin_df.rel_role)
    cols = xsom.columns
    irad_som = [{'i': i, 'j': j, 'irad': i_rad(
        xsom[i], xsom[j])} for i in cols for j in cols]
    irad_som = pd.DataFrame(irad_som)
    return irad_som


def run_u_tests():
    for n in SG.sample_groups:
        SGStats(SG(n)).evaluate_ica_u_scores()
        print(n)


def run_som_kldiv_tests():
    '''
    Run the SOM KLdiv cross-RR-tests for all sample groups
    '''
    for n in SG.sample_groups:
        sg = SG(n)
        stats = SGStats(sg)
        som_rr_df, som_rr_H, som_tokens_binning = stats.som_cell_popdist()

        group_name = sg.group_name
        plot_path = "plots/"+group_name+"/"
        f_kl_name = plot_path+"som_rr_kldiv.tsv"
        f_irad_name = plot_path+"som_rr_irad.tsv"

        kldiv = pd.DataFrame(som_rr_H.sum(), columns=['kldiv'])

        irad_som = get_som_irad_xtab(som_tokens_binning)
        irad_som['group'] = sg.group_name
        irad_som['g_N'] = som_tokens_binning.shape[0]
        irad_som.to_csv(f_irad_name, sep="\t")
        print(f_irad_name)

        kldiv['group'] = sg.group_name
        kldiv['g_N'] = som_tokens_binning.shape[0]
        kldiv.to_csv(f_kl_name, sep="\t")
        print(f_kl_name)

        print(n)


def run_gmm_llh_tests():
    '''
    Run all the GMM likelihood cross-RR-tests for all sample groups
    '''
    for n in SG.sample_groups:
        my_grid = GMMScore(SG(n), "free_topic")
        aa, bb = my_grid.gmm_llh_matrix()

        group_name = my_grid.sg.group_name
        plot_path = "plots/"+group_name+"/"
        f_name = plot_path+"gmm_x_llh_norm.tsv"
        df = aa.round(4)
        df['group'] = group_name

        Ns = my_grid.sg.ic_rr_token_df.groupby(level=0).count()[0]
        Ns = Ns.loc[df.index]
        df['N'] = Ns
        df.to_csv(f_name, sep="\t")

        print(n, f_name)


# In[explanation helpers]


def filter_aps_from_ctx(cstr_list: list, ap_id: str, fmt='upper'):
    '''
    Modifies the syntactic context "pretty print" (3 element list)
    so that only the specified arcpath will be highlighted in markdown
    style in the output.

    Parameters
    ----------
    cstr_list : list
        the s_ctx member from the extools token pretty print function.
    ap_id : str
        arcpath to be highlighted, e.g. "amod>det>"
    fmt : TYPE, optional
        Special features request. The default is 'upper' to 
        capitalize the dependent token.

    Returns
    -------
    out_list : list (of strings)
        A three-element list similar to 
        the s_ctx output from the pretty token print method.
    '''
    
    ap_tail = '(%s)' % ap_id
    out_list = []
    for cstr in cstr_list:
        out = []

        for w in cstr.split(' '):
            wout = re.sub(r'\(.*\)', '', w)
            if w.endswith(ap_tail):
                wout = '**%s**' % wout.upper() if fmt == 'upper' else '**%s**' % wout
            out.append(wout)
        out_list.append(' '.join(out))
    return out_list


def explain_arcpath(arcpath_name: str, sample_grp: SG, n_examples:int=3):
    '''
    Produce an example of certain arcpath name in given sample group object.
    

    Parameters
    ----------
    - arcpath_name : str
      - for instance "amod>"    
    - sample_grp : SG
      - the SG object where to search for examples
    - n_examples : int, default=3
      - wished (max) number of examples

    Returns
    -------
    outs : list of tuples
        The tuples contain the token index and a short string

    '''

    df = sample_grp.get_tokenfeat_df()
    f_df = df[df.arcpath == arcpath_name]
    outs = []

    for t_id in list(f_df['token'].unique())[0:n_examples]:

        tdict = ex.pretty_token(t_id)
        ss = filter_aps_from_ctx(tdict.s_ctx, arcpath_name)
        out = ex.short_token_ctx(ss)

        print(out)
        outs.append((t_id,out))
    return outs

# In[sent_tab]


def sent_tab(tokenid,hilite='amod>',window=2):
    td = ex.pretty_token(tokenid)
    wid = int(td['word_id'])
    synt = td['t_synt']
    x = db.token_sent(tokenid,include_punct=True).copy()
    x['spc']=(1-x['misc'].str.contains('SpaceAfter=No')).apply(lambda n:' '*n)
    txt_sent = ''.join(list(x.form+x.spc))
    desc_sent = ('Document: %s \n Sentence: %d Word: %d' % 
      (td.document, td.sent_id, td.word_id))
    
    snip_df = td.t_synt.loc[(wid-window):(wid+window+1)].copy()
    print(snip_df)
    
    snip_df['arcpath']='\\small '+snip_df['arcpath']
    snip_df['arcpath'] = snip_df['arcpath'].str.replace(hilite,'{\\bf %s}'%hilite)
    x=snip_df.T.to_latex(index=False,header=False).replace(' & ',' &\n  ')
    
    return x, desc_sent, txt_sent
    
    
    

    