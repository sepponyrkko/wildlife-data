#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 10:06:11 2025

@author: seppo
"""
from sample_group import SG
import sample_group_stats as sgstat
import pandas as pd

# The data model





class DBM:
    '''
    Data selection
    '''    

    def set_samplegroup(self, group_spec = "Life#life"):
        self.sg = SG(group_spec)
        print("Set sample group to %s" % group_spec)
        

        aa=pd.DataFrame(self.sg.pair_df.reset_index(),columns=['relation','btoken','rtoken','tagscore']).rename(columns={'btoken':'token','rtoken':'pair'})
        bb=pd.DataFrame(self.sg.pair_df.reset_index(),columns=['relation','btoken','rtoken','tagscore']).rename(columns={'btoken':'pair','rtoken':'token'})
        aa.relation=aa.relation
        bb.relation="="+bb.relation
        self.pairs = pd.concat([aa,bb])
        self.pairs.set_index('token',inplace=True)
        self.pairs.sort_values('tagscore',ascending=False,inplace=True)
        
        return self

    def __init__(self, group_spec = "Life#life"):
        print("DBM init %s" % group_spec)
        self.set_samplegroup(group_spec)
        self.grids = dict()
        print("DBM init ready")
        
    def stats_gmm(self, rel_role):
        if not rel_role in self.grids:
            self.grids[rel_role] = sgstat.GMMScore(self.sg, rel_role)
        return self.grids[rel_role]

    def stats_space(self):
        if not 'stats' in self.__dict__:
            self.stats = sgstat.SGStats(self.sg)
        return self.stats
        
    def gmm_ellipse(self, scg: sgstat.GMMScore, comp_id=0):
        import numpy as np

        comp_id = comp_id if comp_id < scg.gmm.n_components else 0

        a, b, c, d, eigvals, eigvecs = scg.gmm_comp_eigvec(
            comp_id)  # all zinclusive

        (a, b) = np.sqrt(2*eigvals[0:2])  # elongation (eigvals)
        w = eigvecs           # transformation matrix (eig.v)
        m = scg.gmm.means_[comp_id]  # mean position

        alfa = np.linspace(0, 6.28, 24)  # angles

        zz = np.array([np.cos(alfa)*a, np.sin(alfa)*b]
                      + 6*[alfa*0]).T
        vv = np.matmul(zz, w) + m
        return vv

    def space_feats(self, topN: int = 15, relroles=[]):
        stats = self.stats_space()
        
        (fv,fvw)=stats.get_feat_vec_stat()
        (rr_ap_entr,rr_f_entr)=stats.top_relrole_feature_relentr()
        feats = pd.concat([pd.Series(i.index) 
                           for k, i in rr_f_entr.items() 
                               if len(relroles) < 1 or k in relroles])

        fv_hilite = fvw[fvw.index.isin(feats)].sort_values(
            ascending=False).head(topN)
        if len(fv_hilite)<2:
            fv_hilite=fvw.sort_values(ascending=False).head(topN)
        
        return fv.loc[fv_hilite.index]

    def make_arrow_df(self, vector_df):
        blocks = []
        for i in range(len(vector_df)):
            block = pd.concat(
                [vector_df.iloc[i]*0,
                 vector_df.iloc[i],
                 vector_df.iloc[i].apply(lambda x:None)],
                axis=1).T
            blocks.append(block)
        return pd.concat(blocks) if len(blocks) else vector_df



    def get_token_hover(self,token_id):
        import extools as ex
        m = self
        d = m.sg.tokendict
        e = d[token_id] if token_id in d else ex.pretty_token(token_id)
        p_txt = "%s (%s %3.1f%%)\n\n   %s\n" % (e['term'], e['concept_uri'], e['tag_w'], ex.short_token_ctx(e['ctx'],ll=20))
        if not (token_id in m.pairs.index):
            return p_txt+" ((( === no pair === )))"
        
        p_df = m.pairs.loc[[token_id]].head(5).reset_index(drop=True)
        p_df['ctx']=[ex.short_token_ctx(d[p]['ctx'] if p in d else ex.pretty_token(p).ctx,lc=10,ll=15) for p in p_df.pair]
        
        return p_txt + "\n  ---PAIRS---\n" + '\n'.join(list(p_df.relation + " >> " + p_df.ctx))
    
def som_plot(mobj,rel_role):
    from matplotlib.figure import Figure
    import numpy as np
    
    print("updating som from model:",mobj.__class__)
    sm=mobj.sg.get_som_lattice()
    print("got SOM")
    
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots(subplot_kw=dict(xticks=[], yticks=[]))    
    
    SOM = sm.SOM
   
    ax.title.set_text("SOM")
    
    #nolla = ax.imshow(np.swapaxes(SOM[:,:,0:3]+param_slider_val*.1,0,1))
    #fig.colorbar(nolla)

    coords = [(x+0.5*(y%2),y*.85,x,y) for x in range(SOM.shape[0]) for y in range(SOM.shape[1])]
    dims = [0,1,2]
    for c in coords:
        (x,y,xi,yi)=c
        sc = SOM[xi,yi,dims]*.8
        col=np.clip(sc+.5,0,1)
        ax.plot(x, y, marker='h',markersize=15,alpha=.9,color=col)
    
    return fig

# In[testx]
def som_rr_plot(mobj,rel_role):
    from matplotlib.figure import Figure
    import numpy as np
    
    rr=rel_role.split(' ')[0]
    print("som_rr_plot rr",rr)
    
    stats = mobj.stats_space()
    som_rr_df, som_rr_H, som_tokens_df = stats.som_cell_popdist()
    
    
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots(subplot_kw=dict(xticks=range(20), yticks=range(16)))    
       


    som_rr_hits = pd.DataFrame(som_rr_df,columns=som_rr_df.columns[4:])
    rrname = rr.replace('-','_')
    rrname = 'free_topic' if rrname not in som_rr_df.columns else rrname
    maxval = max(som_rr_hits.sum(axis=1))+1
    maxhits = max(som_rr_hits[rrname])+0.1
    for cell,c in som_rr_df.iterrows():

        
        heat = np.sqrt(sum(som_rr_hits.loc[cell])/maxval)
        sc = np.array([np.sin(heat*4-2.5),np.sin(heat*4-0.5),np.sin(heat*4+1.5)])*.5+.5
        hitval = 10*c[rrname]/maxhits
        
        col=np.clip(sc,0,1)

        ax.plot(c.dx, c.cy, marker='h',markersize=12,alpha=1,color=col)
        ax.plot(c.dx, c.cy, marker='h',markersize=hitval,alpha=1,color='black')
    
    ax.title.set_text("SOM %s %s" % (stats.sg.group_name,rrname))
    
    print("som rr plot ready")
    
    return fig

# In[testy]

if __name__ == '__main__':
    m = DBM('Bat#order')
    scg = m.stats_gmm('adaptation_comment')
    scg.compute_feature_prediction()
    x = scg.get_pred_correlation()
