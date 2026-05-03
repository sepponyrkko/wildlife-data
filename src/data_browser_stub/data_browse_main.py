#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:36:54 2025

@author: seppo
"""

import numpy as np
import panel as pn
import plotly.graph_objs as go

pn.extension()
pn.extension("plotly")

# In[widgets]

param_slider = pn.widgets.FloatSlider(
    value=5, start=0, end=20, step=.1, name="Parameter value")


# In[the data]

# Put in here the data
import pandas as pd
from data_browse_model import DBM
import data_browse_model as dbmodel




# In[Data Selector]

ica_axis_selector = pn.widgets.IntSlider(name="Projection Z-axis ICA comp.", start=2, end=10, value=2)

model_r = pn.rx(DBM())

m = model_r.rx.value
sg_select = pn.widgets.Select(name='Select', 
                           groups={'Taxons': m.sg.sample_groups},
                           value=m.sg.group_uri)

n_select = pn.widgets.IntSlider(value=5, start=0, end=50, name='N of examples')
f_select = pn.widgets.IntSlider(value=5, start=0, end=20, name='N of feats')



@pn.depends(sg_select,watch=True)
def sg_model_trigger(selected):
    print("selected: %s"%str(selected))    
    if selected != model_r.rx.value.sg.group_uri:    
        model_r.rx.value=DBM(selected)


def relrole_opts(mobj):
    return ['All ...']+[i+" "+str(j.n_components) for i,j in mobj.sg.get_gmm_dict().items()]

relroles_fn = pn.bind(relrole_opts,model_r)
relrole_select = pn.widgets.Select(name='rel_role category', options=relroles_fn, size=1)


def gmm_comp_opts(mobj,rel_role):
    return list(range(8))

gmm_comps_fn = pn.bind(gmm_comp_opts,model_r,relrole_select)
gmm_comp_select = pn.widgets.Select(name='GMM component', options=gmm_comps_fn, size=1)


def create_data(mobj,ica_z_id,n_sample):
    n_sample = int(n_sample)
    data_df = mobj.sg.ic_rr_token_df.groupby('rel_role').head(n_sample).copy()
    
    zid = int(ica_z_id)
    
    data_df['x'] = data_df[0]
    data_df['y'] = data_df[1]
    data_df['z'] = data_df[zid if zid in data_df.columns else 2]

    return data_df



def create_ellipse(mobj, ica_z_id, relrole):
    
    mobj = model_r.rx.value
    print("create_ellipse",relrole)
    rr=relrole.split(' ')[0]
    print("rr",rr)
    rr = rr if rr in mobj.sg.get_gmm_dict().keys() else "free_topic"   
    
    scg = mobj.stats_gmm(rr)
    
    data_df = pd.concat([
        pd.concat([pd.DataFrame([None]),
                   pd.DataFrame(
                       mobj.gmm_ellipse(
                           scg, compid))
                   ])        
         for 
        compid in range(scg.gmm.n_components)])

    zid = int(ica_z_id)

    data_df['x'] = data_df[0]
    data_df['y'] = data_df[1]
    data_df['z'] = data_df[zid if zid in data_df.columns else 2]

    return data_df


ellipse_fn = pn.bind(create_ellipse,model_r,ica_axis_selector,relrole_select)
scatter_fn = pn.bind(create_data,model_r,ica_axis_selector,n_select)


# In[SOM]

from matplotlib.figure import Figure




def dendro_plot(mobj,rel_role,sel_comp=0):
    print("=======dendro_plot========")
    rr=rel_role.split(' ')[0]
    print("rr",rr)
    rr = rr if rr in mobj.sg.get_gmm_dict().keys() else "free_topic"   
    
    print("dendro_plot from model:",mobj.__class__,rr)
    my_grid = mobj.stats_gmm(rr)
    fig = my_grid.make_dendrogram(sel_comp)    
    
    return fig

dendro_generator_fn = pn.bind(dendro_plot,model_r,relrole_select,gmm_comp_select)
dendro_pane = pn.pane.Matplotlib(dendro_generator_fn, dpi=96,width=600)

som_plot_fn = pn.bind(dbmodel.som_rr_plot,model_r,relrole_select)
    
som_pane = pn.pane.Matplotlib(som_plot_fn, dpi=96)


# In[3D scatter]
# Scatter with "arrows"




# Scatter with "arrows"



def ica_space_plot(data_df,ellipse_df,relrole,n_feat):
    
    n_feat = int(n_feat)

    mobj = model_r.rx.value
    print("ica_space_plot",relrole)
    rr=relrole.split(' ')[0]
    print("rr",rr)
    gmm_rrs = mobj.sg.get_gmm_dict().keys()
    
    fvs = mobj.space_feats(n_feat,[rr] if rr in gmm_rrs else [])
    
    print("fvs",str(list(fvs.index)))
    

    mobj = model_r.rx.value
    arrows = mobj.make_arrow_df(fvs)
    
    # sc_texts = [a+'<br>'+b for (a,b) in list(data_df.index)]
    
    sc_texts = [b+' ... ||'+a.upper()+'||<br>'+mobj.get_token_hover(b).replace('\n','<br>') for (a,b) in list(data_df.index)]
    
    
    sc_colors = pd.factorize(data_df.index.droplevel(1))[0]
    
    marker_pool = ['circle', 'circle-open', 'cross', 'diamond',
        'diamond-open', 'square', 'square-open', 'x']
    
    sc_symbol = [ marker_pool[i%8] for i in pd.factorize(data_df.index.droplevel(1))[0]]
    
    fig = go.Figure(layout=dict(autosize=True,
                                title="%s %s ICA Projection" % (mobj.sg.group_name,rr)))
    fig.add_trace(go.Scatter3d(x=data_df.x,y=data_df.y,z=data_df.z, mode="markers",
                               marker=dict(
                                   size=4,
                                   color=sc_colors, 
                                   colorscale='Viridis',   # choose a colorscale
                                    opacity=0.8,
                                    symbol=sc_symbol),
                               hovertext=sc_texts,
                               hovertemplate="%{x},%{y},%{z}: <br>%{hovertext}",
                               name='Term token'
                               ))
    
    arrowlabels = [t if i%3==1 else "" for i,t in enumerate(list(arrows.index))]
    
    fig.add_trace(go.Scatter3d(x=arrows[0],y=arrows[1],z=arrows[2], 
                               mode="lines+text", text=arrowlabels,
                               hovertemplate="%{x},%{y},%{z}: <br> %{text}",
                               name="features"))
    
    if (len(ellipse_df)):
    
        fig.add_trace(go.Scatter3d(x=ellipse_df.x,y=ellipse_df.y,z=ellipse_df.z, mode="lines",
                                   hovertemplate="GMM",name="GMM"))
    
    fig.update_scenes(xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False)

    
    #fig.layout.autosize = True
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig



space_generator_fn = pn.bind(ica_space_plot,scatter_fn,[],"All 0",f_select)
clusterview_generator_fn = pn.bind(ica_space_plot,scatter_fn,ellipse_fn,relrole_select,f_select)


# In[panel bindings in and out]


cluster_panel = pn.pane.Plotly(
    clusterview_generator_fn, 
#    config={"responsive": True}, 
#    sizing_mode="stretch_width",
    sizing_mode="fixed",
    width=600,height=600
    )

space_panel = pn.pane.Plotly(
    space_generator_fn, 
#    config={"responsive": True}, 
#    sizing_mode="stretch_width",
    sizing_mode="fixed",
    width=1024,height=800
    )


# selection getter

@pn.depends(cluster_panel.param.click_data)
def string_hello_world(click_data):
    return click_data

# callback

@pn.depends(cluster_panel.param.click_data, watch=True)
def print_hello_world(click_data):
    print("hello world", click_data)

# todo:
#    - GMM mesh grid with lines
#    - component arrow df to lines
#    - Label dataframe
#    - SOM map as a view


# In[page layout]

# Very simple page layout

page_layout = pn.template.FastListTemplate(
    title="Token context vector space browser",
)

detail =pn.pane.Markdown("Group name: " + model_r.sg.group_name)

scatter_app= pn.Column(space_panel, string_hello_world)
cluster_app= pn.Row(pn.Column(cluster_panel), pn.Column(relrole_select, gmm_comp_select, dendro_pane))

ica_data_panel = pn.Column(scatter_fn)

tabs = pn.Column(scatter_app)

custer_tab = pn.Tabs(('GMM Components', cluster_app), ('SOM',som_pane), ('Details',ica_data_panel))

tabbs = pn.Tabs(('Token space',tabs),('Cluster',custer_tab))

selectors=pn.Column(sg_select,ica_axis_selector,n_select,f_select)



page_layout.sidebar.append(selectors)
page_layout.sidebar.append(detail)
page_layout.sidebar.append(relrole_select)
page_layout.main.append(tabbs)


page_layout.servable()
