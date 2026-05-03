#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 21:53:27 2025

@author: seppo
"""

import pandas as pd

def make_syn_tree(conllu_df : pd.DataFrame):
    '''
    generated syntax tree as png file (in temp folder)

    Parameters
    ----------
    conllu_df : pd.DataFrame
        The conll-u formatted data frame

    Returns
    -------
    fn : str
        relative filename of generated syntax tree

    '''

    import subprocess
    from hashlib import md5

    conllu_df = conllu_df.rename(
        columns={'id':'ID','head':'HEAD','deprel':'DEPREL','form':'FORM'})

    dot_nds = '\n'.join("    nd_"+conllu_df.ID.astype(str)+" [label = \""+conllu_df.FORM+"\"];")+"\n"

    dot_ndrow = (' -> '.join("nd_"+conllu_df.ID.astype(str))) + " [style = \"invis\";];\n"

    arc_df = conllu_df[conllu_df.HEAD.astype(int)>0]

    dot_arceds = '\n'.join("    nd_"+ arc_df.ID.astype(str)+
                           " -> nd_" + arc_df.HEAD.astype(str) +
                           " [ taillabel = \""+arc_df.DEPREL+"\"," +
                           " labelangle = "+((arc_df.ID.astype(int)<arc_df.HEAD.astype(int))*90-45).astype(str)
                           +" ];")


    dot_arcs = ('subgraph arcs01 {\n' +
                ' edge [dir = both, arrowhead = vee, arrowtail = inv, labeldistance = 1.5, color=salmon];\n' +
                dot_arceds +
                '}\n' )

    dot_doc = ('digraph Q \n  { rankdir=\"LR\"; \n '+
               '  subgraph sent01 {\n' +
               '  node [shape = ellipse; color=gray; ];\n'+               
               dot_nds +
               "\n    " + dot_ndrow + "  }\n" + 
               dot_arcs+'}\n')
    
    
    dig_hex = md5(dot_doc.encode('utf8')).digest().hex()
    fn = "plots/syntrees/tree_%s.png" % dig_hex

    with open("plots/temp.dot","w") as f:
        f.write(dot_doc)
        
    cmd = "dot plots/temp.dot -Tpng -o "+fn

    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    print('returned value:', returned_value)
    return fn