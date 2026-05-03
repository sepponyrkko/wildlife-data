#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:58:03 2024

@author: seppo
"""
import fasttext

skipgram_model = None
sgm_dict = {}

def build_skipgram_model(D_ft : int):
    '''
        Use a pre-trained the skipgram model, if available.
        The tank folder contains a document sample
        large enough for training a skipgram model.
    '''
    global skipgram_model, sgm_dict
    
    if D_ft in sgm_dict:
        return sgm_dict[D_ft]
    
    fn = 'tank/fasttext_%d.model' % D_ft

    try:
        skipgram_model = fasttext.load_model(fn)
        print("-- loaded the skipgram model %s --" % fn)

    except:
        print("-- train the skipgram model %s --" % fn)
        skipgram_model = fasttext.\
            train_unsupervised('tank/words.txt',
                               model='skipgram',
                               dim=D_ft,
                               bucket=50000,
                               thread=4,
                               minCount=3)

        print("-- saving the skipgram model %s --" % fn)
        try:
            skipgram_model.save_model(fn)
        except:
            print("-- saving the skipgram model failed--")
            
    sgm_dict[D_ft] = skipgram_model
    return skipgram_model

if __name__ == '__main__':
    build_skipgram_model(8)
    
