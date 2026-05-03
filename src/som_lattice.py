#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:03:06 2025

@author: seppo
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand


class SomLattice:

    def __init__(self, dim_x, m=20, n=16):
        '''
        Initialize a SOM array.
        SOM algorithm based on 
        https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
        But adjusted to hexagonal lattice

        Parameters
        ----------
        m : int, optional
            Dimension 0 of the SOM grid. The default is 20.
        n : int, optional
            Dimension 1 of the SOM grid. The default is 16.
        '''

        rand = np.random.RandomState(0)
        # Initialize the SOM randomly
        self.SOM = rand.randn(m, n, dim_x).astype(float)


    def find_BMU(self, x):
        '''
        Return the (g,h) index of the BMU in the grid
        '''
        SOM = self.SOM
        distSq = (np.square(SOM - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    def update_weights(self, train_ex, learn_rate, radius_sq,
                       BMU_coord, step=3):
        '''
        Update the weights of the SOM cells when given a single training example
        and the model parameters along with BMU coordinates as a tuple
        '''

        SOM = self.SOM
        g, h = BMU_coord
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            SOM[g, h, :] += learn_rate * (train_ex - SOM[g, h, :])
            return SOM
        # Change all cells in a small neighborhood of BMU
        for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
            for j in range(max(0, h-step), min(SOM.shape[1], h+step)):

                # My own tweaks for the hexagonal grid:
                # - shift the "x" by 0.5 on odd rows,
                # - scale the y axis by sqrt(3/4)

                x_f = i if j % 2 == 0 else i+.5  # lattice "x" of focus
                x_b = g if h % 2 == 0 else g+.5  # lattice "x" of BMU

                dist_sq = np.square(x_f - x_b) + np.square(j - h)*.75

                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                SOM[i, j, :] += learn_rate * \
                    dist_func * (train_ex - SOM[i, j, :])
        return SOM

    def train_SOM(self, train_data, learn_rate=.1, radius_sq=1,
                  lr_decay=.02, radius_decay=.02, epochs=10):
        '''
        Main routine for training an SOM. It requires an initialized SOM grid
        or a partially trained grid as parameter
        '''
        SOM = self.SOM

        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            print("SOM epoch %d of %d" % (epoch+1,epochs))
            rand.shuffle(train_data)
            for train_ex in train_data:
                g, h = self.find_BMU(train_ex)
                self.update_weights(train_ex,
                                    learn_rate, radius_sq, (g, h))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return SOM
    
    def show(self,title='SOM Grid'):
        
        SOM=self.SOM
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(6, 3.5), 
            subplot_kw=dict(xticks=[], yticks=[]))
        ax.imshow(np.swapaxes(SOM[:,:,0:3]+.5,0,1))
        ax.title.set_text(title)
        return fig



if __name__ == '__main__':
    s = SomLattice(3)
    s.show('after init')
    s.train_SOM(rand.randn(100,3)*.5+.5,epochs=10)
    fig = s.show('trained on random data')
    
