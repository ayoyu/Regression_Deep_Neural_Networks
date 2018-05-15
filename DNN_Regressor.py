# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:58:25 2018

@author: Ayoub El khallioui
"""
import tensorflow as tf
import numpy as np

class DNN3_Regressor(object):
    def __init__(self,dim_layers,epochs,learning_rate,random_state):
        self.dim_layers=dim_layers
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.random_state=random_state
        self.g=tf.Graph()
        with self.g.as_default():
            np.random.seed(self.random_state)
            self.build()
            self.init=tf.global_variables_initializer()
        
    def build(self):
        self.X=tf.placeholder(dtype=tf.float32,shape=(None,self.dim_layers[0]),name="input")
        self.y=tf.placeholder(dtype=tf.float32,shape=(None),name="output")
        h1=tf.layers.dense(inputs=self.X,units=self.dim_layers[1],activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(seed=10))
        h2=tf.layers.dense(inputs=h1,units=self.dim_layers[2],activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(seed=10))
        self.logits=tf.layers.dense(inputs=h2,units=self.dim_layers[3],activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=10))
        self.cost=tf.losses.mean_squared_error(labels=self.y,predictions=self.logits)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.cost)

    @staticmethod
    def get_mini_batch(X_train,y_train,size_batch,shuffle=True):
        X=np.array(X_train,copy=True)
        y=np.array(y_train,copy=True)
        data=np.column_stack((X,y))
        if shuffle:
            np.random.shuffle(data)
            X=data[:,:-1]
            y=data[:,-1]
        for i in range(0,X_train.shape[0],size_batch):
            yield(X[i:i+size_batch,:],y[i:i+size_batch])




    
    
        