# -*- coding: utf-8 -*-
"""
Created on Tue May 15 19:38:51 2018

@author: Ayoub El khallioui
"""

import sklearn.datasets as ds
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import tensorflow as tf
from rdnn import DNN3_Regressor
import numpy as np

data=ds.load_boston()
X=data.data
y=data.target
scaler=StandardScaler()
X_std=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.33,random_state=30)

dim_layers=[X_train.shape[1],100,10,1]
model=DNN3_Regressor(dim_layers,epochs=100,learning_rate=0.001,random_state=10)

Session=tf.Session(graph=model.g)
Session.run(model.init)
Loss_epoch=[]
for epoch in range(model.epochs):
    Loss_batch=[]
    for X_batch,y_batch in DNN3_Regressor.get_mini_batch(X_train,y_train,1,shuffle=True):
        _,loss=Session.run([model.optimizer,model.cost],feed_dict={model.X:X_batch,model.y:y_batch})
        Loss_batch.append(loss)
    Loss_epoch.append(np.mean(Loss_batch))
    print("epoch={} => loss={} ".format(epoch,np.mean(Loss_batch)))

y_pred=Session.run(model.logits,feed_dict={model.X:X_test})
mse=metrics.mean_squared_error(y_test,y_pred)
print("mean_squared_error={}".format(mse))
plt.plot(range(0,len(Loss_epoch)),Loss_epoch)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("Learning Curve")

