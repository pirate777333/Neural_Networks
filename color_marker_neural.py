import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

df=pd.read_csv('C:/Users/Josko/Desktop/my_projects/Mach_L/NN/neural_network/en_en/color_marker_6.csv')
df=df.sample(frac=1)

#cdic={'r':0,'g':1,'b':2,'k':3,'m':4,'c':5}
#df['colors']=df['colors'].apply(lambda x: cdic[x])

train=df.iloc[:5000,:]
test=df.iloc[5000:,:]

ohtrc=pd.get_dummies(train.colors).values
ohtrm=pd.get_dummies(train.markers).values
trlab=np.concatenate((ohtrc,ohtrm), axis=1)
#print(trlab)
#np.random.shuffle(train.values)
print(trlab[1])
#print(train.head())
#print(test.head())

model=keras.Sequential([
    keras.layers.Dense(64, input_shape=(2,), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
##    keras.layers.Dense(32, activation='relu'),
##    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(12, activation='softmax') # softmax 1, sigmoid 234
    ])

model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

x=np.column_stack((train.x_axis.values,train.y_axis.values))

np.random.RandomState(seed=42).shuffle(x)
np.random.RandomState(seed=42).shuffle(trlab)

model.fit(x,trlab,batch_size=16,epochs=20)

tx=np.column_stack((test.x_axis.values,test.y_axis.values))

ohtrsc=pd.get_dummies(test.colors).values
ohtsm=pd.get_dummies(test.markers).values
tslab=np.concatenate((ohtrsc,ohtsm), axis=1)

np.random.RandomState(seed=42).shuffle(tx)
np.random.RandomState(seed=42).shuffle(tslab)

print('EVALUATION')
print(model.evaluate(tx,tslab))


#np.random.RandomState(seed=42).shuffle(x)
#np.random.RandomState(seed=42).shuffle(labels)

print(np.round(model.predict(np.array([[19.5,19.5]]))))
