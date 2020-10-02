import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

df=pd.read_csv('C:/Users/Josko/Desktop/my_projects/Mach_L/NN/neural_network/en_en/color_6.csv')
df=df.sample(frac=1)

cdic={'r':0,'g':1,'b':2,'k':3,'m':4,'c':5}
df['colors']=df['colors'].apply(lambda x: cdic[x])

train=df.iloc[:50000,:]
test=df.iloc[50000:,:]

#np.random.shuffle(train.values)

#print(train.head())
#print(test.head())

model=keras.Sequential([
    keras.layers.Dense(32, input_shape=(2,), activation='relu'),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='relu'),
##    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(6, activation='softmax') # softmax 1, sigmoid 234
    ])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x=np.column_stack((train.x_axis.values,train.y_axis.values))

model.fit(x,train.colors.values,batch_size=4,epochs=30)

tx=np.column_stack((test.x_axis.values,test.y_axis.values))

print('EVALUATION')
print(model.evaluate(tx,test.colors.values))

print(np.round(model.predict(np.array([[20,20]]))))
