import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

df=pd.read_csv('C:/Users/Josko/Desktop/my_projects/Mach_L/NN/neural_network/en_en/quadratic_2.csv')
df=df.sample(frac=1)

cdic={'red':0,'blue':1}
df['color']=df['color'].apply(lambda x: cdic[x])

train=df.iloc[:19000,:]
test=df.iloc[19000:,:]

#np.random.shuffle(train.values)

#print(train.head())
#print(test.head())

model=keras.Sequential([
    keras.layers.Dense(24, input_shape=(2,), activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid') # softmax 1, sigmoid 234
    ])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x=np.column_stack((train.x_axis.values,train.y_axis.values))

model.fit(x,train.color.values,batch_size=4,epochs=4)

tx=np.column_stack((test.x_axis.values,test.y_axis.values))

print('EVALUATION')
print(model.evaluate(tx,test.color.values))
