# %%
#!pip install tensorflow tensorflow-gpu polars matplotlib sklearn

# %%
import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from matplotlib import pyplot as plt
import gradio as gr
import pickle 

# %%
df=pd.read_csv("C:\\Users\\kshit\\OneDrive\\Desktop\\Machine Learning\\Twitter_Toxicity\\train.csv")

# %%
X = df['comment_text']
y = df[df.columns[2:]].values

# %%
# words in vocab
MAX_FEATURES = 200000

# %%
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')

# %%
vectorizer.adapt(X.values)

# %%
vectorized_text = vectorizer(X.values)

# %%
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000) #160000 buffer size
dataset = dataset.batch(16)       # 16 SAMPLES
dataset = dataset.prefetch(8)     # HELPS BOTTLENECKS

# %%
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

# %%
model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

# %%
model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# %%
model.summary()

# %%
history = model.fit(train, epochs=10, validation_data=val)

# %%
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

# %% [markdown]
# # Prediction

# %%
input_text = vectorizer('You freaking suck! I am going to punish you.')
input_text

# %%
res = model.predict(np.expand_dims(input_text,0))
res

# %%
df.columns[2:]

# %%
batch_X, batch_y = test.as_numpy_iterator().next()

# %%
(model.predict(batch_X) > 0.5).astype(int)

# %%
res.shape

# %% [markdown]
# # Evaluation

# %%
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

# %%
for batch in test.as_numpy_iterator(): 
    X_true, y_true = batch
    yhat = model.predict(X_true)
    
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

# %%
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

# %%
!pip install typing-extensions 3.7.4

# %%
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text

# %%
interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

# %%
interface.launch(share=True)