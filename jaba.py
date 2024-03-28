# %%
#!pip install torch torchvision pandas matplotlib sklearn
#!pip install tensorflow tensorflow-gpu pandas matplotlib sklearn
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from matplotlib import pyplot as plt
import gradio as gr
import pickle

# %%
df = pd.read_csv("C:\\Users\\kshit\\OneDrive\\Desktop\\Machine Learning\\Twitter_Toxicity\\train.csv")
df.head()

# %%
X = df['comment_text']
y = df[df.columns[2:]].values

# %%
# words in vocab
MAX_FEATURES = 200000

# %%
vectorizer = CountVectorizer(max_features=MAX_FEATURES)
vectorized_text = vectorizer.fit_transform(X).toarray()

# %%
class TextDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return torch.tensor(self.text[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# %%
X_train, X_test, y_train, y_test = train_test_split(vectorized_text, y, test_size=0.3, random_state=42)
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %%
class ToxicityClassifier(nn.Module):
    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.embedding = nn.Embedding(MAX_FEATURES+1, 32)
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x.long())
        x, _ = self.lstm(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# %%
model = ToxicityClassifier()
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

# %%
for epoch in range(10):
    model.train()
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# %%
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000) #160000 buffer size
dataset = dataset.batch(16)       # 16 SAMPLES
dataset = dataset.prefetch(8)     # HELPS BOTTLENECKS

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')

history = model.fit(train, epochs=10, validation_data=val)

plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

input_text = vectorizer('You freaking suck! I am going to punish you.')
input_text

res = model.predict(np.expand_dims(input_text,0))
res

batch_X, batch_y = test.as_numpy_iterator().next()

(model.predict(batch_X) > 0.5).astype(int)

res.shape

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator(): 
    X_true, y_true = batch
    yhat = model.predict(X_true)
    
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

with open('model.pkl','wb') as f:
    pickle.dump(model,f)

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text

interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)