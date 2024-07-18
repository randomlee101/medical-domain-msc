import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Dense, Flatten, LSTM, Embedding, BatchNormalization

# extract data frames from the csv file
df = pd.read_csv('../assets/medical-question-pair.csv')
# sample the data at 100% to randomize the data
df = df.sample(frac=1)
# select 'question_1' and 'question_2' as input
x = df[["question_1", "question_2"]]

# get the shape of 'x'
print(x.shape)

# reshape 'x' to combine 'question_1' and 'question_2' into a single column
new_x = np.reshape(x, (3048 * 2, 1))
# select 'label' as output
y = df[["label"]]

# initialize text vectorization layer with max tokens of the most frequent 300 words
text_vectorization = TextVectorization(max_tokens=300)
# train the vectorization layer on the combined 'x' column
text_vectorization.adapt(new_x)

question_1 = text_vectorization(x["question_1"])
question_2 = text_vectorization(x["question_2"])

# the shape of question_1 and question_2 is quite different
# numpy padding is used to normalize the array
question_1 = np.pad(question_1, ((0, 0), (0, 72 - 53)), 'constant', constant_values=(0, 0))
# horizontally stack the two arrays against each other so they are treated as 1 in the model
x = np.hstack((question_1, question_2))

model = Sequential(
    [
        Flatten(),
        Embedding(300, 64, input_length=x.shape[0]),
        BatchNormalization(),
        LSTM(512),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, validation_split=0.2, epochs=100)