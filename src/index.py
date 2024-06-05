import os
import numpy as np
import pandas as pd
from tensorflow.keras.layers import StringLookup, TextVectorization, Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# read the dataset as csv
df = pd.read_csv('../assets/train.csv')

# randomize data
df = df.sample(frac=1)

# select only the 'text' and 'prediction' columns for analysis
df = df[["text", "prediction"]]


# a function to extract the actual label from the production json string
def prediction_pre_process(x):
    new_x = str(x).split(",")[0]
    new_x = new_x.replace("[{'label':", "").strip()
    return new_x


# applying the function to pre_process the prediction labels
df["prediction"] = df["prediction"].apply(prediction_pre_process)

# a list of unique labels in the prediction column
print(np.unique(df["prediction"]))

# the shape of the data before further pre_processing
print(df.shape)

# a list of prediction labels that have been verified to be nonmedical domains
not_medical_domains = [
    'Autopsy',
    'Chiropractic',
    'Consult - History and Phy.',
    'Diets and Nutritions',
    'Discharge Summary',
    'Emergency Room Reports',
    'IME-QME-Work Comp etc.',
    'Letters',
    'Office Notes',
    'SOAP / Chart / Progress Notes',
    'Speech - Language'
]

# deleting the rows where the predictions
# do not match the criteria of a medical domain
df = df.drop(df[df["prediction"].isin(not_medical_domains)].index)

# counts of each unique predictions
value_counts = df["prediction"].value_counts()
print(value_counts)

# keep only predictions that the value count is greater than 99
df = df[df.prediction.isin(value_counts.index[value_counts.gt(99)])]

# shape of the data with only the medical domains considered
print(df.shape)

# set the input 'x' to the 'text' column
x = np.array(df["text"])
# set the output 'y' to the 'prediction' column
y = np.array(df["prediction"])

# generate the unique outputs from 'y'
# after removing the non-medical domains
unique_y = np.unique(y)

# look up the strings in y to set up rules for encoding
string_lookup = StringLookup(vocabulary=unique_y, output_mode="one_hot")
# assign numerical values to the strings in y based on the rules above
y = string_lookup(y)

# text vectorization layer to preprocess the texts in 'x'
text_vectorization = TextVectorization(split='character')
# training the text vectorization layer based on the data in 'x'
text_vectorization.adapt(x)

# applying the text vectorization layer to the 'x' data
# to convert from string to numerical values
x = np.array(x)

model = Sequential(
    [
        text_vectorization,
        Embedding(1000, 30, input_length=x.shape[0]),
        LSTM(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

model.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')
model.fit(x, y, validation_split=0.2, batch_size=16, epochs=30)

model.summary()
