import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words as w
from fast_aug.text import WordsRandomSubstituteAugmenter
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Dense, Flatten, LSTM, Embedding, BatchNormalization

# fetch all the common english stopwords
english_stopwords = stopwords.words("english")
# fetch all available english words in the nltk library
all_english_words = [w.lower() for w in w.words()]
# initialize the lemmatizer to reduce words to their simplest form
lemmatizer = WordNetLemmatizer()

# initialize word substitution augmenter to
# substitute words with similar words in the english vocabulary
# that are not stopwords at a probability of 5%
aug = WordsRandomSubstituteAugmenter(
    word_params=0.05,
    vocabulary=all_english_words,
    stopwords=set(english_stopwords)
)


# function to apply augmentation
def aug_text_custom(data=""):
    return aug.augment(data)


# function to remove stop words and lemmatize the NOUNS
def remove_stop_words(data=""):
    # splits the text into words
    words = [d for d in data.split()]
    # lemmatize the NOUNS
    # check if it is not a stopword
    # check if the text is purely alphabetical
    words = [lemmatizer.lemmatize(w) for w in words if w not in english_stopwords and w.isalpha()]
    # combine all that is left and set to lower case for uniformity
    remaining_words = str.join(" ", words).lower()
    return remaining_words


# extract data frames from the csv file
df = pd.read_csv('../assets/medical-question-pair.csv')
# group rows and repeat each group 5 times to have more data
df = df.loc[np.repeat(df.index, 5)].reset_index(drop=True)

# sample the data at 100% to randomize the data
df = df.sample(frac=1)

# apply the 'remove_stop_words' and 'aug_text_custom' to question 1 and 2
df["question_1"] = df["question_1"].apply(remove_stop_words).apply(aug_text_custom)
df["question_2"] = df["question_2"].apply(remove_stop_words).apply(aug_text_custom)

# select 'question_1' and 'question_2' as input
x = df[["question_1", "question_2"]]

# get the shape of 'x'
print(x.shape)

# reshape 'x' to combine 'question_1' and 'question_2' into a single column
new_x = np.reshape(x, (15240 * 2, 1))

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
        # the embedding layer calculates the distance between the vectorized texts to form an
        # array with dimension of 2 which is required by the LSTM layer
        Embedding(300, 64, input_length=x.shape[0]),
        BatchNormalization(),
        # LSTM uses an activation function of 'tanh' as it is required by the CuDNN
        # architecture for accelerated inference
        LSTM(512),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision(), AUC()])
history = model.fit(x, y, validation_split=0.2, epochs=100)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Medical Pair RNN Accuracy: Augmented')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# extract actual results from the dataset
cm_labels = np.reshape(np.array(y[-500:]), (1, 500))[0]
# predict the labels from the corresponding training set and round them to result to 0 or 1
predictions = np.round(model.predict(x[-500:]))
# flatten the prediction to match the shape of the labels
predictions = np.reshape(np.array(predictions), (1, predictions.shape[0]))[0]

# plot confusion matrix with 500 sample data
cm = tf.math.confusion_matrix(cm_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Medical Pair RNN Confusion Matrix: Augmented')
plt.show()

