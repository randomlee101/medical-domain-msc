import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words as w
from fast_aug.text import WordsRandomSubstituteAugmenter
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Dense, Flatten, BatchNormalization

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
        # flatten to make the data 1-dimensional
        Flatten(),
        # a regularization layer to keep values between 0 and 1
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        # output layer is 1 because the expected output of size 1
        # the activation function 'sigmoid' can be either 0 or 1 which is suitable for this model
        Dense(1, activation="sigmoid")
    ]
)

# binary_crossentropy is used as a loss function to adjust the weights of the layers
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision(), AUC()])
# the model is being trained over a cycle of 100 with 20% of the data set aside for
# evaluation during training
model.fit(x, y, validation_split=0.2, epochs=100)
