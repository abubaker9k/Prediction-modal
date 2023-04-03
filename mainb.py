import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the data
data = pd.read_csv('justinputoutput.csv')

# Convert input_data and output_data to string data type
input_data = data['input'].astype(str)
output_data = data['output'].astype(str)

# Tokenize the input sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_data)
input_sequences = tokenizer.texts_to_sequences(input_data)

# Determine the maximum sequence length and vocabulary size
max_len = max([len(seq) for seq in input_sequences])
vocab_size = len(tokenizer.word_index) + 1

# Pad the input sequences to the same length
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')

# Tokenize the output sequences
output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(output_data)
output_sequences = output_tokenizer.texts_to_sequences(output_data)

# Convert output sequences to numpy array
output_sequences = np.array(output_sequences)

# Determine the number of unique output labels
num_classes = len(output_tokenizer.word_index) + 1

# Define the model architecture
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
x_train = input_sequences
y_train = tf.keras.utils.to_categorical(output_sequences, num_classes=num_classes)
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('model1.h5')

# Load the saved model
model = tf.keras.models.load_model('model1.h5')

# Create a dictionary to map indices to words
reverse_word_map = dict(map(reversed, output_tokenizer.word_index.items()))

# Make predictions
while True:
    input_str = input("Enter input: ")
    input_sequence = np.zeros((1, max_len))
    words = input_str.split()
    for i, word in enumerate(words):
        if word in tokenizer.word_index:
            input_sequence[0, i] = tokenizer.word_index[word]
    prediction = model.predict(input_sequence)[0]
    predicted_label = np.argmax(prediction)
    predicted_word = reverse_word_map[predicted_label]
    print("Output: ", predicted_word)
