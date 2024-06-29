from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('news_classifier2.h5')   

# Function to load and prepare the tokenizer
def load_tokenizer():
    # Load the data
    real_news_data = pd.read_csv('real_news.csv')
    fake_news_data = pd.read_csv('fake_news.csv')

    # Combine the data
    data = pd.concat([real_news_data, fake_news_data], ignore_index=True)

    # Preprocess the text data
    text_columns = ['title', 'text', 'date']
    combined_text = data[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(combined_text)

    return tokenizer

# Function to preprocess user input
def preprocess_user_input(title, text, date, tokenizer):
    user_input = ' '.join([title, text, date])
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    return padded_sequences

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
        text = request.form['text']
        # subject = request.form['subject']
        date = request.form['date']

        tokenizer = load_tokenizer()
        user_input_preprocessed = preprocess_user_input(title, text, date, tokenizer)
        prediction = model.predict(user_input_preprocessed)[0][0]

        if prediction > 0.5:
            result = f"The news is predicted as Real and the news is {prediction * 100:.2f}% Real."
        else:
            # result = f"The news is predicted as Fake and the news is {(1 - prediction) * 100:.2f}% Fake."
            result = f"The news is predicted as Fake and the news is {(1 - prediction) * 100:.2f}% Fake. "
            disclaimer = "Please note that this prediction is based on the limited data used for training."
            result += disclaimer    
        print("result : ", result)
        
        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)