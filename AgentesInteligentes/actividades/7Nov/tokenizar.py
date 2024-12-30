import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

frase = "The quick brown fox jumps over the lazy dog"

tokens = word_tokenize(frase.lower())
print(tokens)

tokens_sent = sent_tokenize(frase.lower())
print(tokens_sent)

stop_words = set(stopwords.words('english'))
# print(stop_words)


filtered_tokens = tokens
for word in tokens:
    if word in stop_words:
        filtered_tokens.remove(word)

print('----hello---')
print(filtered_tokens)

pos_tags = nltk.pos_tag(filtered_tokens)
print(f"Etiquetas ¨POS: {pos_tags}")

lematizer = WordNetLemmatizer()
lemas = [lematizer.lemmatize(word) for word in filtered_tokens]
print(f'Lemas: {lemas}')

frequency = FreqDist(lemas)
print(f'Frecuencia: {frequency.most_common()}')