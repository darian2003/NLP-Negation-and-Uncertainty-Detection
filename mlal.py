import json
import re
from googletrans import Translator
import spacy


def extract_medical_texts_and_predictions(dataset):
    medical_texts = []
    predictions = []

    for data_element in dataset:
        medical_text = data_element['data']['text']
        medical_texts.append(medical_text)

        for result_element in data_element['predictions']:
            predictions.append(result_element['result'])

    return medical_texts, predictions


def detect_language(text):
    translator = Translator()

    return translator.detect(text).lang


def remove_useless_symbols(text):
    allowed_punctuation = r'\.,;:\"!'
    pattern = f'[^{allowed_punctuation}\\w\\s]'
    return re.sub(pattern, '', text).strip()


def extract_word_positions(text):
    pattern = re.compile(r'\w+|[^\w\s]')
    matches = pattern.finditer(text)

    words_with_indices = {match.group(): {'start': match.start(), 'end': match.end()} for match in matches}

    return words_with_indices


def is_useless(word):
    pattern = re.compile(r"[a-zA-Z]|\d")
    return not pattern.search(word)


def preprocess_text(base_text):
    text = remove_useless_symbols(base_text)
    lang = detect_language(text)

    nlp = spacy.load('es_core_news_md') if lang == 'es' else spacy.load('ca_core_news_md')

    doc = nlp(text)
    # print(f'Text is :{text}')
    lemmas_per_sentence = [[token for token in sentence if not is_useless(token.text)] for sentence in doc.sents]
    # print(f'Token per sentence are:{lemmas_per_sentence[:5]}')

    return lemmas_per_sentence


def extract_features(sentence, i, words_with_indices):
    token = sentence[i]
    word = token.text

    features = {
        'word': word,
        'word_lower': word.lower(),
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'is_digit': word.isdigit(),
        'word_length': len(word),
        'contains_digits': bool(re.search(r'\d', word)),
        'pos': token.pos_,
        'lemma': token.lemma_,
        'start-end': {'start': words_with_indices[word]['start'], 'end': words_with_indices[word]['end']}
    }

    features["prefix_2"] = word[:2]
    features["suffix_2"] = word[-2:]

    if i > 0:
        previous_tokens = [sentence[j - 1].text.lower() for j in range(max(0, i - 6), i)]
        features["previous_words"] = previous_tokens

    if i < len(sentence) - 1:
        next_token = sentence[i + 1].text
        features["next_word"] = next_token.lower()

    return features



def main():
    with open('train_data.json', 'r', encoding='utf-8') as train_file:
        train_dataset = json.load(train_file)

    # with open('test.json', 'r', encoding='utf-8') as test_file:
    #     test_dataset = json.load(test_file)

    medical_texts_train, predictions_train = extract_medical_texts_and_predictions(train_dataset)
    # medical_texts_test, predictions_test = extract_medical_texts_and_predictions(test_dataset)

    sentences = preprocess_text(medical_texts_train[0])
    print(extract_features(sentences[0], 1, extract_word_positions(medical_texts_train[0])))
    # print(extract_word_positions(medical_texts_train[0])[:5])
    
    print(sentences[:5])


if __name__ == '__main__':
    main()
