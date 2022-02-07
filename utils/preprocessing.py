import nltk

nltk.download("stopwords")
nltk.download('punkt')
import spacy
from spacy.symbols import neg
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

STOPWORDS = stopwords.words('english')

NEGATION_TOKENS = {"not", "but", "never"}  # example


# todo : add stopwords removal option (STOPWORDS \ NEGATION_TOKENS)
def find_negations(review, tokenizer):
    """
    sentence negation processing : 
    the word just after a negation ('not') is negated
    ex : 
    - I did not like the movie.
    >> I did NEGATEDWlike the movie.
    """
    tokens = tokenizer.tokenize(review)
    clean_review = ''
    for i, t in enumerate(tokens):
        if t == 'not' and i != len(tokens) - 1:
            tokens[i + 1] = 'NEGATEDW' + tokens[i + 1]
        else:
            clean_review = clean_review + ' ' + t.lower()
    return clean_review.strip()


# todo : use double negation (more difficult to implement)
def whole_sentence_negation(review, tokenizer, negation_tokens=NEGATION_TOKENS):
    """
    sentence negation processing : 
    every word inbetween a negation token and the end of the sentence will be considered negative
    ex : (with negation_tokens containing {'not', 'but'})
    - I did not like the movie.
    >> I did NEGATEDWlike NEGATEDWthe NEGATEDWmovie.
    """
    negation_prefix = 'NEGATEDW'
    result = list()

    for sent in sent_tokenize(review):
        tokens = tokenizer.tokenize(sent)
        tokens = [t.lower() for t in tokens]
        for i, t in enumerate(tokens):
            if t in negation_tokens and i != len(tokens) - 1:
                for j in range(i + 1, len(tokens)):
                    tokens[j] = negation_prefix + tokens[j]
                break  # we only need to parse it once for now
        result.extend(tokens)

    return ' '.join(result)


def find_complex_negations(review, parser):
    """
        sentence negation processing :
        we negate the words which are referenced through a "neg" dependency edge of spacy parser.
        - I did not really like the movie.
        >> I did really NEGATEDWlike the movie.
        """
    negation_prefix = 'NEGATEDW'
    complete_tokens = []
    review = review.lower()
    for sent in sent_tokenize(review):
        doc = parser(sent)
        final_tokens = [w.text for w in doc]
        idx_to_delete = []
        for i, token in enumerate(doc):
            if token.dep == neg:
                idx_to_delete.append(i)
                for j, final_token in enumerate(final_tokens):
                    if final_token == str(token.head):
                        final_tokens[j] = negation_prefix + final_token
                        break
        for index, i in enumerate(idx_to_delete):
            del final_tokens[i-index]
        complete_tokens = complete_tokens + final_tokens
    return ' '.join(complete_tokens).strip()


if __name__ == '__main__':
    from nltk import RegexpTokenizer

    tokenizer = RegexpTokenizer(r'\w+')
    s = 'I did not enjoy the movie. Never had I seen such a bad movie.'
    print(whole_sentence_negation(s, tokenizer))
    print(find_negations(s, tokenizer))
