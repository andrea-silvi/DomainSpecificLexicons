def find_negations(review, tokenizer):
    tokens = tokenizer.tokenize(review)
    clean_review = ''
    for i, t in enumerate(tokens):
        if t == 'not' and i != len(tokens)-1:
            tokens[i + 1] = 'NEGATEDW' + tokens[i + 1]
        else:
            clean_review = clean_review + ' ' + t.lower()
    return clean_review.strip()
