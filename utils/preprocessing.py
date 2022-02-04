from nltk.tokenize import sent_tokenize
def find_negations(review, tokenizer): # whenever 'not' found, connect it with the successive word
    tokens = tokenizer.tokenize(review)
    clean_review = ''
    for i, t in enumerate(tokens):
        if t == 'not' and i != len(tokens)-1:
            tokens[i + 1] = 'NEGATEDW' + tokens[i + 1]
        else:
            clean_review = clean_review + ' ' + t.lower()
    return clean_review.strip()

#DEPRECATED: COULDN'T USE IT BECAUSE PARSING TAKES TOO MUCH TIME.
def find_complex_negations(review, tokenizer, parser, negations_list):
    sentences = sent_tokenize(review)
    clean_review = ''
    for s in sentences:
      tokens = tokenizer.tokenize(s)
      if 'not' in tokens:
          result = parser.raw_parse(s)
          dependency = result.__next__()
          for dep in list(dependency.triples()):
              if str(dep[2][0]) in negations_list:
                  for i, token in enumerate(tokens):
                      if token == str(dep[0][0]):
                          tokens[i] = 'NEGATEDW' + tokens[i]
                          break
                      elif token == 'NEGATEDW' + str(dep[0][0]):
                          tokens[i] = str(dep[0][0])
                          break
          tokens = list(filter(lambda w: w not in negations_list, tokens))
      for t in tokens:
          clean_review = clean_review + ' ' + t.lower()
    return clean_review.strip()
