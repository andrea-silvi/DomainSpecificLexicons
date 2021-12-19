import gzip, json
import numpy as np



def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)



def parseDataset(optimizer):
    '''
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    '''
    reviews = []
    for review in parse(optimizer.dataset_name):
        try:
            if review["overall"] != 3.0:
                r = []
                r.append(review["reviewText"])
                r.append(-1 if review["overall"] < 3.0 else +1)
                review.append(r)
        except KeyError:
            pass

    return np.array(reviews)





if __name__ == '__main__':
    opt = upload_args_from_json(os.path.join("parameters", "AmazonDataset.json"))
    reviews = parseDataset(opt)