#!/usr/bin/python3

import json
import flask
import random
import os
import ankura 
import time
import pickle
from tqdm import tqdm
import sys
import tempfile
import threading
import argparse

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass
parser=argparse.ArgumentParser(
    description='Used for hosting tbuie with a given dataset',
    epilog=('See https://github.com/byu-aml-lab/tbuie\n' +
            '  and https://github.com/byu-aml-lab/ankura/tree/ankura2/ankura\n' +
            '  for source and dependencies\n \n'),
    formatter_class=CustomFormatter)
parser.add_argument('dataset', metavar='dataset',
                    choices=['newsgroups', 'yelp', 'tripadvisor', 'amazon'],
                    help='The name of a dataset to use in this instance of tbuie')
parser.add_argument('port', nargs='?', default=5000, type=int,
                    help='Port to be used in hosting the webpage')
args=parser.parse_args()

dataset_name = args.dataset
port = args.port

app = flask.Flask(__name__, static_url_path='')

user_data = list()

dev_size = 500
number_of_topics = 50
label_weight = 1
smoothing = 0

if dataset_name == 'newsgroups':
    attr_name = 'coarse_newsgroup'
    corpus = ankura.corpus.newsgroups()
elif dataset_name == 'yelp':
    attr_name = 'binary_rating'
    corpus = ankura.corpus.yelp()
elif dataset_name == 'tripadvisor':
    attr_name = 'label'
    corpus = ankura.corpus.tripadvisor()
elif dataset_name == 'amazon':
    attr_name = 'binary_rating'
    corpus = ankura.corpus.amazon()

def calculate_user_data_accuracy(user_data, Q, test_corpus, train_dev_corpus, attr_name):
    for i, data in enumerate(user_data):
        anchor_tokens, anchor_vectors, accuracy = data
        lr_accuracy = ankura.validate.anchor_accuracy(Q, anchor_vectors, test_corpus, train_dev_corpus, attr_name)
        print('Instance', i, 'Free Classifier Accuracy:', accuracy, 'Logistic Regression Accuracy:', lr_accuracy)

@ankura.util.pickle_cache(dataset_name + '.pickle')
def load_data():
    print('Splitting train/dev and test...')
    # 80/20 split into test and train
    split = ankura.pipeline.test_train_split(corpus, return_ids=True)
    (train_dev_ids, train_dev_corpus), (test_ids, test_corpus) = split
    train_dev_size = len(train_dev_ids)
    print(f'  train/dev size: {train_dev_size}')
    print(f'  test size: {len(test_ids)}')


    train_size = train_dev_size - dev_size
    print('Splitting train and dev...')
    # Second split to give train and dev sets
    split = ankura.pipeline.test_train_split(train_dev_corpus, num_train=train_size, num_test=dev_size, return_ids=True)
    (train_ids, train_corpus), (dev_ids, dev_corpus) = split
    print(f'  train size: {train_size}')
    print(f'  dev size: {dev_size}')

    Q, labels = ankura.anchor.build_labeled_cooccurrence(train_dev_corpus, attr_name,
                                                        range(len(train_dev_corpus.documents)), # All are labeled
                                                        label_weight=label_weight, smoothing=smoothing)

    gs_anchor_indices = ankura.anchor.gram_schmidt_anchors(train_dev_corpus, Q, k=number_of_topics, return_indices=True)
    gs_anchor_vectors = Q[gs_anchor_indices]
    gs_anchor_tokens = [[corpus.vocabulary[index]] for index in gs_anchor_indices]

    #This is memory inefficient, since we never use train_corpus.
    return (Q, labels, train_dev_ids, train_dev_corpus,
            train_ids, train_corpus, dev_corpus, dev_ids,
            test_ids, test_corpus, gs_anchor_vectors,
            gs_anchor_indices, gs_anchor_tokens)

# Load the data (will load from pickle if it can)
(Q, labels, train_dev_ids, train_dev_corpus,
    train_ids, train_corpus, dev_corpus, dev_ids,
    test_ids, test_corpus, gs_anchor_vectors,
    gs_anchor_indices, gs_anchor_tokens) = load_data()


@app.route('/')
def serve_itm():
    return app.send_static_file('index.html')

@app.route('/vocab')
def get_vocab():
    return flask.jsonify(vocab=corpus.vocabulary)


@app.route('/finished', methods=['GET', 'POST'])
def finish():

    directory = os.path.join('FinalAnchors', dataset_name)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    pickle.dump(user_data, tempfile.NamedTemporaryFile(mode='wb',
                                                  delete=False,
                                                  prefix=dataset_name,
                                                  suffix='.pickle',
                                                  dir=directory,
    ))

    return 'OK'

@app.route('/topics')
def topic_request():
    raw_anchors = flask.request.args.get('anchors')

    start=time.time()
    if raw_anchors is None:
        anchor_tokens, anchor_vectors = gs_anchor_tokens, gs_anchor_vectors
    else:
        anchor_tokens = json.loads(raw_anchors)
        anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q, train_dev_corpus)
    print('***Time - tandem_anchors:', time.time()-start)

    start=time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=1e-5, get_c=True)

    print('***Time - recover_topics:', time.time()-start)

    start=time.time()
    topic_summary = ankura.topic.topic_summary(topics[:len(train_dev_corpus.vocabulary)], train_dev_corpus)
    print('***Time - topic_summary:', time.time()-start)

    start=time.time()

    classifier = ankura.topic.free_classifier_dream(train_dev_corpus, attr_name, labeled_docs=train_ids, topics=topics, C=C, labels=labels)
    print('***Time - Get Classifier:', time.time()-start)

    contingency = ankura.validate.Contingency()

    start=time.time()
    for doc in dev_corpus.documents:
        gold = doc.metadata[attr_name]
        pred = classifier(doc)
        contingency[gold, pred] += 1
    print('***Time - Classify:', time.time()-start)
    print('***Accuracy:', contingency.accuracy())

    user_data.append((anchor_tokens, anchor_vectors, contingency.accuracy()))

    return flask.jsonify(anchors=anchor_tokens,
                         topics=topic_summary,
                         accuracy=contingency.accuracy())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
