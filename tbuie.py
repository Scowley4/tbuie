#!/usr/bin/python3

import ankura
import json
import flask
import random

app = flask.Flask(__name__, static_url_path='')

attr_name = 'coarse_newsgroup'

def split_train_test(num_docs, train_size=1000, test_size=1000):
    shuffled_doc_ids = list(range(num_docs))
    random.shuffle(shuffled_doc_ids)
    return shuffled_doc_ids[:train_size], set(shuffled_doc_ids[train_size:train_size+test_size])

@ankura.util.pickle_cache('newsgroups.pickle')
def load_data():
    corpus = ankura.corpus.newsgroups()
    train_ids, test_ids = split_train_test(len(corpus.documents))
    Q, labels = ankura.anchor.build_labeled_cooccurrence(corpus, attr_name, train_ids, 500, 1e-5)
    gs_anchor_indices = ankura.anchor.gram_schmidt_anchors(corpus, Q, 20, return_indices=True)
    gs_anchor_vectors = Q[gs_anchor_indices]
    gs_anchor_tokens = [[corpus.vocabulary[index]] for index in gs_anchor_indices]
    return corpus, Q, labels, train_ids, test_ids, gs_anchor_vectors, gs_anchor_tokens

corpus, Q, labels, train_ids, test_ids, gs_anchor_vectors, gs_anchor_tokens = load_data()

@app.route('/')
def serve_itm():
    return app.send_static_file('index.html')

@app.route('/vocab')
def get_vocab():
    return flask.jsonify(vocab=corpus.vocabulary)

@app.route('/finished', methods=['GET', 'POST'])
def finish():
    print(flask.request.get_json())
    return 'OK'


@app.route('/topics')
def topic_request():
    raw_anchors = flask.request.args.get('anchors')

    if raw_anchors is None:
        anchor_tokens, anchor_vectors = gs_anchor_tokens, gs_anchor_vectors
    else:
        anchor_tokens = json.loads(raw_anchors)
        anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q, corpus)

    topics = ankura.anchor.recover_topics(Q, anchor_vectors)
    topic_summary = ankura.topic.topic_summary(topics[:len(corpus.vocabulary)], corpus)

    classifier = ankura.topic.free_classifier(topics, Q, labels)
    ankura.topic.variational_assign(corpus, topics) # except only do this for test...
    contingency = ankura.validate.Contingency()
    for d in test_ids:
        doc = corpus.documents[d]
        gold = doc.metadata[attr_name]
        pred = classifier(doc)
        contingency[gold, pred] += 1

    return flask.jsonify(anchors=anchor_tokens,
                         topics=topic_summary.tolist(),
                         accuracy=contingency.accuracy())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
