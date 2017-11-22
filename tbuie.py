#!/usr/bin/python3

import ankura
import json
import flask

app = flask.Flask(__name__, static_url_path='')

train_size = 1000

@ankura.util.pickle_cache('newsgroups.pickle')
def load_data():
    corpus = ankura.corpus.newsgroups()
    Q = ankura.anchor.build_cooccurrence(corpus)
    gs_anchor_indices = ankura.anchor.gram_schmidt_anchors(corpus, Q, 20, return_indices=True)
    gs_anchor_vectors = Q[gs_anchor_indices]
    gs_anchor_tokens = [[corpus.vocabulary[index]] for index in gs_anchor_indices]
    return corpus, Q, gs_anchor_vectors, gs_anchor_tokens

corpus, Q, gs_anchor_vectors, gs_anchor_tokens = load_data()

@app.route('/')
def serve_itm():
    return app.send_static_file('index.html')

@app.route('/vocab')
def get_vocab():
    return flask.jsonify(vocab=corpus.vocabulary)

@app.route('/finished', methods=['GET', 'POST'])
def finish():
    print('finished')


@app.route('/topics')
def topic_request():
    raw_anchors = flask.request.args.get('anchors')

    if raw_anchors is None:
        anchor_tokens, anchor_vectors = gs_anchor_tokens, gs_anchor_vectors
    else:
        anchor_tokens = json.loads(raw_anchors)
        anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q, corpus)

    topics = ankura.anchor.recover_topics(Q, anchor_vectors)
    topic_summary = ankura.topic.topic_summary(topics, corpus)

    return flask.jsonify(anchors=anchor_tokens,
                         topics=topic_summary,
                         accuracy=.7522883,

                         examples=[])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
