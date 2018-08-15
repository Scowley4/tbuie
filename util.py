import ankura
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.linear_model import LogisticRegression
import scipy
import numpy as np
import time
import pickle
import os
import socket

Z = 'z'
THETA = 'theta'
prior_attr_name = 'lambda'

# corpus_data was pickled as this tuple:
#   (Q, labels, train_dev_ids, train_dev_corpus,
#   train_ids, train_corpus, dev_corpus, dev_ids,
#   test_ids, test_corpus, gs_anchor_vectors,
#   gs_anchor_indices, gs_anchor_tokens)

def get_logistic_regression_accuracy_word_topic_pairs(unpickled_corpus_data, anchors, attribute_name='binary_rating'):
    Q = unpickled_corpus_data[0]
    train_ids = unpickled_corpus_data[2] # train_dev ids and corpus from tbuie
    train_corpus = unpickled_corpus_data[3]
    test_ids = unpickled_corpus_data[8] # This is the test data, not the dev data used in tbuie.
    test_corpus = unpickled_corpus_data[9]
    num_topics = len(anchors)

    train_target = [doc.metadata[attribute_name] for doc in train_corpus.documents]
    test_target = [doc.metadata[attribute_name] for doc in test_corpus.documents]

    topics = ankura.anchor.recover_topics(Q, anchors, 1e-5)

    ankura.topic.sampling_assign(train_corpus, topics, z_attr=Z)
    ankura.topic.sampling_assign(test_corpus, topics, z_attr=Z)

    train_matrix = scipy.sparse.lil_matrix((len(train_corpus.documents), num_topics * len(train_corpus.vocabulary)))
    test_matrix = scipy.sparse.lil_matrix((len(test_corpus.documents), num_topics * len(test_corpus.vocabulary)))

    for i, doc in enumerate(train_corpus.documents):
        for j, t in enumerate(doc.tokens):
            train_matrix[i, t[0] * num_topics + doc.metadata[Z][j]] += 1

    for i, doc in enumerate(test_corpus.documents):
        for j, t in enumerate(doc.tokens):
            test_matrix[i, t[0] * num_topics + doc.metadata[Z][j]] += 1

    lr = LogisticRegression()
    lr.fit(train_matrix, train_target)

    return lr.score(test_matrix, test_target)


def get_logistic_regression_accuracy(Q, train_corpus, test_corpus, anchor_vectors, attribute_name='binary_rating'):
    num_topics = len(anchor_vectors)

    train_target = [doc.metadata[attribute_name] for doc in train_corpus.documents]
    test_target = [doc.metadata[attribute_name] for doc in test_corpus.documents]

    topics = ankura.anchor.recover_topics(Q, anchor_vectors, 1e-5)

    ankura.topic.gensim_assign(train_corpus, topics, theta_attr=THETA)
    ankura.topic.gensim_assign(test_corpus, topics, theta_attr=THETA)

    train_matrix = np.zeros((len(train_corpus.documents), num_topics))
    test_matrix = np.zeros((len(test_corpus.documents), num_topics))

    for d, doc in enumerate(train_corpus.documents):
        train_matrix[d] = doc.metadata[THETA]

    for d, doc in enumerate(test_corpus.documents):
        test_matrix[d] = doc.metadata[THETA]

    lr = LogisticRegression()
    lr.fit(train_matrix, train_target)

    return lr.score(test_matrix, test_target)




# corpus_data was pickled as this tuple:
#   (Q, labels, train_dev_ids, train_dev_corpus,
#   train_ids, train_corpus, dev_corpus, dev_ids,
#   test_ids, test_corpus, gs_anchor_vectors,
#   gs_anchor_indices, gs_anchor_tokens)
def get_fc_test_metrics(Q, labels, train_corpus, test_corpus,
                        anchor_vectors, attr_name='binary_rating'):
    start = time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=1e-5, get_c=True)

    classifier = ankura.topic.free_classifier_dream(train_corpus, attr_name,
                                                    labeled_docs=set(range(len(train_corpus.documents))), topics=topics,
                                                    C=C, labels=labels,
                                                    prior_attr_name=prior_attr_name)
    contingency = ankura.validate.Contingency()
    for doc in test_corpus.documents:
        gold = doc.metadata[attr_name]
        pred = classifier(doc)
        contingency[gold, pred] += 1
    #There was a divide by 0 error here for recall, so I just made an accuracy one
    return contingency.accuracy(), contingency.recall(), contingency.precision()

mytot = 0
mycount = 0

def get_fc_test_acc(Q, labels, train_corpus, test_corpus,
                        anchor_vectors, attr_name='binary_rating'):
    C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=1e-5, get_c=True)

    classifier = ankura.topic.free_classifier_dream(train_corpus, attr_name,
                                                    labeled_docs=set(range(len(train_corpus.documents))), topics=topics,
                                                    C=C, labels=labels,
                                                    prior_attr_name=prior_attr_name)
    start=time.time()
    contingency = ankura.validate.Contingency()
    for doc in test_corpus.documents:
        gold = doc.metadata[attr_name]
        pred = classifier(doc)
        contingency[gold, pred] += 1
    global mytot
    global mycount
    mytot += time.time()-start
    mycount += 1
    print('\tAVERAGE Classify Time:', mytot/mycount)
    print(f'\tThis classify: {time.time()-start}')
    return contingency.accuracy()



def user_data_to_dicts(user_data_raw, dataset, file_path=None):
    user_data_dicts = [{'dataset': dataset, 'file': file_path, 'update_num': i,
                        'max': False, 'min': False, 'first': False,
                        'anchor_tokens': data[0], 'anchor_vectors': data[1], 'fc_dev_acc': data[2]}
                        for i, data in enumerate(user_data_raw)]
    return user_data_dicts

# corpus_data was pickled as this tuple:
#   (Q, labels, train_dev_ids, train_dev_corpus,
#   train_ids, train_corpus, dev_corpus, dev_ids,
#   test_ids, test_corpus, gs_anchor_vectors,
#   gs_anchor_indices, gs_anchor_tokens)
# def get_logistic_regression_accuracy(unpickled_corpus_data, anchors, attribute_name='binary_rating'):
def process_all_user_data(corpus_data_path, user_data_directory_path,
                          dataset_name, n_used=10, attr_name='binary_rating'):

    print('Getting corpus data...')
    corpus_data = pickle.load(open(corpus_data_path, 'rb'))

    (Q, labels, train_dev_ids, train_dev_corpus,
      train_ids, train_corpus, dev_corpus, dev_ids,
      test_ids, test_corpus, gs_anchor_vectors,
      gs_anchor_indices, gs_anchor_tokens) = corpus_data

    all_users_data = []
    print('Starting user_data stuff...')
    for user_num, user_data_file in enumerate(os.listdir(user_data_directory_path)):
        print(f'File {user_num+1}')
        file_path = os.path.join(user_data_directory_path, user_data_file)
        user_data_raw = pickle.load(open(file_path, 'rb'))

        n_updates = len(user_data_raw)
        random_inds = np.random.choice(n_updates, size=min(n_updates, n_used), replace=False)

        user_data_raw = [user_data_raw[i] for i in random_inds]

        user_data = user_data_to_dicts(user_data_raw, dataset_name, user_data_file)


        for i, data in enumerate(user_data):
            print(f'  update: {i+1}/{len(user_data)}')
            start = time.time()
           # lr_acc = get_logistic_regression_accuracy(Q, train_dev_corpus, test_corpus,
           #                                           data['anchor_vectors'], attr_name)
           # end = time.time() - start
           # data['lr_time'] = end
           # data['lr_acc'] = lr_acc

            start = time.time()
            fc_acc = get_fc_test_acc(Q, labels, train_dev_corpus,
                                     test_corpus, data['anchor_vectors'],
                                     attr_name)
            end = time.time() - start
            data['fc_acc'] = fc_acc
            data['fc_time'] = end
            print(end)
        all_users_data += user_data

    return all_users_data

def run_them_all(base_path='UserData', final_anchors_path='UserDataFinalAnchors', n_used=30):
    data_dict = dict()
    dataset_names = ['yelp', 'amazon', 'tripadvisor']
    dataset_attrs = ['binary_rating', 'binary_rating', 'label']
    dataset_names = ['amazon']
    dataset_attrs = ['binary_rating']
    for dataset_name, attr_name in zip(dataset_names, dataset_attrs):
        corpus_pickle_path = os.path.join(base_path, dataset_name + '.pickle')
        user_data_path = os.path.join(base_path, final_anchors_path, dataset_name)
        print('Beginning to use',corpus_pickle_path)
        print('  with data in', user_data_path)
        data = process_all_user_data(corpus_pickle_path, user_data_path,
                                     dataset_name, n_used=n_used, attr_name=attr_name)
        data_dict[dataset_name]=data
    try:
        pass
        #with open('data_dict.pickle', 'wb') as outfile:
        #    pickle.dump(data_dict, outfile)
    finally:
        return data_dict

def get_maxes_mins_and_start(data_dict):
    base_path = 'UserData'
    final_anchors_path = 'UserDataFinalAnchors'
    dataset_names = ['yelp', 'amazon', 'tripadvisor']
    dataset_attrs = ['binary_rating', 'binary_rating', 'label']
    #data_files_used = set(data['file'] for dataset_name in dataset_names
    #                                   for data in data_dict[dataset_name])
    for dataset_name, attr_name in zip(dataset_names, dataset_attrs):
        corpus_pickle_path = os.path.join(base_path, dataset_name + '.pickle')
        user_data_path = os.path.join(base_path, final_anchors_path, dataset_name)
        print('Beginning to use',corpus_pickle_path)
        print('  with data in', user_data_path)

        print('Getting corpus data...')
        corpus_data = pickle.load(open(corpus_pickle_path, 'rb'))

        (Q, labels, train_dev_ids, train_dev_corpus,
          train_ids, train_corpus, dev_corpus, dev_ids,
          test_ids, test_corpus, gs_anchor_vectors,
          gs_anchor_indices, gs_anchor_tokens) = corpus_data

        print('Starting user_data stuff...')
        get_first=True
        for user_num, user_data_file in enumerate(os.listdir(user_data_path)):
            print(f'File {user_num+1}')
            file_path = os.path.join(user_data_path, user_data_file)
            user_data_raw = pickle.load(open(file_path, 'rb'))
            print(len(user_data_raw))

            n_updates = len(user_data_raw)

            max_ind = max(range(n_updates), key=lambda num: user_data_raw[num][2])
            min_ind = min(range(n_updates), key=lambda num: user_data_raw[num][2])
            inds_to_get = [0, min_ind, max_ind] if get_first else [min_ind, max_ind]
            key_labels = ['first','min','max'] if get_first else ['min','max']
            user_data_raw = [user_data_raw[i] for i in inds_to_get]

            user_data = user_data_to_dicts(user_data_raw, dataset_name, user_data_file)

            for i, (data, key_label) in enumerate(zip(user_data, key_labels)):
                print(f'  update: {i+1}/{len(user_data)}')
                start = time.time()
                lr_acc = 0 #get_logistic_regression_accuracy(Q, train_dev_corpus, test_corpus,
                           #                              data['anchor_vectors'], attr_name)
                end = 0 #time.time() - start
                data['lr_time'] = end
                data['lr_acc'] = lr_acc


                start = time.time()
                fc_acc = get_fc_test_acc(Q, labels, train_dev_corpus,
                                         test_corpus, data['anchor_vectors'],
                                         attr_name)
                end = time.time() - start
                data['fc_acc'] = fc_acc
                data['fc_time'] = end
                data_dict[dataset_name].append(data)
                if key_label == 'first':
                    get_first = False
                data[key_label] = True

    drop_dups(data_dict)
    try:
        with open('data_dict_with_maxes.pickle', 'wb') as outfile:
            pickle.dump(data_dict, outfile)
    finally:
        return data_dict

def drop_dups(data_dict):
    print('dropping duplicates')
    for dataset_name in ['yelp', 'amazon', 'tripadvisor']:
        dataset_data = data_dict[dataset_name]
        accs = set()
        to_remove = []
        for data in dataset_data:
            if not data.get('max'):
                data['max'] = False
            if not data.get('first'):
                data['first'] = False
            if not data.get('min'):
                data['min'] = False
        dataset_data.sort(key=lambda d: d.get('max'), reverse=True)
        for d, data in enumerate(dataset_data):
            if (data['fc_dev_acc'], data['fc_acc']) in accs:
                if data.get('max') or data.get('first') or data.get('min'):
                    continue
                to_remove.append(d)
            else:
                accs.add((data['fc_dev_acc'], data['fc_acc']))
        to_remove.sort(reverse=True)
        print(f'dropping {len(to_remove)} from {dataset_name}')
        for ind in to_remove:
            dataset_data.pop(ind)

def plot_user_data(user_data, x_label, y_label, ax=None,
                   xlim=None, ylim=None, xtext=None, ytext=None):
    if ax is None:
        fig, ax = plot.subplots()
    x = [data[x_label] for data in user_data if not (data.get('max'))]# or data.get('min'))]
    y = [data[y_label] for data in user_data if not (data.get('max'))]# or data.get('min'))]
    max_x = [data[x_label] for data in user_data if data.get('max')]
    max_y = [data[y_label] for data in user_data if data.get('max')]
    start_x = [data[x_label] for data in user_data if data['first']]
    start_y = [data[y_label] for data in user_data if data['first']]
    min_x = [data[x_label] for data in user_data if data.get('min')]
    min_y = [data[y_label] for data in user_data if data.get('min')]
    ax.scatter(x, y, s=30, alpha=.40)
    print('dev first', start_x, 'test_first', start_y)
    #ax.scatter(min_x, min_y, marker='o', s=30, alpha=.90)
    #ax.scatter(max_x, max_y, marker='+', s=120, alpha=.90)
    #ax.scatter(start_x, start_y, marker='*', s=500, alpha=1.00,
    #ax.scatter(min_x, min_y, c='deep', marker='o', s=30, alpha=.90)
    ax.scatter(max_x, max_y, color='red', marker='+', s=1000, alpha=1.00,
    edgecolors='black')
    ax.scatter(start_x, start_y, color='k', marker='*', s=1000, alpha=1.00,
    edgecolors='black')

    all_data_x = np.array([data[x_label] for data in user_data])
    all_data_y = np.array([data[y_label] for data in user_data])
    fit = np.polyfit(all_data_x, all_data_y, deg=1)
    #ax.plot(all_data_x, fit[0]*all_data_x + fit[1], color='purple')
    print(np.corrcoef(all_data_x, all_data_y))
    #print(np.corrcoef(max_x, max_y))

    if xtext is None:
        xtext = x_label
    if ytext is None:
        ytext = y_label

    ax.set_xlabel(xtext, fontsize=40)
    ax.set_ylabel(ytext, fontsize=40)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return ax

def get_average_improvement(dataset_name, user_data):
    dataset_data = user_data[dataset_name]

    maxes = [data for data in dataset_data if data.get('max')]
    maxes = sorted(maxes, key=lambda d: d['fc_dev_acc'], reverse=True)[:10]
    n = len(maxes)

    start = [data for data in dataset_data if data.get('first')][0]

    average_fc_dev = sum(data['fc_dev_acc'] for data in maxes)/n
    average_test_fc = sum(data['fc_acc'] for data in maxes)/n

    fc_dev_improvement = average_fc_dev - start['fc_dev_acc']
    fc_test_improvement = average_test_fc - start['fc_acc']
    print('Average dev improvement', fc_dev_improvement)
    print('Average test improvement', fc_test_improvement)


def get_processed_data():
    """Gets the input for the paper to be used in the plot_all_users_data
    function"""
    folder = 'data/emnlp2018_userstudy/'
    data = []
    for filename in os.listdir(folder):
        if filename.startswith('processed'):
            with open(os.path.join(folder,filename), 'rb') as infile:
                data += pickle.load(infile)
    return {'amazon':data}

def get_emnlp_user_study():
    with open('data/data_for_EMNLP_user_study.pickle', 'rb') as infile:
        data = pickle.load(infile)
    return data

def plot_all_users_data(dataset_name, user_data, outfile_name='user_data.pdf'):
    dataset_data = user_data[dataset_name]
    #fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,15))
    fig, ax1 = plt.subplots(figsize=(15,10))
    acc_labels = ['fc_dev_acc', 'fc_acc']

    all_scores = [data[acc_label] for data in dataset_data for acc_label in acc_labels]

    bot, top = min(all_scores)-.01, max(all_scores)+.01
    #plot_user_data(dataset_data, 'fc_dev_acc', 'lr_acc', ax1, xlim=(bot,top), ylim=(bot,top))
    #plot_user_data(dataset_data, 'fc_dev_acc', 'fc_acc', ax2, xlim=(bot,top), ylim=(bot,top))
    #plot_user_data(dataset_data, 'fc_acc', 'lr_acc', ax3, xlim=(bot,top), ylim=(bot,top))

    if dataset_name == 'amazon':
        xlim = (.6, .73) # THESE WERE USED FOR THE PAPER WITH AMAZON
        ylim = (.62, .705)
        ax1.axhline(.669, color='k') #USED FOR PAPER AMAZON
        ax1.axhline(.71, color='r') #SupAnc Paper
        ax1.set_yticks([.62, .64, .66, .68, .70])
    elif dataset_name == 'yelp':
        ax1.axhline(.735, color='k') #Rough estimate for yelp 20 topics
        ax1.axhline(.77, color='r') #SupAnc Paper
        xlim = (.6, .83)
        ylim = (.62, .805)
    else: #Tripadvisor
        ax1.axhline(.726, color='k') #Rough estimate for tripadvisor 20 topics
        ax1.axhline(.776, color='r') #SupAnc Paper
        xlim = (.6, .83)
        ylim = (.62, .805)
    plot_user_data(dataset_data, 'fc_dev_acc', 'fc_acc', ax1, xlim=xlim,
                   ylim=ylim, xtext='Development Set Accuracy', ytext='Test Set Accuracy')
    #ax1.set_title(dataset_name, fontsize='30')
    ax1.tick_params(labelsize=30)
    plt.savefig(outfile_name, format='pdf')
    plt.show()

    return ax1
    get_x_y = lambda x_label, y_label : ([data[x_label] for data in dataset_data],
                                         [data[y_label] for data in dataset_data])

    x_label = 'fc_dev_acc'
    y_label = 'lr_acc'
    x, y = get_x_y(x_label, y_label)
    ax1.scatter(x, y)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_xlim(bot,top)
    ax1.set_ylim(bot,top)

    x_label = 'fc_dev_acc'
    y_label = 'fc_acc'
    x, y = get_x_y(x_label, y_label)
    ax2.scatter(x, y)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_xlim(bot,top)
    ax2.set_ylim(bot,top)

    x_label = 'fc_acc'
    y_label = 'lr_acc'
    x, y = get_x_y(x_label, y_label)
    ax3.scatter(x, y)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_xlim(bot,top)
    ax3.set_ylim(bot,top)
    ax1.set_title(dataset_name, fontsize='30')
    plt.show()

def get_all_user_data():
    user_data = {}
    for dataset_name in ['amazon', 'tripadvisor', 'yelp']:
        user_data.update(make_user_data(dataset_name))
    return user_data


def make_user_data(dataset_name):
    user_data = []
    base = f'UserData3/FinalAnchors/{dataset_name}/processed/'
    for filename in os.listdir(base):
        with open(base+filename, 'rb') as infile:
            data = pickle.load(infile)
        user_data += data
    return {dataset_name: user_data}

def make_split_user_data(dataset_name):
    user_data = []
    base = f'UserData3/FinalAnchors/{dataset_name}/processed/'
    for filename in os.listdir(base):
        with open(base+filename, 'rb') as infile:
            data = pickle.load(infile)
        user_data.append(data)
    return {dataset_name: user_data}


def combine_user_data():
    base = 'UserData3/FinalAnchors'
    user_data = {}
    for dataset_name in ['amazon', 'yelp', 'tripadvisor']:
        dataset_data = []
        path = os.path.join(base, dataset_name, 'processed')
        for filename in os.listdir(path):
            with open(path+filename, 'rb') as infile:
                dataset_data += pickle.load(infile)
        user_data[dataset_name] = dataset_data

def process_filename(filename, dataset_name, attr_name):
    corpus_pickle_path = 'UserData3'
    final_anchors_path = 'UserData3/FinalAnchors/'+dataset_name
    print('Getting corpus data...')
    with open(os.path.join(corpus_pickle_path, dataset_name+'.pickle'), 'rb') as infile:
        corpus_data = pickle.load(infile)

    (Q, labels, train_dev_ids, train_dev_corpus,
      train_ids, train_corpus, dev_corpus, dev_ids,
      test_ids, test_corpus, gs_anchor_vectors,
      gs_anchor_indices, gs_anchor_tokens) = corpus_data

    with open(os.path.join(final_anchors_path, filename), 'rb') as infile:
        user_data_raw = pickle.load(infile)

    n_updates = len(user_data_raw)

    user_data = user_data_to_dicts(user_data_raw, dataset_name, filename)
    user_data[0]['first'] = True

    max_ind = max(range(n_updates), key=lambda num: user_data[num]['fc_dev_acc'])
    user_data[max_ind]['max'] = True

    min_ind = min(range(n_updates), key=lambda num: user_data[num]['fc_dev_acc'])
    user_data[min_ind]['min'] = True

    for i, data in enumerate(user_data):
        print(f'  update: {i+1}/{len(user_data)}')
        start = time.time()
        fc_acc = get_fc_test_acc(Q, labels, train_dev_corpus,
                                 test_corpus, data['anchor_vectors'],
                                 attr_name)
        end = time.time() - start
        data['fc_acc'] = fc_acc
        data['fc_time'] = end
        print(' ', end)

    with open(os.path.join(final_anchors_path, 'processed', 'processed_'+filename), 'wb') as outfile:
        pickle.dump(user_data, outfile)

    return user_data
