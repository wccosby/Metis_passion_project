import os
import re
import logging
import pandas as pd
import numpy as np
import json
from pprint import pprint
from pymongo import MongoClient
import nltk.data


class DataSet(object):
    def __init__(self, batch_size, idxs, xs, qs, ys, include_leftover=False, name=""):
        print "xs size: ", len(xs)
        print "qs size: ", len(qs)
        print "ys size: ", len(ys)
        assert len(xs) == len(qs) == len(ys), "X, Q, and Y sizes don't match."
        assert batch_size <= len(xs), "batch size cannot be greater than data size."
        self.name = name or "dataset"
        self.idxs = idxs
        self.num_examples = len(idxs)
        self.xs = xs
        self.qs = qs
        self.ys = ys
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.include_leftover = include_leftover
        self.num_batches = int(self.num_examples / self.batch_size) + int(include_leftover)
        self.reset()

    def get_next_labeled_batch(self):
        assert self.has_next_batch(), "End of epoch. Call 'complete_epoch()' to reset."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if self.include_leftover and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        xs, qs, ys = zip(*[[self.xs[i], self.qs[i], self.ys[i]] for i in cur_idxs])
        self.idx_in_epoch += self.batch_size
        return xs, qs, ys

    def has_next_batch(self):
        if self.include_leftover:
            return self.idx_in_epoch + 1 < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        np.random.shuffle(self.idxs)


'''
need this to take a raw block of text and then parse it into sentences (and do
any other data clearning that i decide needs to happen later on) and return
a list of lists (1 list where all sublists are the sentences)
'''
def _tokenize_question(raw):
    tokens = re.findall(r"[\w]+", raw) # finds every word
    normalized_tokens = [token.lower() for token in tokens]
    return normalized_tokens

def _tokenize_story(raw):
    '''
    will get a full story, need to break it into sentences and then words
    return a list of lists of the words in each sentence
    '''
    # break into list of sentences
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    story_sent_list = sent_detector.tokenize(raw.strip())

    # TODO break into words
    stories = []
    for sent in story_sent_list: # loop over all the sentences in the story
        tokens = re.findall(r"[\w]+", sent) # gets all the words in the sentence
        normalized_tokens = [token.lower() for token in tokens] # get list of the words
        stories.append(normalized_tokens) # add the list of words in the sentence to the list of sentences in the story

    return stories

'''
called from read_amazon_split
defines the vocabulary, paragraphs (x input), questions and answers
** db_auth should be a dictionary with the required database information
* client_address = str
* collection_name = str
* username = str
* password = str
'''
def read_amazon_db(db_auth):
    ## returns:
        # vocab_set --> set of all words in the vocabulary
        # paragraphs --> list of lists of lists of all stories
            # --> each sublist is a "story/paragraph/fact set"
                # --> each sub-sub-list is a sentence in that story/paragraph/fact set
        # questions --> list of all questions on the stories (should match 1 to 1 with the stories)
        # answers --> list of all answers to all the questions

    # connect to the server with the collections/databases
    client = MongoClient(db_auth['client_address'])

    # connect to the specific collection
    db = client[db_auth['collection_name']]

    # pass in credentials
    db.authenticate(db_auth['username'],db_auth['password'], source=db_auth['collection_name'])

    # initialize all the stuffz
    vocab_set = set()
    paragraphs = []
    questions = []
    answers = []

    # now need a reference to the right collection
    documents = db[db_auth['collection_name']]

    # do a loop over the data
    count = 0 # TODO remove this, its just for testing to make things run faster
    for document in documents.find():
        count += 1
        if count % 1000 == 0:
            break
        # just put the answer into the document
        if (type(document['facts']) == unicode):
            answer = document['answer']
            answers.append(answer)
            # tokenize the question
            question = _tokenize_question(document['question'])
            questions.append(question)
            # add question words into the vocab set
            vocab_set |= set(question)
            vocab_set.add(answer)

            # deal with the paragraph/story thing
            # paragraph is a list of lists
                # [[words,in,sentence,1],[words,in,sentence,2],...,[words,in,sentence,n]]
            paragraph = _tokenize_story(document['facts'])
            paragraphs.append(paragraph)
            # add to the vocab set inside the tokenize function?
            paragraph_list = []
            questions_list = []
            answers_list = []

            paragraph_list.append(paragraphs[:len(paragraphs)/2])
            paragraph_list.append(paragraphs[len(paragraphs)/2:])

            questions_list.append(questions[:len(questions)/2])
            questions_list.append(questions[len(questions)/2:])

            answers_list.append(answers[:len(answers)/2])
            answers_list.append(answers[len(answers)/2:])


    return list(vocab_set), paragraph_list, questions_list, answers_list


''' Reading in amazon data '''
##TODO
# word vectors~~~~~
def read_amazon_split(batch_size):
    '''
    Not using an exterior dictionary of words...only using the words that showed up
    in the training...can probably make this better
    '''
    # calls read_amazon_files
    # * client_address = str
    # * collection_name = str
    # * username = str
    # * password = str
    db_auth = {}
    db_auth['client_address'] = 'ds019936.mlab.com:19936'
    db_auth['collection_name'] = 'test_network'
    db_auth['username'] = 'wcc3af'
    db_auth['password'] = 'test_network'

    # vocab_set_list, paragraphs_list, questions_list, answers_list = zip(*[read_amazon_db(file_paths) for file_paths in file_paths_list])
    vocab_set_list, paragraphs_list, questions_list, answers_list = read_amazon_db(db_auth)

    print("paragraphs_list: ", len(paragraphs_list))
    print("questions_list: ", len(questions_list))
    print("answers_list: ", len(answers_list))

    vocab_set = vocab_set_list[0]
    vocab_map = dict((v, k+1) for k, v in enumerate(sorted(vocab_set))) # this is word -> index (i think) with '<UNK>' as index=0
    vocab_map["<UNK>"] = 0

    ''' get the index of the word, return index for <UNK> token if word is not in the vocabulary '''
    def _get(vm, w): # w = word, vm = vocabulary_map
        if w in vm:
            return vm[w]
        return 0

    ''' this is basically the final step in making the data sets '''
    # TODO word2vec or glove vectors here instead of just indices
    ## Makes the inputs to the networks (why u mke dis so complicated???? quadruple nested list comprehensions??? REALLY???)
    # TODO need to make this split into a train and test set
    xs_list = [[[[_get(vocab_map, word) for word in sentence] for sentence in paragraph] for paragraph in paragraphs] for paragraphs in paragraphs_list]
    qs_list = [[[_get(vocab_map, word) for word in question] for question in questions] for questions in questions_list]
    ys_list = [[_get(vocab_map, answer) for answer in answers] for answers in answers_list]

    print "xs size: ", len(xs_list)
    print "qs size: ", len(qs_list)
    print "ys size: ", len(ys_list)

    count_sets = 0
    for i,(xs, qs, ys) in enumerate(zip(xs_list,qs_list,ys_list)):
        count_sets += 1

    print "count sets: ",count_sets

    data_sets = [DataSet(batch_size, list(range(len(xs))), xs, qs, ys)
                 for xs, qs, ys in zip(xs_list, qs_list, ys_list)]

    # just for debugging
    for data_set in data_sets:
        data_set.vocab_map = vocab_map
        data_set.vocab_size = len(vocab_map)
    # return data_setst
    return data_sets


def split_val_amazon(data_set, ratio):
    end_idx = int(data_set.num_examples * (1-ratio))
    left = DataSet(data_set.batch_size, list(range(end_idx)), data_set.xs[:end_idx], data_set.qs[:end_idx], data_set.ys[:end_idx])
    right = DataSet(data_set.batch_size, list(range(len(data_set.xs) - end_idx)), data_set.xs[end_idx:], data_set.qs[end_idx:], data_set.ys[end_idx:])
    return left, right


def get_max_sizes_amazon(*data_sets):
    max_sent_size = max(len(s) for ds in data_sets for idx in ds.idxs for s in ds.xs[idx])
    max_ques_size = max(len(ds.qs[idx]) for ds in data_sets for idx in ds.idxs)
    return max_sent_size, max_ques_size


if __name__ == "__main__":
    train, test = read_babi(32, "data/tasks_1-20_v1-2/en", 1)
    # print train.vocab_size, train.max_m_len, train.max_s_len, train.max_q_len
    # print test.vocab_size, test.max_m_len, test.max_s_len, test.max_q_len
    x_batch, q_batch, y_batch = train.get_next_labeled_batch()
    max_sent_size, max_ques_size = get_max_sizes(train, test)
    print(max_sent_size, max_ques_size)
