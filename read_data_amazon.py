import os
import re
import logging
import pandas as pd
import numpy as np
import json
from pprint import pprint
from pymongo import MongoClient



class DataSet(object):
    def __init__(self, batch_size, idxs, xs, qs, ys, include_leftover=False, name=""):
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
def _tokenize(raw):
    pass

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
    for document in documents.find():
        # just put the answer into the document
        answers.append(document['answer'])
        # tokenize the question
        question.append(_tokenize_question(document['question']))
        # add question words into the vocab set
        vocab_set |= set(question)
        vocab_set.add(answer)

        # deal with the paragraph/story thing
        paragraph = _tokenize_story(document['description'])
        paragraphs.append(paragraph)
        # add to the vocab set inside the tokenize function?

    return vocab_set, paragraphs, questions, answers

    # in case i ever pass in multiple files instead of just one
    # for file_path in file_paths:
    #     with open(file_path, 'r') as fh:
    #         lines = fh.readlines()
    #         paragraph = []
    #         for line_num, line in enumerate(lines):
    #             # sm = _s_re.match(line) # matches pattern of a sentence
    #             # qm = _q_re.match(line) # matches pattern of a question
    #
    #             # if it is a question
    #             if qm:
    #                 id_, raw_question, answer, support = qm.groups()
    #                 question = _tokenize(raw_question)
    #                 paragraphs.append(paragraph[:])
    #                 questions.append(question)
    #                 answers.append(answer)
    #                 vocab_set |= set(question)
    #                 vocab_set.add(answer)
    #
    #             # if it is a sentence in the paragraph/story
    #             elif sm:
    #                 id_, raw_sentence = sm.groups()
    #                 sentence = _tokenize(raw_sentence) # tokenize the sentence
    #                 if id_ == '1':
    #                     paragraph = []
    #                 paragraph.append(sentence)
    #                 vocab_set |= set(sentence)
    #             else:
    #                 logging.error("Invalid line encountered: line %d in %s" % (line_num + 1, file_path))
    #         print("Loaded %d examples from: %s" % (len(paragraphs), os.path.basename(file_path)))





''' Reading in amazon data '''
##TODO
# word vectors~~~~~
def read_amazon_split(batch_size, train_file_path, test_file_path):
    # calls read_amazon_files
    vocab_set_list, paragraphs_list, questions_list, answers_list = zip(*[read_amazon_db(file_paths) for file_paths in file_paths_list])
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
    xs_list = [[[[_get(vocab_map, word) for word in sentence] for sentence in paragraph] for paragraph in paragraphs] for paragraphs in paragraphs_list]
    qs_list = [[[_get(vocab_map, word) for word in question] for question in questions] for questions in questions_list]
    ys_list = [[_get(vocab_map, answer) for answer in answers] for answers in answers_list]

    data_sets = [DataSet(batch_size, list(range(len(xs))), xs, qs, ys)44
                 for xs, qs, ys in zip(xs_list, qs_list, ys_list)]
    # just for debugging
    for data_set in data_sets:
        data_set.vocab_map = vocab_map
        data_set.vocab_size = len(vocab_map)
    # return data_setst
    pass


# ''' reading in amazon data '''
# def read_amazon(batch_size):
#     # train_file_path = "" + file_name + "_train"
#     # test_file_path = "" + file_name + "_train"
#
#     return read_amazon_split(batch_size, train_file_path, test_file_path)

def split_val(data_set, ratio):
    end_idx = int(data_set.num_examples * (1-ratio))
    left = DataSet(data_set.batch_size, list(range(end_idx)), data_set.xs[:end_idx], data_set.qs[:end_idx], data_set.ys[:end_idx])
    right = DataSet(data_set.batch_size, list(range(len(data_set.xs) - end_idx)), data_set.xs[end_idx:], data_set.qs[end_idx:], data_set.ys[end_idx:])
    return left, right


def get_max_sizes(*data_sets):
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
# print x_batch
