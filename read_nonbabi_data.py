import os
import re
import logging
from pprint import pprint
import numpy as np
from collections import defaultdict

class DataSet(object):
    def __init__(self, batch_size, idxs, xs, qs, ys, include_leftover=False, name=""):
        # assert len(xs) == len(qs) == len(ys), "X, Q, and Y sizes don't match."
        print "batch size: ", batch_size
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

# TODO : split data into val

def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw) # finds every word
    normalized_tokens = [token.lower() for token in tokens]
    return normalized_tokens



_s_re = re.compile("^F:")
_q_re = re.compile("^Q:")
_a_re = re.compile("^A:")


'''
called from read_babi_split
defines the vocabulary, paragraphs (x input), questions and answers
'''
def read_babi_files(file_paths):

    vocab_set = set()
    paragraphs = []
    questions = []
    answers = []

    # create the w2v dictionary
    with open('/Users/williamcosby/Documents/metis/Passion_Project_Stratus/data/glove/glove.6B/glove.6B.50d.txt', 'r') as f:
        content = f.readlines()
    # content is a list of the word vectors, so need to split each line and get the word
    w2v_dict = defaultdict(list)
    # initialize an unkown word at 0s for the size of the embedding dim
    w2v_dict['<UNK>'] = [0 for i in range(50)]
    for vec in content:
        split_vec = vec.split()
        num_list = []
        for num in split_vec[1:]:
            num_list.append(float(num))
        # vocab_set.add((split_vec[0],num_list))
        w2v_dict[split_vec[0]] = num_list
    # print("WORD2VEC DICT: ", type(w2v_dict))
    # print("WORD2VEC ACCESS: ", w2v_dict['unk'])

    # populate the vocab set with words in the dictionary
    # NOTE the bottom code populates the vocab set with all the words in w2v
    # for key in w2v_dict.keys():
    #     vocab_set.add(key)

    for file_path in file_paths:
        with open(file_path, 'r') as fh:
            lines = fh.readlines()
            paragraph = []
            for line_num, line in enumerate(lines):
                line = line.strip('\n')
                sm = _s_re.match(line) # matches pattern of a sentence
                qm = _q_re.match(line) # matches pattern of a question
                am = _a_re.match(line)

                # if it is a question, peel off the 'Q: ' beginning part
                if qm:
                    raw_question = line[2:].strip() # should start from the space after Q:
                    question = _tokenize(raw_question)
                    # for word in question:
                    #     if word not in w2v_dict.keys():
                    #         w2v_dict[word] = [0 for i in range(50)]
                    questions.append(question)
                    vocab_set |= set(question)
                    # now that we've hit a question we know we're at the end of the "story"
                    # add the paragraph so far to the paragraphs list
                    paragraphs.append(paragraph)
                    paragraph = [] # clear the paragraph to start adding new things to it on the next time we hit a sentence

                # if it is a sentence/part of the paragraph, peel off the 'A: ' part
                elif sm:
                    raw_sentence = line[2:].strip()
                    sentence = _tokenize(raw_sentence)
                    # for word in sentence:
                    #     if word not in w2v_dict.keys():
                    #         w2v_dict[word] = [0 for i in range(50)]
                    paragraph.append(sentence)
                    vocab_set |= set(sentence)
                # line represents an answer
                elif am:
                    answer = line[2:].strip()
                    answers.append(answer)
                    # for word in answer:
                    #     if word not in w2v_dict.keys():
                    #         w2v_dict[word] = [0 for i in range(50)]
                    vocab_set.add(answer)
                else:
                    logging.error("Invalid line encountered: line %d in %s" % (line_num + 1, file_path))

            print("Loaded %d examples from: %s" % (len(paragraphs), os.path.basename(file_path)))

    return w2v_dict, vocab_set, paragraphs, questions, answers


''' called in return statement of read_babi '''
def read_babi_split(batch_size, *file_paths_list):
    # calls read_babi_files
    w2v_dict, vocab_set_list, paragraphs_list, questions_list, answers_list = zip(*[read_babi_files(file_paths) for file_paths in file_paths_list])
    vocab_set = vocab_set_list[0]
    w2v_dict = w2v_dict[0]
    # print("TESTING THE INITIAL ZIP THING: ", type(w2v_dict))
    # print(w2v_dict['unk'])
    # index as key and word as value
    # idx_to_word = dict((k,v) for k, (word, vector) in enumerate(sorted(vocab_set)))
    # idx_to_word[0] = "<UNK>"
    # word as key and index as value this needs to span the entire word vector set

    # need to construct a dictionary of {word: index} and a list of the word vectors as they appear
    vocab_map = {}
    w2v_vectors = []
    idx_to_word = {}
    for idx, word in enumerate(sorted(vocab_set)):
        # if idx == 0:
            # print("WORD TEST: ", word)
            # print("type fo w2v_dict: ", type(w2v_dict))
            # print("one last test: ", w2v_dict['unk'])
        vocab_map[word] = idx
        w2v_vectors.append(w2v_dict[word])
        idx_to_word[idx] = word
    #
    # vocab_map = dict((word, k) for k, (word, vector)  in enumerate(sorted(vocab_set))) # this is word -> index (i think) with '<UNK>' as index=0
    # vocab_map["<UNK>"] = 0
    # print "vocab_size: ",len(vocab_map)
    # print idx_to_word


    ''' get the index of the word, return index for <UNK> token if word is not in the vocabulary '''
    #TODO add the word vector look ups right here...return the vector instead of the index
    def _get(vm, w): # w = word, vm = vocabulary_map
        if w in vm:
            return vm[w]
        return 0

    ''' this is basically the final step in making the data sets '''
    # TODO word2vec or glove vectors here instead of just indices
    ## Makes the inputs to the networks
    xs_list = [[[[_get(vocab_map, word) for word in sentence] for sentence in paragraph] for paragraph in paragraphs] for paragraphs in paragraphs_list]
    qs_list = [[[_get(vocab_map, word) for word in question] for question in questions] for questions in questions_list]
    ys_list = [[_get(vocab_map, answer) for answer in answers] for answers in answers_list]

    # data sets are now a list of word vectors for the sentences instead of list
    # of indices

    data_sets = [DataSet(batch_size, list(range(len(xs))), xs, qs, ys)
                 for xs, qs, ys in zip(xs_list, qs_list, ys_list)]
    # print "datasets: ",len(data_sets)
    # just for debugging
    for data_set in data_sets:
        data_set.vocab_map = vocab_map
        data_set.vocab_size = len(vocab_map)
    # print "WORD VECTORS!!!!!!!!: ", len(w2v_vectors),len(w2v_vectors[0])
    w2v_vectors = np.array(w2v_vectors)
    print("VECTOR TYPE: ",type(w2v_vectors))
    print("VECTOR TYPE: ",type(w2v_vectors[0][0]))
    print("VECTOR SHAPE: ",w2v_vectors.shape)
    return data_sets, idx_to_word, np.array(w2v_vectors)


''' reading in babi data '''
def read_babi(batch_size, dir_path, task, suffix=""):
    prefix = "%s" % str(task)
    train_file_paths = []
    test_file_paths = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name.startswith(prefix) and file_name.endswith(suffix + "_train.txt"):
        # if file_name.endswith(suffix + "_train.txt"):
            train_file_paths.append(file_path)
        elif file_name.startswith(prefix) and file_name.endswith(suffix + "_test.txt"):
        # elif file_name.endswith(suffix + "_test.txt"):
            test_file_paths.append(file_path)


    ''' calls read_babi_split '''
    return read_babi_split(batch_size, train_file_paths, test_file_paths)


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
    train, test = read_babi(1, "data/tasks_1-20_v1-2/en", 1)
    # print train.vocab_size, train.max_m_len, train.max_s_len, train.max_q_len
    # print test.vocab_size, test.max_m_len, test.max_s_len, test.max_q_len
    x_batch, q_batch, y_batch = train.get_next_labeled_batch()
    max_sent_size, max_ques_size = get_max_sizes(train, test)
    print(max_sent_size, max_ques_size)
# print x_batch
