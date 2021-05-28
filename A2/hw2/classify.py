'''
Ayuj Prasad
CS 540 - A2 - Spring 2021
'''
import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------

def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    out_of_vocab = 0
    file = open(filepath, 'r', encoding='utf-8')
    for word in file:
        word = word.rstrip('\n')
        if word in vocab and word in bow:
            bow[word] += 1
        elif word in vocab:
            bow[word] = 1
        else:
            out_of_vocab += 1
    if out_of_vocab > 0:
        bow[None] = out_of_vocab

    return bow


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    for listV in label_list:
        count = 0
        for element in training_data:
            if element['label'] == listV:
                count += 1      
        logprob[listV] = math.log((count + smooth)/(len(training_data)+2))

    return logprob


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}

    vocab_with_label = {}
    for word in vocab:
        vocab_with_label[word] = 0 
    vocab_with_label[None] = 0
    n = 0
    for element in training_data:
        if element['label'] == label: 
            for word, n_w in element['bow'].items():
                if word in vocab_with_label:
                    vocab_with_label[word] = vocab_with_label[word] + n_w
                else:
                    vocab_with_label[None] = vocab_with_label[None] + n_w

                n = n + n_w
    for word, n_w in vocab_with_label.items():
        word_prob[word] = math.log((n_w + smooth)/(n + smooth*(len(vocab)+1)))

    return word_prob


##################################################################################

def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)

    vocab = create_vocabulary(training_directory, cutoff)
    retval['vocabulary'] = vocab
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, ['2020','2016'])
    retval['log prior'] = log_prior
    prob_2016 = p_word_given_label(vocab, training_data, '2016')
    prob_2020 = p_word_given_label(vocab, training_data, '2020')
    retval['log p(w|y=2016)'] = prob_2016
    retval['log p(w|y=2020)'] = prob_2020

    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}

    p_label_2016 = model['log prior']['2016']
    p_label_2020 = model['log prior']['2020']
    current_bow = create_bow(model['vocabulary'], filepath)
    for word, n_w in current_bow.items():
        p_label_2016 += model['log p(w|y=2016)'][word] * n_w
        p_label_2020 += model['log p(w|y=2020)'][word] * n_w
    if p_label_2016 > p_label_2020:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'
    retval['log p(y=2016|x)'] = p_label_2016
    retval['log p(y=2020|x)'] = p_label_2020

    return retval