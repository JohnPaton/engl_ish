#from nltk import sent_tokenize, word_tokenize
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd
import random
#import newspaper
import pickle
import os
import seaborn as sns

class Distribution():
    '''
    A discrete distribution of states. Two distributions are available, an
    absolute distribution for tracking observations, and a normalized
    distribution which sums to one and is generate from the absolute dist.
    '''
    def __init__(self, abs_dict = {}):
        self.abs = deepcopy(abs_dict)
        self.norm = None
        self._normalized = False

    def increment(self, key, amount = 1):
        # Increment state key by amount in the absolute distribution
        # Add the key if it isn't already present
        if key in self.abs:
            self.abs[key] += amount
        else:
            self.abs[key] = amount

        self._normalized = False

    def replace_abs(self, new_abs):
        # Replace the abosolute distribution
        self.abs = deepcopy(new_abs)

    def normalize(self):
        # Create the normalized distribution based on the current absolute one
        # Values of the normalized distribution will always sum to one
        total = sum(self.abs.values())
        self.norm = {}
        for key in self.abs:
            self.norm[key] = self.abs[key]/total
        self._normalized = True

    def unzip_abs(self, max_key=None, sort=False):
        # Unzip the absolute distribution into matching lists of keys and values
        # optionally choose a maximum key value or sort keys
        keys = self.abs.keys()

        if sort:
            keys = sorted(keys)

        if max_key != None:
            keys = list(filter(lambda k: k <= max_key, keys))
        else:
            keys = list(keys)

        values = [self.abs[k] for k in keys]

        return keys, values

    def unzip_norm(self, max_key=None, sort=False):
        # Unzip the normal distribution into matching lists of keys and values
        # Update the normalized dist if it is not up to date
        # optionally choose a maximum key value  or sort keys
        if not self._normalized:
            self.normalize()

        keys = self.norm.keys()

        if sort:
            keys = sorted(keys)

        if max_key != None:
            keys = list(filter(lambda k: k <= max_key, keys))
        else:
            keys = list(keys)

        values = [self.norm[k] for k in keys]

        return keys, values

    def select(self, starts_with):
        # Create a new distribution by selecting only the keys that start with
        # the given value
        out = Distribution()
        n = len(starts_with)
        for k in self.keys():
            if str(k)[0:n] == starts_with:
                out.increment(k, self.abs[k])

        return out

    def state_to_dist(self, current_state):
        # Create a new distribution using the keys starting with current_state,
        # and summing over the next character in each of those keys to create
        # the new distribution. This is in effect the output of a markov model
        out = Distribution()
        n = len(current_state)
        for k in self.abs.keys():
            if k[0:n] == current_state:
                out.increment(k[n], self.abs[k])

        return out

    def plot(self, norm=True, max_key=-1, width=1, **kwargs):
        if norm and not self._normalized:
            self.normalize()

        if max_key >= 0:
            keys = sorted(self.keys())
            keys = list(filter(lambda k: k <= max_key, keys))
            if norm:
                vals = [self.norm[k] for k in keys]
            else:
                vals = [self.abs[k] for k in keys]
        else:
            keys, vals = self.unzip_norm() if norm else self.unzip_abs()

        plt.bar(keys, vals, width=width, **kwargs)
        plt.show()

    def total(self):
        return sum(self.abs.values())
    
    def keys(self):
        return list(self.abs.keys())     

    def eject(self, key):
        # Remove key from the distribution
        self.abs.pop(key)
        self._normalized = False

    def copy(self):
        # Return a copy of the distribution so that editing the copy will
        # not affect the original
        return deepcopy(self)

    def reset(self):
        # Reset the distribution to its initial state
        self.abs = {k:0 for k in self.abs}
        self.norm = None
        self._normalized = False

    def add_key(self, key):
        # Add a new key to the distribution with value zero
        # Equivalent to increment(key, 0)
        self.increment(key, 0)

    def draw(self):
        # Draw a value from the distribution using norm as the
        # probability distribution
        if not self._normalized:
            self.normalize()
        
        keys, values = self.unzip_norm()
        return np.random.choice(keys, p=values)


class Markov_Model():
    '''
    A class describing a Markov model. Each state corresponds to a Distribution
    describing the probabilities of moving to the next state.
    '''
    def __init__(self):
        self.model = {}
        self._normalized = False

    def from_keys(self):
        # The initial ("from"( states of the model
        return list(self.model.keys())

    def to_keys(self):
        # The final ("to") states of the model
        s = self.from_keys()
        return list(self.model[s[0]].keys()) if s else []

    def add_to(self, key):
        # Add a final ("to") state
        for k in self.model:
            self.model[k].add_key(key)
        self._normalized = False

    def add_from(self, key):
        # Add an initial ("to") state
        if key not in self.model:
            self._normalized = False
            self.model[key] = Distribution({k:0 for k in self.to_keys()})         

    def increment(self, i, f, amount = 1):
        # increment the transition i -> f by amount
        in_from = i in self.model
        if not in_from:
            self.add_from(i)

        in_to = f in self.model[i].abs
        if not in_to:
            self.add_to(f)
        
        self._normalized = False

        self.model[i].increment(f, amount)
            
    def copy(self):
        return deepcopy(self)

    def normalize(self):
        for k in self.model:
            self.model[k].normalize()

        self._normalized = True

    def get_val(self,i,f):
        return self.model[i].abs[f]

    def get_prob(self,i,f):
        if not self.model[i]._normalized:
            self.model[i].normalize()

        return self.model[i].norm[f]

    def draw(self, current):
        # draw a next state based on the current state
        return self.model[current].draw()

    def to_df(self):
        # return the probability model as a pandas dataframe, with from states as rows
        # and to states as columns

        if not self._normalized:
            self.normalize()

        from_keys = sorted(self.from_keys())
        to_keys = sorted(self.to_keys())

        data = np.zeros((len(from_keys), len(to_keys)))

        for i, from_key in enumerate(from_keys):
            for j, to_key in enumerate(to_keys):
                data[i,j] = self.model[from_key].norm[to_key]

        return pd.DataFrame(data, columns = to_keys, index = from_keys)



    def plot(self, linewidth = 0.1, **kwargs):
        # to do
        # plot the markov probability matrix as a heatmap
        # kwargs passed to seaborn.heatmap
        pass

class Language_Model():
    '''
    A class describing a complete language model. Train the model using the
    language_model function on a series of tokenized sentences.
    '''
    def __init__(self, order):
        self.order = order
        self.sent_lens = Distribution()
        self.word_lens = Distribution()
        self.char_counts = Distribution()
        self.firsts = [Distribution() for _ in range(order)]
        self.lasts = [Distribution() for _ in range(order)]
        self.markov_models = [Markov_Model() for _ in range(order)]
        self.mid_cap_prob = 0.0
        self.mid_punct_prob = 0.0
        self.end_puncts = Distribution()
        self.mid_puncts = Distribution()
        self.singles = Distribution()

    def restrict_order(self, order):
        # restrict the order of the model when generating text to compare results
        self.order = min(order, self.order)

    def reset_order(self):
        # reset to the highest available order
        self.order = len(self.markov_models)

    def copy(self):
        return deepcopy(self)

    def language_gen(self, num_sentences):
        # Generate a paragraph with the given number of sentences        
        sents = []
        
        while len(sents) < num_sentences:
            # determine sentence lengths from model
            sent_len = self.sent_lens.draw()
            sent = self.sentence_gen(sent_len)
            sents.append(sent)

        return ' '.join(sents)

    def sentence_gen(self, length):
        # Generate a sentence of the given length
        
        # shorten some names for convenience
        word_lens = self.word_lens
        mid_cap_prob = self.mid_cap_prob
        mid_punct_prob = self.mid_punct_prob
        mid_puncts = self.mid_puncts
        end_puncts = self.end_puncts

        words = []

        # build the sentence
        while len(words) < length:
            current_len = len(words)

            # generate a new word with length from distribution
            word_len = word_lens.draw()
            word = self.word_gen(word_len)
            
            # capitalize word if it's the first in the sentence
            if current_len == 0:
                word = capitalize(word)

            # capitalize mid-sentence word with probability from model
            elif random.uniform(0,1) < mid_cap_prob:
                word = capitalize(word)

            # end sentence with punctuation from model
            if current_len == length-1:
                word += end_puncts.draw()

            # add mid-sentence punctuation with probability from model
            elif random.uniform(0,1) < mid_punct_prob:
                word += mid_puncts.draw()

            # don't repeat words
            if current_len == 0:  
                words.append(word)
            elif word.lower() != words[-1].lower():
                words.append(word)

        # return the sentence
        return ' '.join(words)

    def word_gen(self, length):
        # Generate a word of the given length from the model
        
        # shorten some names for convenience
        models = self.markov_models
        firsts = self.firsts
        lasts = self.lasts
        singles = self.singles
        order = self.order

        vowels = {'a','e','i','o','u'}

        # special case for single-letter words        
        if length == 1:
            return singles.draw()

        # if we can already form a word out of a single state, do so
        if length <= order:
            # consider both starting and ending states
            d = modulate_dist(firsts[length-1],lasts[length-1])
            word = d.draw()

            # make sure the word contains a vowel
            if vowels.intersection(set(word)):
                return word
            else:
                return self.word_gen(length)

        # start the word by drawing an initial state
        word = firsts[order-1].draw()

        # follow the Markov chain to generate the rest of the word
        while len(word) < length:
            current_order = order
            current_state = word[-current_order:]

            new_letter = ''

            # try to choose the next letter
            while not new_letter:
                # get the appropriate distribution from the Markov model
                if current_state in models[current_order-1].model:
                    d = models[current_order-1].model[current_state]

                    # if we're close to the end of the word, take endings
                    # into account
                    try:
                        if length-len(word) < current_order:
                            n_tail = current_order + len(word) - length
                            tail = word[-n_tail:]
                            tail_dist = lasts[current_order].state_to_dist(tail)
                            d = modulate_dist(d, tail_dist)
                            
                        new_letter = d.draw()

                    # if we can't match up the distributions, try again with a
                    # shorter state
                    except:
                        current_order -= 1
                        current_state = word[-current_order:]
                else:
                    current_order -= 1
                    current_state = word[-current_order:]

            word += new_letter

        # make sure the word contains a vowel
        if vowels.intersection(set(word)):
            return word
        else:
            return self.word_gen(length)
        
    def to_df(self, order=1):
        return self.markov_models[order-1].to_df()
        

   

def draw_from(dist, current_state):
    selected = dist.select(current_state)
    

def add_dist(a,b):
    # add two distributions together
    c = a.copy()
    for k in b.keys():
        c.increment(k, b.abs[k])

    c._normalized = False

    return c

def add_markov(a,b):
    # add the absolute distributions of two markov models together
    c = a.copy()
    for f in b.from_keys():
        for t in b.to_keys():
            c.increment(f,t,b.model[f][t])
    return c
        

def load_sentences(text, language='english'):
    # tokenize a block of text into a nested list for training
    sent_list = sent_tokenize(text)
    sents = [word_tokenize(sent, language) for sent in sent_list]
    return sents
    

def modulate_dist(a,b):
    # modulate two distributions by multiplying them together and renormalizing
    c = Distribution()

    d = {}
    # multiply or add items from a
    for k, v in a.abs.items():
        try:
            d[k] = v*b[k]
        except:
            d[k] = v

    # add remaining items from b
    for k, v in b.abs.items():
        if k not in d:
            d[k] = v

    c = Distribution(d)

    c.normalize()

    return c

def capitalize(s):
    # capitalize a word 
    out = list(s)
    out[0] = out[0].upper()
    return ''.join(out)
        
            

def language_model(sents, order):
    # train a Language_Model of the given order using sents
    # sents should be a list of sentences generated by load_sentences or nltk's
    # tokenizers
    model = Language_Model(order)

    # counts of sentence end punctuation
    periods = 0
    questions = 0
    exclams = 0

    # counts of punctuation and capitalization mid-sentence
    mid_cap_count = 0
    mid_word_count = 0
    mid_punct_count = 0
    word_count = 0

    # when to display progress
    tenth_of_sents = np.floor(len(sents)/10)

    # gather the information about each sentence
    for s_ind, s in enumerate(sents):
        # display the progress
        if s_ind % tenth_of_sents == 0:
            print('Training progress:', int(np.round(100*(s_ind+1)/len(sents))),'percent')

        # count the number of words (not including punctuation tokens)
        s_len = 0
        for i, w in enumerate(s):
            if i!=0:
                mid_word_count += 1
                if w.isalpha() and not w.islower():
                    mid_cap_count += 1
                    
            if w.isalpha():
                word_count += 1
                s_len += 1
                model.word_lens.increment(len(w))

                if len(w) == 1:
                    model.singles.increment(w)
                    
            if w.isalpha():
                word = w.lower()
                alph = True
            else:
                word = ''
                alph = False

            for j in range(len(w)):
                if alph:
                    for o in range(order):
                        if j>o:
                            model.markov_models[o].increment(word[j-(o+1):j],word[j])
                        if j == o:
                            model.firsts[o].increment(word[:(o+1)])
                        if j == len(w)-(o+1):
                            model.lasts[o].increment(word[-(o+1):])
                                            
                model.char_counts.increment(w[j])
                
            if i == len(s) - 1 and w in ['.','!','?']:
                model.end_puncts.increment(w)

            if w in [',',';',':']:
                model.mid_puncts.increment(w)
                mid_punct_count += 1
                    
        
        model.sent_lens.increment(s_len)

    model.mid_cap_prob = mid_cap_count/mid_word_count
    model.mid_punct_prob = mid_punct_count/word_count
    
    return model
    
def word_gen(length, model):
    models = model.markov_models
    firsts = model.firsts
    lasts = model.lasts
    singles = model.singles
    order = model.order

    vowels = {'a','e','i','o','u'}
    
    if length == 1:
        return singles.draw()

    if length <= order:
        d = modulate_dist(firsts[length-1],lasts[length-1])
        word = d.draw()

        if vowels.intersection(set(word)):
            return word
        else:
            return word_gen(length, model)

    word = firsts[-1].draw()

    while len(word) < length:
        current_order = order
        current_state = word[-current_order:]

        new_letter = ''

        while not new_letter:
            if current_state in models[current_order-1].from_keys():
                d = models[current_order-1].model[current_state]
                
                try:
                    if length-len(word) < current_order:
                        n_tail = current_order + len(word) - length
                        tail = word[-n_tail:]
                        tail_dist = lasts[current_order].state_to_dist(tail)
                        d = modulate_dist(d, tail_dist)
                        
                    new_letter = d.draw()
                    
                except:
                    current_order -= 1
                    current_state = word[-current_order:]
            else:
                current_order -= 1
                current_state = word[-current_order:]

        word += new_letter

    if vowels.intersection(set(word)):
        return word
    else:
        return word_gen(length, model)
    
def sentence_gen(length, model):
    word_lens = model.word_lens
    mid_cap_prob = model.mid_cap_prob
    mid_punct_prob = model.mid_punct_prob
    mid_puncts = model.mid_puncts
    end_puncts = model.end_puncts

    words = []
    
    while len(words) < length:
        current_len = len(words)
        
        word_len = word_lens.draw()
        word = word_gen(word_len,model)

        # capitalization
        if current_len == 0:
            word = capitalize(word)
            
        elif random.uniform(0,1) < mid_cap_prob:
            word = capitalize(word)

        # puntcuation
        if current_len == length-1:
            word += end_puncts.draw()
        elif random.uniform(0,1) < mid_punct_prob:
            word += mid_puncts.draw()

        if current_len == 0:  
            words.append(word)
        elif word.lower() != words[-1].lower():
            words.append(word)

    return ' '.join(words)

def language_gen(num_sentences, model):
    sent_lens = model.sent_lens
    
    sents = []
    
    while len(sents) < num_sentences:
        sent_len = sent_lens.draw()
        sent = sentence_gen(sent_len, model)
        sents.append(sent)

    return ' '.join(sents)


def download_newspaper(news_urls, language = 'english', pickle_sents = True):
    if type(news_urls) is str:
        urls = [news_urls]
    else:
        urls = news_urls

    sents_parsed = []

    for url in urls:
        source = newspaper.build(url, memoize_articles=False, keep_images=False)
        print('downloading articles from', url)
        source.download()
        source.download_articles(2)
        print(len(source.articles),'articles downloaded')
        print('parsing html')
        source.parse()
        source.parse_articles()
        
        arts = [article.text for article in source.articles]
        
        text = '\n\n'.join(arts)
        
        sents_parsed += load_sentences(text, language)
        print(len(sents_parsed),'sentences parsed')
        
    if pickle_sents:
        outfile = '\\sources\\'+language+'_newspaper_'+str(len(sents_parsed))+'_source.pickle'

        with open(outfile, 'wb') as h:
            pickle.dump(sents_parsed, h)
    
    return sents_parsed

def load_source(filename):
    filepath = os.path.join(os.getcwd(), 'sources',filename)
    with open(filepath,'rb') as h:
        sents = pickle.load(h)

    return sents
    
def load_model(filename):
    filepath = os.path.join(os.getcwd(), 'models',filename)
    with open(filepath,'rb') as h:
        model = pickle.load(h)

    return model

def train_model(sents, language, order=3,pickl=False):
    print('building model')
    model = language_model(sents, order)

    if pickl:
        outfile = language+'_'+str(order)+'_newspaper_'\
                  +str(len(sents))+'.pickle'

        filepath = os.path.join(os.getcwd(), 'models', outfile)
        with open(filepath, 'wb') as h:
            pickle.dump(model, h)
   
    return model

def train_pickle_model(sents, language, order=3):
    import trainer
    return trainer.train_pickle_model(sents, language, order)









