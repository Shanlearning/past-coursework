###############################################################################
# load data
f = open("C:\\Users\\zhong\\Dropbox\\idea\\MCMC project\\stories\\fanrenxiuxianchuan_wangyu.txt" , encoding="gbk")
s = f.read()
len(s)

###############################################################################
# words tokens
import jieba
import numpy as np
import re
dat = re.split("\n",s)
dat = [re.sub("\s","",item) for item in dat]
dat = np.delete(dat,np.where([item == "" for item in dat]))
dat2 = ''.join(dat)
all_tokens = list(jieba.cut(str(dat2)))

###############################################################################
# given a list of tokens, return a dictionary where key are the existing unique tokens
# and values are their counts; also return the size N of a genre's corpus
def get_uniCounts(all_tokens):
    unigram_table = {}
    for token in all_tokens:
        if token in unigram_table:
            unigram_table[token] += 1
        else:
            unigram_table[token] = 1
    return unigram_table, len(all_tokens)


uni = get_uniCounts(all_tokens)[0]
len(uni)

###############################################################################
def get_biCounts(all_tokens):
    uniCounts, length = get_uniCounts(all_tokens)
    bigram_table = {}
    num_bigrams = 0
    for x in range(0, length - 1):
        if all_tokens[x] in bigram_table:
            if all_tokens[x + 1] in bigram_table[all_tokens[x]]:
                bigram_table[all_tokens[x]][all_tokens[x + 1]] += 1
            else:
                bigram_table[all_tokens[x]][all_tokens[x + 1]] = 1
                num_bigrams += 1
        else:
            bigram_table[all_tokens[x]] = {}
            bigram_table[all_tokens[x]][all_tokens[x + 1]] = 1
            num_bigrams += 1
    return bigram_table, num_bigrams


table = get_biCounts(all_tokens)[0]


np.sum([len(table[key]) for key in table.keys()])






np.random.choice(keys, 10000, replace=True, p=probs)


#input min and max number of words to generate, a genre and the optional part
#of a sentence with default value ='', return a randomly generated sentence using bigrams
import random

def get_biSentence(min,max,table,sentence=''):
    print ("computing bigrams and generating random sentence:")
    length=len(sentence)
    if length==0:
        sentence=random.choice(list(table['韩立']))
    sentence_tokens= list(jieba.cut(str(sentence)))
    last_word=sentence_tokens[-1]
    for x in range(max):
        generating=True
        while (generating):
            if last_word in table:
                next=random.choice(list(table[last_word]))
            else:
                next=random.choice(list(table.keys()))
            generating=False
            if (next=='.' and len<min):
                generating=True
        sentence=sentence+' '+next
        if next=='.':
            return sentence
        length+=1
        last_word=next
    return sentence+'。'

get_biSentence(5,50,table)

def generate_english_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)   
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]   
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]
