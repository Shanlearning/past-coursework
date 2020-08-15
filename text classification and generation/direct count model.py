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

all_tokens[200:230]
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

# total number of unique bigram
np.sum([len(table[key]) for key in table.keys()])

###############################################################################

def get_biSentence(min,max,table,sentence=''):
    print ("computing bigrams and generating random sentence:")
    length=len(sentence)
    if length==0:
        sentence=np.random.choice(list(table.keys()),1, replace=True, 
                 p=list(np.divide(list(uni.values()),sum(uni.values()))))[0]
    sentence_tokens= list(jieba.cut(str(sentence)))
    last_word=sentence_tokens[-1]
    for x in range(max):
        generating=True
        while (generating):
            if last_word in table:
                next=np.random.choice(list(table[last_word].keys()), 1, replace=True, 
                 p=list(np.divide(list(table[last_word].values()),sum(table[last_word].values()))))[0]
            else:
                next=np.random.choice(list(table.keys()),1, replace=True, 
                 p=list(np.divide(list(uni.values()),sum(uni.values()))))[0]
            generating=False
            if (next=='。' and length<min):
                generating=True
        sentence=sentence+' '+next
        if next=='.':
            return sentence
        length+=1
        last_word=next
    return sentence+'。'    
    
get_biSentence(5,50,table,sentence = '一只')
















