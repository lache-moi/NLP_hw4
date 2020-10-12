from collections import Counter
from math import log, sqrt

f1 = 'Cranfield_collection_HW/cran.qry'
f2 = 'Cranfield_collection_HW/cran.all.1400'
output_file = 'Cranfield_collection_HW/output.txt'
closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

# reads file and splits it into list of Bag of Words
def split_documents(filename):
    entries = []
    with open(filename) as f:
        lines = f.readlines()
        read_entry = False
        read_title = False
        entry = []

        for line in lines:
            if line:
                line_list = line.split()

                if line_list[0] == '.I':
                    entries.append(entry)
                    entry = []
                    read_entry = False
                elif line_list[0] == '.A':
                    read_title = False
                elif read_entry or read_title:
                    entry.extend([check_word(i) for i in line_list if check_word(i)])

                elif line_list[0] == '.W':
                    read_entry = True
                elif line_list[0] == '.T':
                    read_title = True

        entries.append(entry)
        entries.pop(0)
        # print(entries)
        # print(len(entries))
        f.close()
        return entries

# takes list of Bag of Words and returns list of tally of words in each BoW and
def process_entries_list(entries_list):

    word_counter = []
    word_set = set()

    for entry in entries_list:
        entry_tally = Counter()
        for word in entry:
            entry_tally[word] += 1
            word_set.add(word)
        word_counter.append(entry_tally)
    # print(len(word_set))
    return word_counter, word_set

def tf_idf_vectors(vocabulary, tally):
    docs_containing_word = Counter()
    tf_idf_list = []
    for entry in tally:
        for word in vocabulary:
            if word in entry:
                docs_containing_word[word] += 1

    for entry in tally:
        tf_idf = []
        for word in vocabulary:
            if word not in entry:
                tf_idf.append(0)
            else:
                tf_idf.append(entry[word] * log(len(tally)/docs_containing_word[word]))
        tf_idf_list.append(tf_idf)
    # print(len(tf_idf_list))
    return tf_idf_list

def output_search_result(output_file, query_tf_idf, article_tf_idf):
    outF = open(output_file, "w")
    for i in range(len(query_tf_idf)):
        scores = []
        for j in range(len(article_tf_idf)):
            scores.append((cos_similarity(query_tf_idf[i],article_tf_idf[j]), i+1, j+1))
        # print(scores)
        sorted_scores = sorted(scores, key = zero_index, reverse= True)
        for score in sorted_scores:
            outF.write(f"{score[1]} {score[2]} {score[0]}\n")

    outF.close()

# taken from https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
def cos_similarity(v1, v2):
    sumxx, sumyy, sumxy = 0,0,0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    if sumxx * sumyy == 0:
        return 0
    return sumxy/sqrt(sumxx * sumyy)

def check_word(word):
    start = 0
    end = len(word)

    if word.isnumeric() or word in closed_class_stop_words:
        return

    while start < len(word) and not word[start].isalpha():
        start += 1
    while end >= 0 and not word[end-1].isalpha():
        end -= 1

    if start >= len(word) or end < 0 or start >= end:
        return

    return word[start:end]

def zero_index(l1):
    return l1[0]

queries_entries_list = split_documents(f1)
articles_entries_list = split_documents(f2)

query_tally, query_set = process_entries_list(queries_entries_list)
article_tally, article_set = process_entries_list(articles_entries_list)
vocabulary = list(query_set)
query_tf_idf = tf_idf_vectors(vocabulary, query_tally)
article_tf_idf = tf_idf_vectors(vocabulary, article_tally)
output_search_result(output_file, query_tf_idf, article_tf_idf)
