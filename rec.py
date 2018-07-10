import pandas as pd
import jieba.analyse
import nltk
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import operator
import heapq
import datetime
import bisect
from random import choice
from random import randint
from collections import Counter

class Rec:
    def __init__(self):
        jieba.analyse.set_stop_words("data/stopWord.txt")
        self.word2vec = KeyedVectors.load_word2vec_format('/home/young/AnacondaProjects/keras/embedding/wiki.zh.vec', binary=False)
        tags_file = open('tags_dict.txt', 'r')
        self.tags_dict = eval(tags_file.read())
        self.scene_dict = {}
        df = pd.read_csv('tourist_cndbpedia5.csv')
        title_list = df['title']
        detail_list = df['detail']
        intro_list = df['intro2']
        for i in range(len(title_list)):
            self.scene_dict[title_list[i]] = str(detail_list[i]) + str(intro_list[i])
        self.old_scene_dict = self.scene_dict
        self.scene_dict = {}
        for key in self.old_scene_dict.keys():
            self.scene_dict[key] = self.getkeyword_from_tr(self.old_scene_dict[key], 25)
        print('Finished loading data!')

    def getkeyword_from_tr(self, text, topk):
        keywords = jieba.analyse.textrank(text, topK=topk, allowPOS=('n','nz','v','vd','vn','l','a','d'))
        return keywords

    def jaro_distance(self, string1, string2):

        if len(string1) < len(string2):
            longerString = string2
            shorterString = string1
        else:
            longerString = string1
            shorterString = string2
        
        # Should be rounded down
        allowedRange = (len(longerString) // 2) - 1
        mappingIndices = [-1] * len(shorterString)
        shortMached = []
        longMatched = []
        matches = 0

        for index, char in enumerate(shorterString):
            for secondIndex in range(max(0, index - allowedRange), min(len(longerString), index + allowedRange + 1)):
                if char == longerString[secondIndex]:
                    matches = matches + 1
                    mappingIndices[index] = secondIndex
                    shortMached.append(char)
                    longMatched.insert(secondIndex, char)
                    break

        halfTranspositions = 0
        for naturalIndex in range(0, len(shortMached)):
            if (mappingIndices[naturalIndex] != naturalIndex)  & (shortMached[naturalIndex] != longMatched[naturalIndex]):
                halfTranspositions = halfTranspositions + 1
        
        # The Jaro Similarity simj of two given strings s1 and s1 is:
        # simj = 0, if m = 0
        # simj = (1/3)(m/|s1| + m/|s2| + (m-t)/m), otherwise
        # where:
        # |S1| is the length of s1
        # m is the number of matching characters (see below);
        # t is half the number of transpositions (see below).
        # Two characters from s1 and s2 respectively, 
        # are considered matching only if they are the same and not farther than max(|s1|,|s2|)/2 - 1
        # Each character of s1 is compared with all its matching characters in s2. 
        # The number of matching (but different sequence order) characters divided by 2 defines the number of transpositions.
        
        if matches == 0:
            return 0
        return ((matches / len(longerString)) + (matches / len(shorterString)) + ((matches - (halfTranspositions // 2))/matches)) / 3

    def jaro_winkler_distance(self, string1, string2):
        jaro = self.jaro_distance(string1, string2)
        prefix = 0
        for index, char in enumerate(string1[:4]):
            if char == string2[index]:
                print(char)
                prefix = prefix + 1
            else:
                break

        if (jaro > 0.7):
            return jaro + ((prefix * 0.1) * (1 - jaro))
        else:
            return jaro_distance

    def weight_choice(self, weight):
        weight_sum = []
        sum = 0
        for a in weight:
            sum += a
            weight_sum.append(sum)
        t = randint(0, sum - 1)
        return bisect.bisect_right(weight_sum, t)

    def recProcess_keyword_based(self, keywords, keywords_tags):
        if len(keywords) == 1:
            keywords_tags.append('普通')
        w_list = []
        w = []
        c = Counter(keywords_tags)
        for tag in list(c):
            w_list.append(tag)
            w.append(c[tag])
        one = w_list[self.weight_choice(w)]
        
        # outside initialization #
        result_list = {}
        keywords_string = ''
        length_keywords = len(keywords) 
        for keyword in keywords:
            keywords_string += self.old_scene_dict[keyword]
            
        # the number of keywords increasas along with the nunmber of scenes #
        keywords_list =  self.getkeyword_from_tr(keywords_string, 18+length_keywords)
        
        for key, value in self.scene_dict.items():
            flag = True
            if key in keywords:
                flag = False
            # inside initialization #
            inside_buffer = []
            mid_buffer = []
            other_keywords = self.scene_dict[key]
            tags = self.tags_dict[key]
            
            # try recomend scences only from the same type #
            intersection_list = list(set(tags).intersection(set(keywords_tags)))
            if len(intersection_list) > 0 and one in tags:
                # if the number of keywords is less than 10, the performance will be not enough #
                if len(other_keywords) > 10:
                    for lk in keywords_list:
                        for ok in other_keywords:
                            try:
                                # word2vec distance (can be replaced by other models) #
                                inside_buffer.append(self.word2vec.distance(lk, ok))
                            except:
                                # jaro wrinkle distance (can be replaced by other string distances) #
                                inside_buffer.append(self.jaro_distance(lk,ok))
                        if inside_buffer:
                            # selecting the highest similarity as a simgle tag's feature #
                            mid_buffer.append(max(inside_buffer))
                        inside_buffer = []
                    # selecting the top 5 similarities as overall features #
                    final_scores = heapq.nlargest(5, mid_buffer)
                    # calculating the mean score as final measurement #
                    final_scores_mean = np.mean(final_scores)
                    # saving the scores for sorting#
                    result_list[key] = final_scores_mean
                       
                    # clean #
                    inside_buffer = []
                    mid_buffer = []
        return result_list

    def sort_by_value(self, d): 
        items=d.items() 
        backitems=[[v[1],v[0]] for v in items] 
        backitems.sort() 
        return [ backitems[i][1] for i in range(0,len(backitems))]

    def find_keywords_range(self, keywords):
        return_list = []
        for keyword in keywords:
            return_list += self.tags_dict[keyword]
            return_list = list(set(return_list))
        return return_list

    def main(self, keywords):
        start = datetime.datetime.now()
        keywords_tags = self.find_keywords_range(keywords)
        sort_list_inside = self.sort_by_value(self.recProcess_keyword_based(keywords, keywords_tags))
        end = datetime.datetime.now()
        print (end-start)
        return sort_list_inside[:50]

keywords = ['故宫','象牙山']
rec = Rec()
rec.main(keywords)
