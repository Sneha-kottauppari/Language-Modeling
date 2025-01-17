"""
Language Modeling Project
Name:
Roll No:
"""

from random import choices
import language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    fp=open(filename,'r')
    lines=fp.readlines()
    corpus_text=[]
    for each_line in lines:
        if len(each_line) > 1:
            inner_list=each_line.split()
            corpus_text.append(inner_list)
    return corpus_text 


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    corp_length=0
    for each_list in corpus:
        corp_length+= len(each_list)
    return corp_length


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    vocab_in_corpus=[]
    for each_list in corpus:
        for word in each_list:
            if word not in vocab_in_corpus:
                vocab_in_corpus.append(word)
    return sorted(vocab_in_corpus)


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    unigrams=buildVocabulary(corpus)
    count_unigrams={}
    for all in unigrams:
        count_unigrams[all]=0
    for word in unigrams:
        for eachlist in corpus:
            count_unigrams[word] += eachlist.count(word)
    return count_unigrams


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    start_word=[]
    for each_line in corpus:
        if each_line[0] not in start_word:
            start_word.append(each_line[0])
    return start_word

'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    start_word_count={}
    for each_line in corpus:
        word=each_line[0]
        if word not in start_word_count.keys():
            start_word_count[word]=1
        else:
            start_word_count[word]+=1
    return start_word_count
'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    bigram_words={}
    for each_line in corpus:
        for i in range(len(each_line)-1):
            first_word=each_line[i]
            sec_word=each_line[i+1]
            if (first_word not in bigram_words.keys()):
                sec_word_count={}
                sec_word_count[sec_word]=1
                bigram_words[first_word]= sec_word_count
            else:
                if sec_word not in bigram_words[first_word]:
                    bigram_words[first_word][sec_word] = 1
                else:
                    bigram_words[first_word][sec_word]+=1
    return bigram_words
### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    prob_unigrams=[]
    for each in unigrams:
        probability=(1/len(unigrams))
        prob_unigrams.append(probability)
    return prob_unigrams


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    unigram_probability=[]
    for word in unigrams:
        count_of_unigram=unigramCounts[word]
        probability=count_of_unigram/totalCount
        unigram_probability.append(probability)
    return unigram_probability


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    bigram_prob_dict={}
    for key,inner_dict in bigramCounts.items():
        words_list=[]
        probabilities=[]
        prevWord=key
        bigram_prob_dict[prevWord]={}
        for k,v in inner_dict.items():
            words_list.append(k)
            prob=v/unigramCounts[prevWord]
            probabilities.append(prob)
        bigram_prob_dict[prevWord]["words"]=words_list
        bigram_prob_dict[prevWord]["probs"]=probabilities
    return bigram_prob_dict


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    full_dict={}
    top_words={}
    temp_dict={}
    for i in range(len(words)):
        full_dict[words[i]]=probs[i]
    temp_dict=sorted(full_dict.items(),key= lambda x:x[1],reverse=True)
    for each in temp_dict:
        if len(top_words)<count:
            if each[0] not in ignoreList:
                top_words[each[0]]=each[1]
    # print(top_words)
    return top_words
'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices, random
def generateTextFromUnigrams(count, words, probs):
    list_sentence=[]
    while len(list_sentence)<count:
        random_word=choices(words,probs)
        list_sentence.append(random_word[0])
    sentence=list_sentence[0]
    for i in range(len(list_sentence[1:])):
        sentence=sentence+" "+list_sentence[i]
    return sentence


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    list_of_words=[]
    while len(list_of_words)<count:
        if len(list_of_words)==0 or list_of_words[-1] == '.':
            word=choices(startWords,startWordProbs)
            list_of_words.append(word[0])

        else:
            Lword=list_of_words[-1]
            word_list=bigramProbs[Lword]["words"]
            prob_list=bigramProbs[Lword]["probs"]
            word=choices(word_list,prob_list)
            list_of_words.append(word[0])
    sentence=list_of_words[0]

    for i in list_of_words[1:]:
        sentence=sentence+" "+i
    return sentence


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]           

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    top_50_words={}
    unigrams= buildVocabulary(corpus)
    unigrams_count=countUnigrams(corpus)
    len=getCorpusLength(corpus)
    unigrams_probs=buildUnigramProbs(unigrams,unigrams_count,len)
    top_50_words=getTopWords(50,unigrams,unigrams_probs,ignore)
    barPlot(top_50_words,"top 50 unigrams")
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    start_words=getStartWords(corpus)
    start_words_counts=countStartWords(corpus)
    len=getCorpusLength(start_words)
    start_words_probs=buildUnigramProbs(start_words,start_words_counts,len)
    top_start_words=getTopWords(50,start_words,start_words_probs,ignore)
    barPlot(top_start_words,"top 50 most frequent start words")
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    all_bigram_probs=buildBigramProbs(countUnigrams(corpus),countBigrams(corpus))
    word_list=all_bigram_probs[word]["words"]
    probs_list=all_bigram_probs[word]["probs"]
    topnextwordsdict=getTopWords(10,word_list,probs_list,ignore)
    barPlot(topnextwordsdict,"top next words of "+"-- "+word)
    return


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    ##aquiring first  corpus top words##
    corpus1_unigrams=buildVocabulary(corpus1)
    count_corpus1=countUnigrams(corpus1)
    len_corpus1=getCorpusLength(corpus1)
    unig_prob_corpus1=buildUnigramProbs(corpus1_unigrams,count_corpus1,len_corpus1)
    topword_corpus1=getTopWords(topWordCount,corpus1_unigrams,unig_prob_corpus1,ignore)
    ##aquiring second  corpus top words##
    corpus2_unigrams=buildVocabulary(corpus2)
    count_corpus2=countUnigrams(corpus2)
    len_corpus2=getCorpusLength(corpus2)
    unig_prob_corpus2=buildUnigramProbs(corpus2_unigrams,count_corpus2,len_corpus2)
    topword_corpus2=getTopWords(topWordCount,corpus2_unigrams,unig_prob_corpus2,ignore)
    ##combining top words and calculating probs list# #
    combined_top_words=list(topword_corpus1.keys())
    for k,v in topword_corpus2.items():
        word=str(k)
        if word not in combined_top_words:
            combined_top_words.append(word)
    corpus1_probs=[]
    corpus2_probs=[]
    final_result={}
    for each in combined_top_words:
        if each in corpus1_unigrams: 
            index_word=corpus1_unigrams.index(each)
            corpus1_probs.append(unig_prob_corpus1[index_word])
        else : 
            corpus1_probs.append(0)
        if each in corpus2_unigrams: 
            index_word=corpus2_unigrams.index(each)
            corpus2_probs.append(unig_prob_corpus2[index_word])        
        else : corpus2_probs.append(0)
    final_result["topWords"]=combined_top_words
    final_result["corpus1Probs"]=corpus1_probs
    final_result["corpus2Probs"]=corpus2_probs
    return final_result


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    graph_data=setupChartData(corpus1,corpus2,50)
    sideBySideBarPlots(graph_data["topWords"],graph_data["corpus1Probs"],graph_data["corpus2Probs"],name1,name2,title)
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    data_for_scatter=setupChartData(corpus1,corpus2,30)
    scatterPlot(data_for_scatter["corpus1Probs"],data_for_scatter["corpus2Probs"],data_for_scatter["topWords"],title)
    return


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()

#     ## Uncomment these for Week 2 ##
# """
    # print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    # test.week2Tests()
    # print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek2()
# # """

#     ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()

    # test.testCountBigrams()
    # test.testBuildUniformProbs()
    # test.testBuildUnigramProbs()
    # test.testGenerateTextFromBigrams()
    # test.testSetupChartData()