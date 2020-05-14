import easygui as eg
import numpy as np
import pickle
from copy import deepcopy
import operator
from nltk.stem import PorterStemmer
import webbrowser




def user_input(message, title):
    text = eg.enterbox(message, title)
    if text is None:
        exit()
    return text

def display_results_gui (webLinks, page_order, PreprocessedQuery,numberOfResults):
    
    # print(str(PreprocessedQuery),"hi")
    message = "Preprocessed query: "+str(PreprocessedQuery)+"\nThese are the results of your query, you can double click" \
                                                   " or select and press ok on a" \
                                                   " result to open the web page in a new tab of your default browser." \
                                                   " Press cancel to go back to the main menu."
    results = []
    for i in range(0, numberOfResults):
        results.append(webLinks[page_order[i][0]])

    return eg.choicebox(message, "SearchEngine results", results)

    


def show_links_gui(webLinks, page_order, query,numberOfResults):
    choice = display_results_gui(webLinks, page_order, query,numberOfResults)
    if choice is None:
        exit()
    else:
        OpenUrl(choice)

def OpenUrl(url):
    webbrowser.open_new(url)

if (__name__ == "__main__"):

    pagesCount = 4000
    query = user_input("Search Query: ", "SearchEngine")
    # print(query)


    # converting into lower case and splitting the words
    query = query.split(" ")
    temp_q = ""
    ps = PorterStemmer()
    for word in query:
        temp = word.lower()
        temp = ps.stem(temp)
        temp_q += (temp + " ")

    query = temp_q.rstrip(" ")

    with open("./uic_crawledLinks_"+str(pagesCount),"rb") as data_file:
        webLinks = pickle.load(data_file)

    with open("./index2_"+str(pagesCount),"rb") as outfile:
        stream_length_title = pickle.load(outfile)
        stream_length = pickle.load(outfile)
        IDF_title = pickle.load(outfile)
        IDF = pickle.load(outfile)
        inv_index_title = pickle.load(outfile)
        inv_index = pickle.load(outfile)
        TF_IDF_title = pickle.load(outfile)
        TF_IDF = pickle.load(outfile)
        no_slashes = pickle.load(outfile)
        len_URL = pickle.load(outfile)
        outlink_count = pickle.load(outfile)
        inlink_count = pickle.load(outfile)
        url_split = pickle.load(outfile)

    with open("./pagerank_"+str(pagesCount),"rb") as outfile:
        page_rank = pickle.load(outfile)

    IDF_queue = 0
    IDF_queue_title = 0
    for word in query:
        IDF_queue += IDF.get(word, 0)
        IDF_queue_title += IDF_title.get(word, 0)

    termFrequency_sum = {}
    termFrequency_min = {}
    termFrequency_max = {}
    covered_terms = {}

    for document, word_mapping in inv_index.items():
        covered_terms[document] = 0

    temp = query.split(" ")
    for word in temp:
        for document, word_mapping in inv_index.items():
            termFrequency_sum[document] = termFrequency_sum.get(document, 0) + word_mapping.get(word, 0)
            if (word_mapping.get(word, 0) != 0):
                covered_terms[document] = covered_terms.get(document) + 1

            if (termFrequency_min.get(document, 0) == 0):
                termFrequency_min[document] = word_mapping.get(word, 0)
            else:
                if (termFrequency_min[document] > word_mapping.get(word, float("Inf"))):
                    termFrequency_min[document] = word_mapping.get(word)

            if (termFrequency_max.get(document, 0) == 0):
                termFrequency_max[document] = word_mapping.get(word, 0)
            else:
                if (termFrequency_max[document] < word_mapping.get(word, 0)):
                    termFrequency_max[document] = word_mapping.get(word, 0)

    covered_term_ratio = {}
    for document, freq in covered_terms.items():
        covered_term_ratio[document] = freq / len(temp)

    termFrequency_title_sum = {}
    termFrequency_title_min = {}
    termFrequency_title_max = {}
    termFrequency_title_mean = {}
    covered_terms_title = {}

    temp = query.split(" ")
    for document, word_mapping in inv_index_title.items():
        covered_terms_title[document] = 0
    for word in temp:
        for document, word_mapping in inv_index_title.items():
            termFrequency_title_sum[document] = termFrequency_title_sum.get(document, 0) + word_mapping.get(word, 0)
            if (word_mapping.get(word, 0) != 0):
                covered_terms_title[document] = covered_terms_title.get(document) + 1

            if (termFrequency_title_min.get(document, 0) == 0):
                termFrequency_title_min[document] = word_mapping.get(word, 0)
            else:
                if (termFrequency_title_min[document] > word_mapping.get(word, float("Inf"))):
                    termFrequency_title_min[document] = word_mapping.get(word)

            if (termFrequency_title_max.get(document, 0) == 0):
                termFrequency_title_max[document] = word_mapping.get(word, 0)
            else:
                if (termFrequency_title_max[document] < word_mapping.get(word, 0)):
                    termFrequency_title_max[document] = word_mapping.get(word, 0)


    tf_idf_title_sum = {}
    tf_idf_title_min = {}
    tf_idf_title_max = {}
    tf_idf_title_mean = {}
    for word in temp:
        for document, word_mapping in TF_IDF_title.items():
            tf_idf_title_sum[document] = tf_idf_title_sum.get(document, 0) + word_mapping.get(word, 0)

            if (tf_idf_title_min.get(document, 0) == 0):
                tf_idf_title_min[document] = word_mapping.get(word, 0)
            else:
                if (tf_idf_title_min[document] > word_mapping.get(word, float("Inf"))):
                    tf_idf_title_min[document] = word_mapping.get(word)

            if (tf_idf_title_max.get(document, 0) == 0):
                tf_idf_title_max[document] = word_mapping.get(word, 0)
            else:
                if (tf_idf_title_max[document] < word_mapping.get(word, 0)):
                    tf_idf_title_max[document] = word_mapping.get(word, 0)

    tf_idf_title_mean = deepcopy(tf_idf_title_sum)
    for document, tf in tf_idf_title_mean.items():
        if (covered_terms_title[document] != 0):
            tf_idf_title_mean[document] /= covered_terms_title[document]
        else:
            tf_idf_title_mean[document] = 0

    rank_pages = {}
    for document in inv_index.keys():
        ip = []
        ip.append(40*covered_terms_title[document])
        ip = sum(ip)
        rank_pages[document] = ip

    page_rank_order = sorted(rank_pages.items(), key=lambda kv: kv[1], reverse=True)
    message = "TOP RESULTS FOUND \n" \
          "Enter the number of results which you need to display \n" \
          " Press cancel to close the popup"

    i=user_input(message,"Number of results needed")
    results = []
    for i in range(0, int(i)):
        print(webLinks[page_rank_order[i][0]])

    # print(results)
    show_links_gui(webLinks,page_rank_order,query,int(i))



