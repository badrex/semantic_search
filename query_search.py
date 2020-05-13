#!/usr/bin/python3

"""
Main module for the semantic search engine
Developed by Badr M. Abdullah (babdullah@lsv.uni-saarland.de)
"""

from collections import defaultdict
import json
import pickle
import torch
import numpy as np

# Flair for word and documents embeddings
import flair
from flair.embeddings import BytePairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence

# FAISS for the search index
import faiss

# from search_utils
from search_utils import NLProcessor, QueryProcessor, SearchIndex, compute_IDF


def main():
    # A set of global variables for the data structure
    # page to text dict to show text snippets from pages
    print(f"Read page2text dictionary from desk ...")
    page2text = np.load(
        'data_structures/page2text.npy',
        allow_pickle='TRUE'
    ).item()

    NUM_PAGES = len(page2text)

    # document frequency statistics
    print(f"Read document frequency dictionary from desk ...")
    doc_frequency = np.load(
        'data_structures/doc_frequency.npy',
        allow_pickle='TRUE'
    ).item()

    # make dict for IDF
    print(f"Compute IDF for each term in the collection ...")
    word2IDF = {
        w:compute_IDF(w, doc_frequency, NUM_PAGES) for w in doc_frequency
    }

    # index to page ID, retrieves page ID as in the database
    print(f"Read page2text dictionary from desk ...")
    index2pageID = np.load(
        'data_structures/idx2pageID_TFIDF.npy',
        allow_pickle='TRUE'
    ).item()

    # initialize word and document embeddings from Flair
    print(f"Initialize Fliar embeddings ...")
    BPE_embeddings = BytePairEmbeddings('de', dim=300, syllables=200000)

    document_embeddings = DocumentPoolEmbeddings(
        [
            BPE_embeddings
        ]
    )

    # initialize query processor object
    print(f"Initialize query processor object with Spacy and Flair ...")
    nlp_processor = NLProcessor('de_core_news_sm')

    query_processor = QueryProcessor(
        nlp_processor,
        word2IDF,
        BPE_embeddings,
        document_embeddings
    )

    # load search index from desk, initialize SearchIndex object
    print(f"Read FAISS search index from desk ...")
    faiss_index  = faiss.read_index('data_structures/page_index_BPE_TFIDF.index')
    search_index = SearchIndex(faiss_index, index2pageID)


    # query processing and page retrieval starts here

    # NOTE: here the server code should be added to listen to a port,
    # get a user query, and respond to a query
    user_query = 'START'

    while True:
        user_query = input("Please enter query here: ")

        if user_query == 'exit': break

        print("You entered: " + user_query)

        # embed query
        query_emb = query_processor.get_embedding(user_query, apply_IDF=True)

        # retrieve similar entries with search index
        sim_entries = search_index.retrieve_similar(
            query_emb,
            num_to_retrieve=10 # if more than 10 pages, change this number
        )

        # store search results in dict
        search_results = defaultdict(lambda: defaultdict(str))

        for rank, entry in enumerate(sim_entries):
            book, page_num = '_'.join(s for s in entry.split('_')[:2]).split('_')
            text_snippet = ' '.join(page2text[book + '_' + page_num])[:200]

            search_results[rank + 1]['book'] = book
            search_results[rank + 1]['page_num'] = page_num
            search_results[rank + 1]['snippet'] = text_snippet

            # print(f"Query: {user_query:<20}\t"
            #       f"Rank: {rank+1:>3}\t"
            #       f"Book: {book:>7}\t"
            #       f"Page num: {page_num:>3}")
            #
            # print(f"Page text: {text_snippet}")
            # print('---------')

        # JSONify output results
        json_string = json.dumps(
            search_results,
            indent=4,
            sort_keys=True,
            ensure_ascii=False
        ).encode('utf8')

        print(json_string.decode())


if __name__ == '__main__':
    main()
