#!/usr/bin/python3

"""
Module for search utilities: SearchIndex, QueryProcessor and NLProcessor
Developed by Badr M. Abdullah (babdullah@lsv.uni-saarland.de)
"""

from collections import defaultdict
import torch
import numpy as np
import faiss
from flair.data import Sentence
import spacy


# CLASS: SearchIndex
class SearchIndex:
    """A class to build and represent the search index from FAISS object."""

    def __init__(
        self,
        search_index: faiss.IndexFlatIP,
        index2id: defaultdict
    ):
        """
        Instantiate object, set index and indicies data structure.
        """
        self.search_index = search_index
        self.index2id = index2id
        self.id2index = {_id:idx for idx, _id in index2id.items()}


    def retrieve_similar(
        self,
        query_emb: np.ndarray,
        num_to_retrieve: int=10,
        retrieve_ids: bool=True
    ):
        """Given a query embedding (1D np.array), return similar entries."""

        # re-shape query array
        query_emb = np.expand_dims(query_emb, axis=0)

        # use search index to retrieve similar entries
        _, sim_indicies = self.search_index.search(query_emb, num_to_retrieve)

        if retrieve_ids:
            return [self.index2id[entry] for entry in sim_indicies[0]]

        return sim_indicies[0]


# CLASS: QueryProcessor
class QueryProcessor:
    """A class for query processing functionality."""

    def __init__(
        self,
        nlp_processor,
        IDF_dict,
        word_embeddings,
        document_embeddings,
        word_index=None
    ):
        """
        Instantiate object, set NLP processor, and set word index.
        """
        self.nlp_processor = nlp_processor  # required for text processing
        self.word_index = word_index        # required for query expansion
        self.IDF_dict = IDF_dict            # required for term weightening

        self.doc_embeddings = document_embeddings
        self.tok_embeddings = word_embeddings

    def process_query(self, query_str: str):
        """Given query text, process and get word tokens."""
        # check for valid text string
        if not query_str:
            raise ValueError(f'Empty string is not a valid query.')

        tokens = self.nlp_processor.process_text(
            query_str,
            filter_stopwords=True
        )

        return tokens


    def get_embedding(
        self,
        query_str: str,
        expand_query: bool=False,
        apply_IDF: bool=False
    ):
        # text processing for the query
        query_tokens = self.process_query(query_str)

        # obtain a Flair Sentence object from the processed query
        query_sent = Sentence(' '.join(query_tokens))

        # apply IDF if query is longer than one word tokens
        if apply_IDF and len(query_sent) > 1:
            self.tok_embeddings.embed(query_sent)
            query_tok_embs = []

            # iterate over token in the query
            for tok in query_sent:
                # get str from Token object
                w = tok.text

                try:
                    # get word IDF from IDF_dict
                    word_IDF = self.IDF_dict[w] if w in self.IDF_dict else 1.0

                    # get word embedding and append to list
                    tok_emb = word_IDF*tok.get_embedding().detach().numpy()
                    query_tok_embs.append(tok_emb)

                except:
                    pass # if not possible to get word embedding, ignore

            if len(query_tok_embs) ==0:
                raise ValueError(f'No valid word tokens were found in query.')

            else:
                # convert list to numpy array and obtain average embedding
                query_emb = np.array(query_tok_embs).mean(axis=0)

        else:
            self.doc_embeddings.embed(query_sent)
            query_emb = query_sent.get_embedding().detach().numpy()

        # apply query expansion
        if expand_query:
            pass

        # L2normalize
        query_emb = query_emb/np.linalg.norm(query_emb)

        return query_emb


# CLASS: NLProcessor
class NLProcessor:
    """A class to encapsulate NLP pipeline functionality from Spacy."""

    def __init__(self, model):
        """
        Instantiate object and set NLP spacy model.
        Exmaple: nlp =  NLProcessor('de_core_news_sm')
        """
        self.nlp_model = spacy.load(model)

    def process_text(
        self,
        text_str: str,
        filter_stopwords: bool=False,
        filter_puncts: bool=True,
        filter_digits: bool=True
    ):

        # check for valid text string
        if not text_str:
            raise ValueError(f'Empty string is not valid for text processing.')

        tokens = []

        text_x = self.nlp_model(text_str)

        for tok in text_x:
            if filter_stopwords and tok.is_stop:  continue # remove stopwords
            if filter_puncts    and tok.is_punct: continue # remove punctuations
            if filter_digits    and tok.is_digit: continue # remove digits

            tokens.append(tok.text)

        if not tokens:
            raise ValueError(f'No valid tokens were found. Use content words.'
                             f'Avoid digits and punctuations in the text.')

        return tokens


# FUNCTION: compute_IDF
def compute_IDF(term: str, DF_dict: defaultdict, num_pages: int):
    try:
        n = num_pages - DF_dict[term] + 0.5
        d = DF_dict[term] + 0.5

    except:
        n = num_pages + 0.5
        d = num_pages/10

    return np.max([np.log10(n/d), 0.0])
