import re

import jsonlines
import pandas as pd
import pyterrier as pt
import unidecode

if not pt.started():
    pt.init()


def _gesis_doc_iter(path):
    with jsonlines.open(path) as reader:
        for obj in reader:
            title = obj.get("title") or ""
            title = title[0] if type(title) is list else title
            abstract = obj.get("abstract") or ""
            abstract = abstract[0] if type(abstract) is list else abstract
            yield {"docno": obj.get("id"), "text": " ".join([title, abstract])}


class Ranker(object):

    def __init__(self):
        self.idx = None

    def index(self):
        pass

    def rank_publications(self, query, page, rpp):

        itemlist = []

        return {
            "page": page,
            "rpp": rpp,
            "query": query,
            "itemlist": itemlist,
            "num_found": len(itemlist),
        }


class Recommender(object):

    def __init__(self):
        self.idx_publications = None
        self.idx_datasets = None
        self.title_lookup = {}

    def index(self):
        iter_indexer = pt.IterDictIndexer("./index/publications", meta={"docno": 100})
        doc_iter = _gesis_doc_iter("./data/gesis-search/documents/publication.jsonl")
        indexref = iter_indexer.index(doc_iter)
        self.idx_publications = pt.IndexFactory.of(indexref)

        with jsonlines.open(
            "./data/gesis-search/documents/publication.jsonl"
        ) as reader:
            for obj in reader:
                self.title_lookup[obj.get("id")] = obj.get("title")

    def recommend(self, item_id, page, rpp):
        itemlist = []

        doc_title = self.title_lookup.get(item_id)
        doc_title = re.sub(r"[^\w\s]", " ", doc_title)
        doc_title = unidecode.unidecode(doc_title)

        if doc_title is not None:
            topics = pd.DataFrame.from_dict({"qid": [0], "query": [doc_title]})
            retr = pt.BatchRetrieve(
                self.idx_publications, controls={"wmodel": "TF_IDF"}
            )
            retr.setControl("wmodel", "TF_IDF")
            retr.setControls({"wmodel": "TF_IDF"})
            res = retr.transform(topics)
            itemlist = list(res["docno"][page * rpp : (page + 1) * rpp])

        return {
            "page": page,
            "rpp": rpp,
            "item_id": item_id,
            "itemlist": itemlist,
            "num_found": len(itemlist),
        }
