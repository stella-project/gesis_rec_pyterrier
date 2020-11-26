### Micro template implementing recommendations with the help of Pyterrier

This repository contains an adapted version of the [STELLA Micro-Template](https://github.com/stella-project/stella-micro-template). It is implemented with the help of [pyterrier](https://github.com/terrier-org/pyterrier) - the Python wrapper to [Terrier](https://github.com/terrier-org/terrier-core). The dataset recommendations are based on the abstracts of the datasets and queries made from the titles of the target items. The index contains the abstracts of the datasets. When providing the publication identifier (target item of the recommendation), it will be translated into the publication title, which, in turn, is used to query the index with a tfidf-based algorithm.