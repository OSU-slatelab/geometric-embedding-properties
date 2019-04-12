# geometric-embedding-properties

Source code and detailed results for

- Whitaker et al, "[Characterizing the impact of geometric properties of word embeddings on task performance](https://arxiv.org/abs/1904.04866)." In Proceedings of RepEval 2019.

This code is released under MIT License. If you use it in your own work, please cite the following paper:
```
@inproceedings{Whitaker2019RepEval,
  author = {Whitaker, Brendan and Newman-Griffis, Denis and Haldar, Aparajita and Ferhatosmanoglu, Hakan and Fosler-Lussier, Eric},
  title = {Characterizing the impact of geometric properties of word embeddings on task performance},
  booktitle = {Proceedings of the Third Workshop on Evaluating Vector Space Representations for NLP (RepEval)},
  year = {2019}
}
```

## Implementations

This repository includes implementations of the embedding transformation methods described in the above paper.  They are broken down into three modules:

- `affine` - Implementations of affine transformations.  For more details, see [specific README](affine/README.md).
- `CDE` - Implementation of cosine distance encoding (CDE) transformation.  For more details, see [specific README](CDE/README.md).
- `NNE` - Implementation of nearest neighbor encoding (NNE) transformations.  For more details, see [specific README](NNE/README.md).

### Evaluation tasks

For evaluation tasks, we relied on two other repositories:

- [kudkudak/word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks) for intrinsic evaluations
- [drgriffis/Extrinsic-Evaluation-tasks](https://github.com/drgriffis/Extrinsic-Evaluation-Tasks) for extrinsic evaluations

## Data

Our full tables of results are included in the `detailed-results` directory.  This includes separate files for intrinsic and extrinsic tasks for each set of word embeddings used.

The reference word embeddings we used are linked below:

- Word2Vec - [300-d GoogleNews embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- GloVe - [300-d embeddings from 840B Common Crawl](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- FastText - [300-d Wikipedia/UMBC/StatMT embeddings with subword information](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip)
