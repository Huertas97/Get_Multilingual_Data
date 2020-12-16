# Get_Multilingual_Data

# Index
 
 * [Repository purpose](#repository-purpose)
 * [Scripts](#scripts)
 * [How to use](#how-to-use)
 * [References](#references)
 
# Repository purpose

This repository collects the scripts required for downloading multilingal data. We use these scripts to download a specific number of sentences from three data sources: [TED2020](https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md), [WikiMatrix](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix) and [OPUS-NewsCommentart](http://opus.nlpl.eu/). The multilingual data can be used to fit a multilingual PCA from the multilingual embeddings computed by a multilingual model or any other purpose. 


# Scripts

The repository is composed of three scripts:

* `get_TED2020_sentences.py`: This script downloads the TED2020 corpus and create parallel sentences tsv files. The TED2020 corpus is a crawl of transcripts from TED and TEDx talks, which are translated to 100+ languages. With this script the user can select the amount of sentences and the languages desired.  The TED2020 corpus is downloaded automatically only for the languages selected.

* `get_news_opus.py`: OPUS-NewsCommentary is one of the different dataset available in OPUS. It consists of a parallel corpus of News Commentaries provided by Workshop on Statistical Machine Translation (WMT). This type of data is the most related to fact-check news we hope to face up. 

* `get_wikimatrix_sentences.py`: This script automatically downloads WikiMatrix corpus for the languages selected. The WikiMatrix corpus is a crawl of mined sentences from Wikipedia in  different languages. With this script the user can select the amount of sentences and the languages desired. We only used pairs with scores above 1.075, as pairs below this threshold were often of bad quality.

All of these scripts are adapted from [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library. 


# Type of Multilingual Data

The multilingual data that can be extracted with the scripts from these repository could be parallel data (same sentences in different languages) or non-parallel data (different sentences per language). The scripts `get_TED2020_sentences.py` and `get_news_opus.py` extract parallel sentences. Meanwhile, `get_wikimatrix_sentences.py` extracts non-parallel data. 

The same sentence in different languages should theorically have the same vectorization. However, there might be some variability among languages. Including this variability in the data to fit the PCA is highly recommended. Parallel data from TED2020 and OPUS-NewsCommentart are used for this purpose. As we just mentioned, parallel data is extremely useful for including the language embedding representation variability in the PCA. However, introducing a specific set of sentences for each language is also required. WikiMatrix sentences ensures that PCA includes the specific representation for each language. 


# Languages

Each data set used has a range of available languages. The most limited data set is OPUS-NewsCommentary with 15 languages: ar, cs, de, en, es, fr, hi, it, ja, nl, pl, pt, ru, tr, zh. All the official languages acronyms can be consulted [here](https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry). 
TED2020 and WikiMatrix support more than 100 languages. 

# How to use

A detailed example of the usage of these scripts is shown in the `notebooks` folder. The example is a Google Colab notebook. It runs the code on Google servers, and you don’t need to install anything to run any code. Moreover, anyone with a Google account can just copy the notebook on his own Google Drive account.

For more information, the help documentation for each script is accesible using the following command:

```
$ python <script> --help
```

For example:
```
$ python get_TED2020_sentences.py --help
```

```
This script downloads the TED2020 corpus and create parallel sentences tsv files

The TED2020 corpus is a crawl of transcripts from TED and TEDx talks, which 
are translated to 100+ languages. With this script the user can select the 
amount of sentences and the languages desired. 

The TED2020 corpus is downloaded automatically only for the languages selected.
          
Usage:

    python get_TED2020_sentences [options] 

Options:
    -n, --n_sentences            Number of sentences to collect
    -l, --languages              Languages ​​from which we extract sentences
    -h, --help                   Help documentation


Example. Extract TED 2020  arabic and italian sentences:
    python get_TED2020_sentences.py --n_sentences 500 --languages ar,it
```
# References

Holger Schwenk et al. “Wikimatrix: mining 135m parallel sentences in 1620 language pairsfrom wikipedia”. In:arXiv:1907.05791 [cs](July 2019). arXiv: 1907.05791.url:http://arxiv.org/abs/1907.0579

Jörg Tiedemann. “Parallel Data, Tools and Interfaces in OPUS”. In:Proceedings of theEighth International Conference on Language Resources and Evaluation (LREC’12). Is-tanbul, Turkey: European Language Resources Association (ELRA), May 2012, pp. 2214–2218.url:http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.
