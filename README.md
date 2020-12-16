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

* `get_TED2020_sentences.py`

* `get_news_opus.py`

* `get_wikimatrix_sentences.py`

All of these scripts are adapted from [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library. 


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
