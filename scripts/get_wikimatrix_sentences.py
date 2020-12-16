# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:39:18 2020

Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/multilingual/get_parallel_data_wikimatrix.py

This script downloads the WikiMatrix corpus (https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix)

The WikiMatrix corpus is a crawl of mined sentences from Wikipedia in 
different languages.
With this script the user can select the amount of sentences and the languages 
desired. 

The WikiMatrix corpus is downloaded automatically only for the languages selected. 

@author: Álvaro Huertas García
"""

# Adapted from get_parallel_data_wikimatrix.py 
import os
import sentence_transformers.util
import gzip
import csv
from tqdm.autonotebook import tqdm
from optparse import OptionParser
import re
import pandas as pd
import sys

# Process command-line options
parser = OptionParser(add_help_option=False)

# General options
parser.add_option('-n', '--n_sentences', type='int', help='Number of sentences to retrieve')
parser.add_option('-l', '--languages', help='Languages ​​from which we extract sentences')
parser.add_option('-h', '--help', action='store_true', help='Show this help message and exit.')
(options, args) = parser.parse_args()

def print_usage():
    print("""
This script automatically downloads WikiMatrix corpus for the languages selected.   
The WikiMatrix corpus is a crawl of mined sentences from Wikipedia in 
different languages. With this script the user can select the amount of 
sentences and the languages desired. We only used pairs with scores
above 1.075, as pairs below this threshold were often of bad quality.
       
          
Usage:

    python get_wikimatrix_sentences.py [options] 

Options:
    -n, --n_sentences            Number of sentences to collect
    -l, --languages              Languages ​​from which we extract sentences
    -h, --help                   Help documentation


Example. Extract Wikimatrix arabic and italian sentences:
    python get_wikimatrix_sentences.py --n_sentences 500 --languages ar,it""")
    sys.exit()

if not options.n_sentences or not options.languages or options.help:
    print_usage()


languages = options.languages.split(",")
print("Recovering WikiMatrix data for for languages: {}".format(" ".join(languages)))
source_languages = set(['en'])     # Source language 
target_languages = set(languages)  # New languages for parallel data


# Number of sentences we want for parallel data
num_pca_sentences = options.n_sentences
# Only use sentences with a LASER similarity score above the threshold
threshold = 1.075  

download_url = "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/"
download_folder = "../datasets/WikiMatrix/"
parallel_sentences_folder = "parallel-sentences/"
Wiki_sentences_folder = "parallel-sentences/Wikimatrix/"


os.makedirs(os.path.dirname(download_folder), exist_ok=True)
os.makedirs(parallel_sentences_folder, exist_ok=True)
os.makedirs(Wiki_sentences_folder, exist_ok=True)

# Donwload WikiMatrix data for languages selected
for source_lang in source_languages:
    for target_lang in target_languages:
        filename_train_pca = os.path.join(Wiki_sentences_folder, "WikiMatrix-{}-{}-train_pca.tsv.gz".format(source_lang, target_lang))

        if not os.path.exists(filename_train_pca):
            langs_ordered = sorted([source_lang, target_lang])
            wikimatrix_filename = "WikiMatrix.{}-{}.tsv.gz".format(*langs_ordered)
            wikimatrix_filepath = os.path.join(download_folder, wikimatrix_filename)

            if not os.path.exists(wikimatrix_filepath):
                print("Download", download_url+wikimatrix_filename)
                try:
                    sentence_transformers.util.http_get(download_url+wikimatrix_filename, 
                                                        wikimatrix_filepath)
                except:
                    print("Was not able to download", download_url+wikimatrix_filename)
                    continue

            if not os.path.exists(wikimatrix_filepath):
                continue
            
            # Create parallel sentences files
            pca_sentences = []
            pca_sentences_set = set()
            extract_pca_sentences = True

            with gzip.open(wikimatrix_filepath, 'rt', encoding='utf8') as fIn:
                for line in fIn:
                    score, sent1, sent2 = line.strip().split('\t')
                    sent1 = sent1.strip()
                    sent2 = sent2.strip()
                    score = float(score)

                    if score < threshold:
                        break

                    if sent1 == sent2:
                        continue

                    if langs_ordered.index(source_lang) == 1: 
                        sent1, sent2 = sent2, sent1 #Swap, so that src lang is sent1

                    # Avoid duplicates
                    if sent1 in pca_sentences_set or sent2 in pca_sentences_set:
                        continue

                    if extract_pca_sentences:
                        pca_sentences.append([sent1, sent2])
                        pca_sentences_set.add(sent1)
                        pca_sentences_set.add(sent2)

                        if len(pca_sentences) >= num_pca_sentences:
                            extract_pca_sentences = False
                    else:
                        break
            
            # Save to file the selected parallel sentences
            print("Write", len(pca_sentences), "PCA train sentences", filename_train_pca)
            with gzip.open (filename_train_pca, 'wt', encoding='utf8') as fOut:
                for sents in pca_sentences:
                    fOut.write("\t".join(sents))
                    fOut.write("\n")




print("Creating data frame for Wikimatrix languages: {}".format("-".join(languages)))
files = os.listdir(os.path.join(Wiki_sentences_folder))
df_files = []

# Create a data frame from each folder (source of data)
if "en" in languages:
  english = True
else:
  english = False

for f in files:
  if not f.startswith(".") and not f.startswith("df"):
      print(f)
      data_origin, lang1, lang2 = re.findall("([A-Za-z|0-9]+)-([A-Za-z]{2})-([A-Za-z]{2})", f)[0]
      df = pd.read_csv(os.path.join(Wiki_sentences_folder, f), header=None, 
                       names=[lang1, "sentences"], sep='\t', 
                       quoting=csv.QUOTE_NONE)
      df["lang"] = lang2
      df_files.append(df.iloc[:, [1,2]]) 
      if english:
        df = pd.read_csv(os.path.join(Wiki_sentences_folder, f), header=None, 
                         names=["sentences", lang2], sep='\t', 
                         quoting=csv.QUOTE_NONE)
        df["lang"] = lang1
        df_files.append(df.iloc[:, [0,2]])
        english = False

save_PATH = os.path.join(Wiki_sentences_folder, "df_wikimatrix_{}.pkl".format("-".join(languages)))
print("---Saving results in {} ---".format(save_PATH))
# Concatenate all the languages from a data source by columns
df_wikimatrix = pd.concat(df_files, axis = 0) 
df_wikimatrix["from"] = data_origin # Set the data origin the first column
df_wikimatrix = df_wikimatrix.reindex(sorted(df_wikimatrix.columns), axis=1) # Order columns
df_wikimatrix.reset_index(drop=True, inplace = True)
df_wikimatrix.to_pickle(save_PATH)

# Removing downloaded files
print("--- Removing downloaded files ---")
for f in os.listdir(Wiki_sentences_folder):
  if not f.startswith(".") and not f.startswith("df"):
    os.remove(os.path.join(Wiki_sentences_folder, f))
print("--- Finish ---")