# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 20:27:43 2020

Adapted from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/multilingual/get_parallel_data_opus.py

OPUS (http://opus.nlpl.eu/) is a great collection of different parallel datasets
for more than 400 languages. On the website, you can download parallel datasets 
for many languages in different formats. 

NewsCommentary is one of the different dataset available in OPUS. It consists of
a parallel corpus of News Commentaries provided by Workshop on Statistical Machine 
Translation (WMT). This type of data is the most related to fact-check news we 
hope to face up. 

Requirements: 
    This scripts requiers opustools. You can install it with the following command:
        $ pip install opustools


@author: Álvaro Huertas García
"""

from opustools import OpusRead
import os
import random
import gzip
import sys
from optparse import OptionParser
import re
import pandas as pd

# Process command-line options
parser = OptionParser(add_help_option=False)

# General options
parser.add_option('-n', '--n_sentences', type='int', help='Number of sentences to retrieve')
parser.add_option('-l', '--languages', help='Languages ​​from which we extract sentences')
parser.add_option('-h', '--help', action='store_true', help='Show this help message and exit.')

(options, args) = parser.parse_args()

def print_usage():
    print("""       
OPUS-NewsCommentary is one of the different dataset available in OPUS. It consists of
a parallel corpus of News Commentaries provided by Workshop on Statistical Machine 
Translation (WMT). This type of data is the most related to fact-check news we 
hope to face up. 


Requirements: 
    This scripts requiers opustools. You can install it with the following command:
        $ pip install opustools
        
Usage:

    python get_news_opus.py [options] 

Options:
    -n, --n_sentences            Number of sentences to collect
    -l, --languages              Languages ​​from which we extract sentences
    -h, --help                   Help documentation

Example. Extract OPUS-NewsCommentary arabic and italian sentences:
    !python get_news_opus.py --n_sentences 500 --languages ar,it""")
    sys.exit()

if not options.n_sentences or not options.languages or options.help:
    print_usage()


languages = options.languages.split(",")
print("Recovering OPUS-NewsCommentary sentences for languages: {}".format(" ".join(languages)))
corpora = ['News-Commentary']
source_languages = ['en']      # Source language
target_languages = languages   # New languages for parallel data

n_sentences = options.n_sentences
output_folder = 'parallel-sentences/News'
opus_download_folder = './opus'

#Iterator over all corpora / source languages / target languages combinations and download files
os.makedirs(output_folder, exist_ok=True)

for corpus in corpora:
    for src_lang in source_languages:
        for trg_lang in target_languages:
            output_filename = os.path.join(output_folder, "{}-{}-{}.tsv.gz".format(corpus, src_lang, trg_lang))
            if not os.path.exists(output_filename):
                print("Create:", output_filename)
                try:
                    read = OpusRead(directory=corpus, source=src_lang, target=trg_lang, write=[output_filename], download_dir=opus_download_folder, preprocess='raw', write_mode='moses', suppress_prompts=True)
                    read.printPairs()
                except:
                    print("An error occured during the creation of", output_filename)
                    os.remove(os.path.join(output_filename))

print("Creating data frame for News Commentarys OPUS languages: {}".format("-".join(languages)))
random.seed(0) # For repoducibility porpuse
sentences = []
langs = []

# Create a data frame from each folder (source of data)
if "en" in languages:
  english = True
else:
  english = False

for f in os.listdir(output_folder):
  if not f.startswith(".") and not f.startswith("df"):
    corpus = []
    sent1 = []
    sent2 = []
    origin, lang1, lang2 = re.findall("([A-Za-z|0-9|-]+)-([A-Za-z]{2})-([A-Za-z]{2})", f)[0]
    print(f)
    with gzip.open(os.path.join(output_folder, f), 'rt', encoding='utf8') as fIn:
        for line in fIn:
          if not line.startswith("."):
            corpus.append(line.strip())
        
        count = 0
        for sample in corpus:
          pair = sample.split("\t")
          if count < n_sentences: # stop when achieve desired n sentences
            if len(pair) == 2:
              sent1.append(pair[0])
              sent2.append(pair[1])
              count += 1
          else: 
            break
        if english: # if english 
          print("Creating english sentences parallel to", lang2)
          sentences += sent1
          langs += [lang1] * len(sent1)
          english = False
        sentences += sent2
        langs += [lang2] * len(sent2)
        
save_PATH = os.path.join(output_folder, "df_News_{}.pkl".format("-".join(languages)))
print("---Saving results in {} ---".format(save_PATH))       
df_OPUS = pd.DataFrame({"sentences": sentences, "lang": langs})
df_OPUS["from"] = "OPUS_News_Commentary"
df_OPUS = df_OPUS.reindex(sorted(df_OPUS.columns), axis=1) # Order columns
df_OPUS.to_pickle(save_PATH)

# Removing downloaded files
print("--- Removing downloaded files ---")
for f in os.listdir(output_folder):
  if not f.startswith(".") and not f.startswith("df"):
    os.remove(os.path.join(output_folder, f))
print("--- Finish ---")
