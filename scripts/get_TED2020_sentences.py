# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:51:30 2020
Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/multilingual/get_parallel_data_ted2020.py

This script downloads the TED2020 corpus (https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md)
 and create parallel sentences tsv files

The TED2020 corpus is a crawl of transcripts from TED and TEDx talks, which 
are translated to 100+ languages.
With this script the user can select the amount of sentences and the languages 
desired. 

The TED2020 corpus is downloaded automatically only for the languages selected. 

@author: Álvaro Huertas García
"""

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
    python get_TED2020_sentences.py --n_sentences 500 --languages ar,it""")
    sys.exit()

if not options.n_sentences or not options.languages or options.help:
    print_usage()


languages = options.languages.split(",") 
print("Recovering TED2020 talks for languages: {}".format(" ".join(languages)))

source_languages = set(['en'])       # Source language                           
target_languages = set(languages)    # New languages for parallel data

# Number of sentences we want for parallel data
train_pca_sentences = options.n_sentences   
download_url = "https://sbert.net/datasets/ted2020.tsv.gz"
ted2020_path = "../datasets/ted2020.tsv.gz" # Path of the TED2020.tsv.gz file.
parallel_sentences_folder = "parallel-sentences/"
TED_sentences_folder = "parallel-sentences/TED2020/"



# Download data
os.makedirs(os.path.dirname(ted2020_path), exist_ok=True)
if not os.path.exists(ted2020_path):
    print("ted2020.tsv.gz does not exists. Try to download from server")
    sentence_transformers.util.http_get(download_url, ted2020_path)


# Create data folders
os.makedirs(parallel_sentences_folder, exist_ok=True)
os.makedirs(TED_sentences_folder, exist_ok=True)
train_pca_files = []
files_to_create = []
for source_lang in source_languages:
    for target_lang in target_languages:
        output_filename_train_pca = os.path.join(TED_sentences_folder, "TED2020-{}-{}-train_pca.tsv.gz".format(source_lang, target_lang))
        train_pca_files.append(output_filename_train_pca)
        if not os.path.exists(output_filename_train_pca):
            files_to_create.append({'src_lang': source_lang, 'trg_lang': target_lang,
                                    'fTrain': gzip.open(output_filename_train_pca, 'wt', encoding='utf8'),
                                    'Count': 0, 
                                    "Completed":0
                                    })

# Read by line the TED2020 data and extract the nº of sentences required
target_text_set = set()
if len(files_to_create) > 0:
    print("Parallel sentences files {} do not exist. Create these files now".format(", ".join(map(lambda x: x['src_lang']+"-"+x['trg_lang'], files_to_create))))
    with gzip.open(ted2020_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
          if sum([d["Completed"] for d in files_to_create]) == len(languages):
                break
          else:
            for outfile in files_to_create:
                src_text = line[outfile['src_lang']].strip()
                trg_text = line[outfile['trg_lang']].strip()
                if (src_text != "" and trg_text != "") and (trg_text not in target_text_set):
                    if outfile['Count'] < train_pca_sentences : # user option
                        outfile['Count'] += 1
                        fOut = outfile['fTrain']
                        fOut.write("{}\t{}\n".format(src_text, trg_text))

                        # To avoid duplicate sentences
                        target_text_set.add(trg_text)
                    else:
                        outfile['Completed'] = 1
                        pass

                    

    for outfile in files_to_create:
        outfile['fTrain'].close()

print("Creating data frame for TED2020 languages: {}".format("-".join(languages)))
files = os.listdir(os.path.join(TED_sentences_folder))
df_files = []

# Create a data frame from each folder (source of data)
for f in files:
  if not f.startswith(".") and not f.startswith("df"):
      print(f)
      data_origin, lang1, lang2 = re.findall("([A-Za-z|0-9]+)-([A-Za-z]{2})-([A-Za-z]{2})", f)[0]
      df = pd.read_csv(os.path.join(TED_sentences_folder, f), header=None, names=[lang1, "sentences"], sep='\t', quoting=csv.QUOTE_NONE)
      df["lang"] = lang2
      df_files.append(df.iloc[:, [1,2]]) 

save_PATH = os.path.join(TED_sentences_folder, "df_TED_{}.pkl".format("-".join(languages)))
print("---Saving results in {} ---".format(save_PATH))
# Concatenate all the languages from a data source by columns
df_TED = pd.concat(df_files, axis = 0) 
# Set the data origin as the first column
df_TED["from"] = data_origin 
 # Order columns
df_TED = df_TED.reindex(sorted(df_TED.columns), axis=1)
df_TED.reset_index(drop=True, inplace = True)
# Save results
df_TED.to_pickle(save_PATH)

# Removing downloaded files
print("--- Removing downloaded files ---")
for f in os.listdir(TED_sentences_folder):
  if not f.startswith(".") and not f.startswith("df"):
    os.remove(os.path.join(TED_sentences_folder, f))
print("--- Finish ---")