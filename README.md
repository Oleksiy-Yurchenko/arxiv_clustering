Topic extraction is performed in 2 stages.

1st stage - execution of file preprocessing.py - file performs preprocessing of the arxiv-metadata-oai-snapshot.json. 
The result of preprocessing, a dataframe with preprocessed titles abstracts and categories will be stored in the file "prepr_voc_singles_no_latex_ver_4_stops.pk"

2nd stage - execution of fiel svd_kmeans.py - performs feature extraction with TF-IDF, dimensinality reduction with LSA-SVD, clustering with K-Means and vizualization with t-SNE. 
File takes as input a dataframe with preprocessed titles, abstracts, categories ("prepr_voc_singles_no_latex_ver_4_stops.pk") and produces vizualization plot, log file with topics and clustering results.

dataset_analyze.py - file generates diagrams and statistics, not required for 

requirements.txt - contains the list of all required modules

spell_corrections.py - file creates a text file with suggested spell corrections. 

sp_corr.py - contains a list with manually selected spell corrections generated by spell_corrections.py

svd_500f_2min_df_5000lmax_df_1000K_trial_2.log - contains a log with clustering scores and all generated topics.

sciences.py - contains sciences with categories.

vocab.py - contains foreign texts.

common_words.py - contains common words, abbreviations, withdrawn items, greek letters and combinations.

dataset_exceptions.py - contains id of the articles with incorrectly formatted LaTex, foreign articles, articles with incorrect line split.

mapping.py - contains plurals to singulars mapping.
