"""Generation of suggested spell corrections"""
import pkg_resources
from symspellpy import SymSpell, Verbosity
import pandas as pd


sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, 0, 1, encoding='utf-8')

df = pd.read_pickle('prepr_voc_singles_no_latex_ver_5_stops.pk')

abstract_list = df['abstract']
title_list = df['title']


tf_vocabulary = {}


for index, row in df.iterrows():
    text = []
    if isinstance(row['title'], str) and isinstance(row['abstract'], list):
        text += row['abstract']
    elif isinstance(row['title'], list) and isinstance(row['abstract'], str):
        text += row['title']
    elif isinstance(row['title'], list) and isinstance(row['abstract'], list):
        text = row['title'] + row['abstract']
    # text_set = set(text)
    for token in text:
        tf_vocabulary[token] = tf_vocabulary.get(token, 0) + 1


with open('spell_corrections.txt', 'wb') as f:
    for token, count in tf_vocabulary.items():
        suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
        for suggestion in suggestions:
            if suggestion != token:
                tuple_1 = (token, suggestion.term)
                entry = str(count) + ' ' + str(tuple_1) + ',\n'
                f.write(entry.encode('utf8'))
                