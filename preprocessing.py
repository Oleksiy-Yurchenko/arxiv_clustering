import warnings
import unicodedata
import json
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
import logging
from nltk.corpus import stopwords
import langdetect
from string import punctuation
from vocab import f_text
from mapping import plurals_singles_map
from sp_corr import spell_corrections
from common_words import common, withdrawn, abbr_list, greek
from sciences import CS, MATH, Q_BIO, ECON, Q_FIN, PHYSICS, EESS, STAT
from dataset_exceptions import bad_latex, foreign, bad_split


### LOGGING MODULE ###
warnings.filterwarnings('error', category=Warning, append=True)
LEVEL = logging.DEBUG
logger = logging.getLogger('Arxiv_clust')
logger.setLevel(LEVEL)
fh = logging.FileHandler('text_processing.log')
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


### DECLARATION OF CONSTANTS ###
TAGS_SET = {'NN', 'NNS', 'NNP', 'FW', 'JJ', 'JJR', 'JJS', 'VBP', 'VBG'}
SYMBOL_PATTERN = punctuation + """⇌^▹▿⊓≻*°⊞⊖⊛⇔«»ב⊑’⊎◃↷∣♮←∣↷ı†⊔∇∇∩∇△א≥≤”“≠→∈ϵ∉♢√±∼\⊙·≈∫∝–≃⇔⊆⊇↦≲⪅≡‖∥⊥≪≫⊂×⟩⟨⊗⊕⟶≅≳≼≽↔∧∨∋∓⌈⌉⋀⋁⋂⋃——⪅↪⊊▵∘⋊𝖠∀⩾⩽÷𝕀↓↑⌊⌋∖≅…⋯∗⋆⌉⌈∏∞□∃≊⇝∅ø⫋∽↘▹≧⊃⊃∇⋉⇌ħ≺≺∪∇±υ∩∙♯⋉≍⇒≀∪⊠∃"""
ABBRV_PATTERN = re.compile(r'\sad\.|\sc\.|\scf\.|cit\.|\def\.|e\.g\.|\seg\s|\se\.g\s|\sed\s|\set\s|\bet\sal\.|\set\sal\s|\seds\s|\seds\.|\sel\s|\setc\.|\setc\s|\sid\.|\sid\s|i\.e\.|\sie\s|\sibid\.|\sibid\s|\sinf\.|\sinf\s|\sillus\.|\sillus\s|\sloc\s|\sloc\.|\sms\.|\sms\s|\smss\.|\smss |\sn/a\s|\sna\s|\sn\.b\.|\sn\.b\s|\snb\s|\snd\s|\sno\.|\sno\s|\sop\.|\sop\s|\sp\.|\sp\s|\spg\.|\spg\s|\spp\.|\spp\s|\spgs\.|\spgs\s|pseud\.|\spseud\s|\spub\s|\spub\.|\sqtd\s|q\.v\.|\sqv\b|\sq.v\s|resp\.|\sresp\s|\ssic\s|\ssc\s|\ssc\.|s\.v\.|\ss\.v\s|\ssup\s|\ssup\.|\strans\s|\sup\s|\sv\s|\sv\.|\sviz\s|\svol\.|\svol\b|\svols\.|\svols\b|\svs\.|\svs\s')
EMAIL_PATTERN = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
WWW_PATTERN = re.compile(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:".,<>?«»“”‘’]))')
LEN_PATTERN = len(SYMBOL_PATTERN)
EXCEPTIONS = bad_latex + foreign + bad_split


### DECLARATION OF FUNCTIONS###
def nltk_pos_tagger(nltk_tag):
    """Function converts NLTK tags to a shorter form"""
    if nltk_tag.startswith('J'):
        return "a"
    elif nltk_tag.startswith('V'):
        return "v"
    elif nltk_tag.startswith('N') or nltk_tag == "FW":
        return "n"
    else:
        return None

def detect_language(text):
    """Function detects language of the texts"""
    return langdetect.detect(text)

def strip_accents(s):
    """Function normalizes Unicode accents"""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def remove_latex(text):
    """ Function removes LaTeX formatting from the texts """
    if not text:
        return ''
    else:
        text_splitted = text.splitlines()
        text = ' '.join(text_splitted)
        if re.search(r'\$[a-zA-Z0-9\!\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\s]+\$', text, flags=re.I):
            text = re.sub(r'\$[a-zA-Z0-9\!\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\s]+\$', '', text, flags=re.I)
        if re.search(r'(\\\[.+\\\])', text, flags=re.I):
            text = re.sub(r'(\\\[.+\\\])', '', text, flags=re.I)
        if re.search(r'(\\\([a-zA-Z0-9\!\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\s]+\\\))', text, flags=re.I):
            text = re.sub(r'(\\\([a-zA-Z0-9\!\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\s]+\\\))', '', text, flags=re.I)
        if re.search(r'\$\$[a-zA-Z0-9\!\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\s]+\$\$', text, flags=re.I):
            text = re.sub(r'\$\$[a-zA-Z0-9\!\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\s]+\$\$', '', text, flags=re.I)
        if re.search(r'\\cite\{.+}', text, flags=re.I):
            text = re.sub(r'\\cite\{.+}', '', text, flags=re.I)
    return text

def remove_dashes(text):
    """Function removes endashes, hyphens, emdashes"""
    text_no_hyp = text.replace('-', ' ')
    text_no_dash = text_no_hyp.replace('–', ' ')
    text_no_emdash = text_no_dash.replace('—', ' ')
    return text_no_emdash

def remove_patterns(text):
    """Function removes E-mails, Web Addresses, scientific abbreviations, corrects incorrect word breaks."""
    if re.search(ABBRV_PATTERN, text):
        text = re.sub(ABBRV_PATTERN, ' ', text)
    if re.search(WWW_PATTERN, text):
        text = re.sub(WWW_PATTERN, ' ', text)
    if re.search(EMAIL_PATTERN, text):
        text = re.sub(EMAIL_PATTERN, ' ', text)
    if re.search(r'\slution|-lution|–lution|—lution|\s-lution|\s–lution|\s—lution\slutions|-lutions|–lutions|—lutions|\s-lutions|\s–lutions|\s—lutions', text):
        text = re.sub(r'\slution|-lution|–lution|—lution|\s-lution|\s–lution|\s—lution\slutions|-lutions|–lutions|—lutions|\s-lutions|\s–lutions|\s—lutions', 'lution', text)

    if re.search(r'\sndacy|-ndacy|–ndacy|—ndacy|\s-ndacy|\s–ndacy|\s—ndacy|\sndacies|-ndacies|–ndacies|—ndacies|\s-ndacies|\s–ndacies|\s—ndacies', text):
        text = re.sub(
            '\sndacy|-ndacy|–ndacy|—ndacy|\s-ndacy|\s–ndacy|\s—ndacy|\sndacies|-ndacies|–ndacies|—ndacies|\s-ndacies|\s–ndacies|\s—ndacies', 'ndacy', text
        )
    if re.search(r'\sation|-tation|–tation|—tation|\sations|-tations|–tations|—tations', text):
        text = re.sub(
            r'\sation|-tation|–tation|—tation|\sations|-tations|–tations|—tations', 'tation', text
        )
    if re.search(r'\sity|-ity|–ity|—ity|\sities|-ities|–itiesy|—ities', text):
        text = re.sub(
            r'\sity|-ity|–ity|—ity|\sities|-ities|–itiesy|—ities', 'ity', text
        )
    if re.search(r'\stion|-tion|–tion|—tion|\stions|-tions|–tions|—tions', text):
        text = re.sub(
            r'\stion|-tion|–tion|—tion|\stions|-tions|–tions|—tions', 'tion', text
        )
    return text

def remove_punctuation(tokens):
    """Function removes punctuation and various symbols."""
    no_punct = []
    for token in tokens:
        for word in token.translate(str.maketrans(SYMBOL_PATTERN, ' ' * LEN_PATTERN)).split():
            no_punct.append(word)
    return no_punct

def sentence_processing(sentences:list):
    """Function splits texts into sentences, detects language of each sentence, tokenizes sentences into tokens,
    removes withdrawn texts, normalizes accents."""
    tokens = []
    for sentence in sentences:
        try:
            if detect_language(sentence) == 'en':
                for token in nltk.word_tokenize(sentence):
                    if token in withdrawn:
                        return []
                    if token not in SYMBOL_PATTERN:
                        token = unicodedata.normalize('NFKC', token)
                        token = strip_accents(token)
                        if '^' not in token:
                            tokens.append(token)
        except langdetect.LangDetectException:
            pass
        return tokens

def singularize(tokens:list):
    """Function singularizes based on assumption that if token without 's' on the end exists, then the token
    with 's' ending can be converted to singular form."""
    singularized = []
    for token in tokens:
        if token.endswith('s'):
            if token[:-1] in tokens:
                singularized.append(token[:-1])
        else:
            singularized.append(token)
    return singularized

def correct_spelling(tokens:list, misspelt:list, correct_tokens:list):
    """Function corrects spelling of the tokens by looking up in the vocabularies produced by symspellpy"""
    corrected_tokens = []
    for token in tokens:
        if token in misspelt:
            token_ind = misspelt.index(token)
            token = correct_tokens[token_ind]
        elif token.endswith('cs'):
            token = token[:-1]
        corrected_tokens.append(token)
    return corrected_tokens

def lemmatize(tags:list, list_of_plurals:list, list_of_singulars:list, set_tokens_end_s:set):
    """Function lemmatizes tokens with NLTK and vocabulary of plurals / singles, stores tokens with 's' ending to set"""
    lemmatized_tokens = []
    for tag in tags:
        if tag[1] in TAGS_SET:
            lemmatized_tag = WNL.lemmatize(tag[0], nltk_pos_tagger(tag[1]))
            if lemmatized_tag.endswith('s'):
                lemmatized_tag = WNL.lemmatize(lemmatized_tag)
                if lemmatized_tag in list_of_plurals:
                    lemm_tag_index = list_of_plurals.index(lemmatized_tag)
                    lemmatized_tag = list_of_singulars[lemm_tag_index]
            lemmatized_tokens.append(lemmatized_tag)
            if lemmatized_tag.endswith('s'):
                set_tokens_end_s.add(lemmatized_tag)
    return lemmatized_tokens

def tokenize(text:str, misspelt:list, correct_tokens:list, set_tokens_end_s:set, token_dictionary: dict):
    """Master function which tokenizes text strings, removes patterns (emails, www addresses), removes dashes,
    tokenizes text to sentences and to tokens, removes punctuation, lemmatizes tokens, removes tokens which are not
    nouns, verbs, adjectives, removes stop words, removes digits, removes single letters, corrects spelling, stores
    possible plurals for further processing, add tokens to dictionary of tokens."""
    text_low = text.lower()
    text_splitted = text_low.splitlines()
    text_no_new_lines = ' '.join(text_splitted)
    text_no_patterns = remove_patterns(text_no_new_lines)
    text_no_dashes = remove_dashes(text_no_patterns)
    sentences = nltk.sent_tokenize(text_no_dashes)
    tokens = sentence_processing(sentences)
    no_punct = remove_punctuation(tokens)
    tags = nltk.pos_tag(no_punct)
    lemmatized_tokens = lemmatize(tags, plurals, singles, set_tokens_end_s)
    no_stops = [word for word in lemmatized_tokens if word not in stops]
    no_digits = [word for word in no_stops if not word.isdecimal()]
    no_letters = [word for word in no_digits if len(word) > 1]
    corrected_tokens = correct_spelling(no_letters, misspelt, correct_tokens)
    length_tokens = len(corrected_tokens)

    for token in corrected_tokens:
        token_dictionary[token] = token_dictionary.get(token, 0) + 1

    return corrected_tokens, length_tokens



singles = [pair[1] for pair in plurals_singles_map]
plurals = [pair[0] for pair in plurals_singles_map]
misspelt = [pair[0] for pair in spell_corrections]
correct_tokens = [pair[1] for pair in spell_corrections]


### CREATION OF STOPWORD LIST ###
stops = stopwords.words('english') # Stopwords from NLTK
stops.extend(common) # NLTK stopwords merged with science frequently used words
stops.extend(greek)
stops.extend(abbr_list)
foreign_stops = set(strip_accents(f_text.lower()).split())
stops.extend(list(foreign_stops))


n = 0 # Articles Counter
WNL = WordNetLemmatizer() # CREATION OF CLASS INSTANCE FOR WORD NET LEMMATIZER FROM NLTK MODULE ###
df = pd.DataFrame()

token_dict = {}

categories_full = []
categories_main = []
categories_split = []

s_set = set()
non_eng_text = []
title_len_list = []
abstract_len_list = []
title_list = []
abstract_list = []

lb = 0 # lower bound
ub = 1800000 # upper bound

with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    try:
        for line in f:
            if lb < n < ub:
                if n not in EXCEPTIONS:
                    json_line = json.loads(line)
                    title_raw = json_line['title']
                    title_raw = remove_latex(title_raw)
                    abstract_raw = json_line['abstract']
                    abstract_raw = remove_latex(abstract_raw)
                    category_raw = json_line['categories']
                    try:
                        language_title = detect_language(title_raw)
                    except langdetect.LangDetectException:
                        language_title = None
                    try:
                        language_abstract = detect_language(abstract_raw)
                    except langdetect.LangDetectException:
                        language_abstract = None
                    if language_title == 'en':
                        title_processed, token_len_title = tokenize(title_raw, plurals, singles, s_set, token_dict)
                    else:
                        title_processed, token_len_title = [], 0
                    if language_abstract == 'en':
                        abstract_processed, token_len_abstract = tokenize(abstract_raw, plurals, singles, s_set, token_dict)
                        if abstract_processed:
                            if title_processed != abstract_processed:
                                title_list.append(title_processed)
                            else:
                                title_list.append([])
                            abstract_list.append(abstract_processed)
                            title_len_list.append(token_len_title)
                            abstract_len_list.append(token_len_abstract)
                            main_category = []
                            categories_full.append(category_raw)
                            category_split = category_raw.split()
                            categories_split.append(category_split)
                            for category in category_split:
                                if category in CS:
                                    main_category.append('cs')
                                elif category in MATH:
                                    main_category.append('math')
                                elif category in Q_BIO:
                                    main_category.append('q_bio')
                                elif category in ECON:
                                    main_category.append('econ')
                                elif category in Q_FIN:
                                    main_category.append('q_fin')
                                elif category in PHYSICS:
                                    main_category.append('physics')
                                elif category in EESS:
                                    main_category.append('eess')
                                elif category in STAT:
                                    main_category.append('stat')
                            categories_main.append(main_category)
                    else:
                        text = title_raw + ' ' + abstract_raw
                        non_eng_text.append(text)
            n += 1
            if not n % 1000:
                print(n)
            if n == ub:
                break
    except (json.decoder.JSONDecodeError, IndexError):
        pass


### GENERATION OG POSSIBLE PLURALS ###
possible_plurals = set()

for token in s_set:
    if len(token) > 1:
        poss_plural = token[:-1]
        if poss_plural in token_dict:
            possible_plurals.add(poss_plural)


### STORAGE OF PREPROCESSED DATA TO DATAFRAME ###
df['title'] = title_list
df['abstract'] = abstract_list
df['len_title'] = title_len_list
df['len_abstract'] = abstract_len_list
df['category_main'] = categories_main
df['categories_full'] = categories_full
df['categories_split'] = categories_split

df_plurals = pd.DataFrame(list(s_set))
df_non_eng = pd.DataFrame(non_eng_text)
df_non_eng.to_pickle('non_eng_no_latex_ver_6_stops.pk', protocol=4)
df.to_pickle('prepr_voc_singles_no_latex_ver_6_stops.pk', protocol=4)
df_plurals.to_pickle('possible_plurals_no_latex_ver_6_stops.pk', protocol=4)
print()
