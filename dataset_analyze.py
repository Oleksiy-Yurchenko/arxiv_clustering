import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


### LOGGING BLOCK ###
LEVEL = logging.DEBUG
logger = logging.getLogger('SVD_logger')
logger.setLevel(LEVEL)
fh = logging.FileHandler('dataset_analyze.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


### EXTRACTION OF DATAFRAME ###
df = pd.read_pickle('prepr_voc_singles_no_latex_ver_4.pk')


print("Number of unique categories: {}".format(df['categories_full'].nunique()))
logger.info("Number of unique categories: {}".format(df['categories_full'].nunique()))




### GENERATION OF SCIENCE CATEGORIES PIE CHART ###

category_main = df['category_main']

cs = 0
math = 0
q_bio = 0
econ = 0
q_fin = 0
physics = 0
eess = 0
stat = 0

n = 0
for category in category_main:
    # n += 1
    # print(i)
    if category[0] == 'cs':
        cs += 1
    if category[0] == 'math':
        math += 1
    if category[0] == 'q_bio':
        q_bio += 1
    if category[0] == 'econ':
        econ += 1
    if category[0] == 'q_fin':
        q_fin += 1
    if category[0] == 'physics':
        physics += 1
    if category[0] == 'eess':
        eess += 1
    if category[0] == 'stat':
        stat += 1


print('Qtty cs: ', cs)
logger.info('Qtty cs: {}'.format(cs))
print('Qtty math: ', math)
logger.info('Qtty math: {}'.format(math))
print('Qtty q_bio: ', q_bio)
logger.info('Qtty q_bio: {}'.format(q_bio))
print('Qtty econ: ', econ)
logger.info('Qtty econ: {}'.format(econ))
print('Qtty q_fin: ', q_fin)
logger.info('Qtty q_fin: {}'.format(q_fin))
print('Qtty physics: ', physics)
logger.info('Qtty physics: {}'.format(physics))
print('Qtty eess: ', eess)
logger.info('Qtty eess: {}'.format(eess))
print('Qtty stat: ', stat)
logger.info('Qtty stat: {}'.format(stat))


names = ['cs', 'math', 'q_bio', 'econ', 'q_fin', 'physics', 'eess', 'stat']
values = [cs, math, q_bio, econ, q_fin, physics, eess, stat]
plt.rcParams.update({'font.size': 9})
plt.pie(values, labels=names, autopct='%0.1f%%', pctdistance=1.2, labeldistance=1.35)
plt.title('Categories of Sciences')
plt.savefig('pie_chart_categories.png')
 

### GENERATION OF TITLES LENGTH DISTRIBUTION CHART ###
title_len_list = df['len_title']
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.hist(title_len_list, bins=30, label=f"{df['len_title'].describe()}")
ax1.set_ylabel('Number of documents')
ax1.set_xlabel('Length of titles')
ax1.set_title('Distribution of titles length')
ax1.legend(fontsize="x-small")
# plt.savefig('distribution_title_length.png')



### GENERATION OF ABSTRCTS LENGTH DISTRIBUTION CHART ###
abstract_len_list = df['len_abstract']
ax2 = fig.add_subplot(122)
ax2.hist(abstract_len_list, bins=100, label=f"{df['len_abstract'].describe()}")
ax2.set_ylabel('Number of documents')
ax2.set_xlabel('Length of abstracts')
ax2.set_title('Distribution of abstracts length')
ax2.legend(fontsize="x-small")
fig.tight_layout(pad=1.0)
plt.savefig('distribution_titles_abstract_length.png')


def join_list(input):
    text = ''
    if isinstance(input, list):
        text += ' '.join(input)
    elif isinstance(input, str):
        text += input
    return text

df['merged_title_abstract'] = df['title'] + df['abstract']
df['text'] = df.apply(lambda row: join_list(row['merged_title_abstract']), axis=1)
corpus = df['text']



vectorizer = TfidfVectorizer(
    max_df=1.0,
    min_df=0.0,
    use_idf=True,
    norm='l2',
    strip_accents='unicode'
    )

X_all = vectorizer.fit_transform(corpus)

all = X_all.shape[1]
print('Quantity of all tokens: ', all)
logger.info('Quantity of all tokens'.format(all))


vectorizer = TfidfVectorizer(
    max_df=5000,
    min_df=2,
    use_idf=True,
    norm='l2',
    strip_accents='unicode'
    )

X_target = vectorizer.fit_transform(corpus)

target = X_target.shape[1]
print('Quantity of tokens with min_df = 2 and max_df = 5000: ', target)
logger.info('Quantity of tokens with min_df = 2 and max_df = 5000'.format(target))


vectorizer = TfidfVectorizer(
    max_df= 1,
    use_idf=True,
    norm='l2',
    strip_accents='unicode'
    )

X_one = vectorizer.fit_transform(corpus)

one = X_one.shape[1]
print('Quantity of tokens with DF = 1: ', one)
logger.info('Quantity of tokens with DF = 1'.format(one))


vectorizer = TfidfVectorizer(
    min_df=5001,
    use_idf=True,
    norm='l2',
    strip_accents='unicode'
    )

X_rest = vectorizer.fit_transform(corpus)
rest = X_rest.shape[1]
print('Quantity of tokens with DF > 5000: ', rest)
logger.info('Quantity of tokens with DF > 5000'.format(rest))

doc_freq = [one, target, rest]
doc_freq_ranges = ['df = 1', 'min_df = 2 and max_df = 5000', 'df > 5000']

fig, ax = plt.subplots()

df_doc_freq = pd.DataFrame([['df = 1', one], ['min_df = 2 and max_df = 5000', target], ['df > 5000', rest]],
                           columns=['Document Frequency','Token Count'])
sns.color_palette('bright')
sns.barplot(df_doc_freq, x='Document Frequency', y='Token Count', hue='Token Count', palette=["C0", "C1", "C2"]).set_title('Token Count - Document Frequency')

plt.savefig('token_frequency_count.png')
