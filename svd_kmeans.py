import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import logging
from sklearn.decomposition import PCA


# CONFIGURATION SETTINGS
N_COMP = 500    # Number of components for Truncated SVD
TRUE_K = 2 * N_COMP    # Number of clusters for K-Means
MIN_DF = 2      # Minimum document frequency of TF-IFD
MAX_DF = 5000   # Maximum document frequency of TF-IFD
PERPLEXITY = 30  # Perplexity of TSNE


# LOGGING BLOCK
LEVEL = logging.DEBUG
logger = logging.getLogger('SVD_logger')
logger.setLevel(LEVEL)
fh = logging.FileHandler('svd_{0}f_{1}min_df_{2}lmax_df_{3}K_trial_2.log'.format(N_COMP, MIN_DF, MAX_DF, TRUE_K))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


# DECLARATION OF FUNCTIONS
def join_list(input):
    text = ''
    if isinstance(input, list):
        text += ' '.join(input)
    elif isinstance(input, str):
        text += input
    return text


# UNPACKING THE DATAFRAME
df = pd.read_pickle('prepr_voc_singles_no_latex_ver_4.pk')


# EXTRACTION OF TEXT CORPUS
df['merged_title_abstract'] = df['title'] + df['abstract']
df['text'] = df.apply(lambda row: join_list(row['merged_title_abstract']), axis=1)
corpus = df['text']
print('Corpus size: ', corpus.shape[0])
logger.info('Corpus size: {}'.format(corpus.shape[0]))


# FEATURE EXTRACTION WITH TF-IDF
vectorizer = TfidfVectorizer(
    max_df=MAX_DF,
    min_df=MIN_DF,
    use_idf=True,
    norm='l2',
    strip_accents='unicode'
    )

X = vectorizer.fit_transform(corpus)
print('Shape of the TF-DF matrix after 1st vectorizer: ', X.shape)
logger.info('Shape of the TF-DF matrix after 1st vectorizer: {}'.format(X.shape))


# FILTERING ROWS WITH ALL ZEROS
non_zeros = X.getnnz(axis=1)
non_zero_list = [ind for ind, x in enumerate(list(non_zeros)) if x > 0]
corpus_no_zeros = df['text'].iloc[non_zero_list]


# FEATURE EXTRACTION WITH TF-IDF ON NON-MATRIX
X_nz = vectorizer.fit_transform(corpus_no_zeros)
print('X.shape = ', X_nz.shape, 'Shape of the TF-DF matrix after 2nd vectorizer')
logger.info('Shape of the TF-DF matrix after 2nd vectorizer {}'.format(X_nz.shape))

# EXTRACTION OF TERMS
terms = vectorizer.get_feature_names_out()


# DIMENSION REDUCTION WITH TRUNCATED SVD
svd = TruncatedSVD(n_components=N_COMP, algorithm='arpack', random_state=42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X_LSA_SVD = lsa.fit_transform(X_nz)


print("X.shape = {} after LSA".format(X_LSA_SVD.shape))
logger.info("X.shape = {} after LSA".format(X_LSA_SVD.shape))

explained_variance = svd.explained_variance_ratio_.sum()
logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
print(
    "Explained variance of the SVD step: {}%".format(int(explained_variance * 100))
)


# CLUSTERING WITH K-MEANS
sil_scores = []
km_objects = []
best_cluster = 0

for k in range(2, TRUE_K):
    km = MiniBatchKMeans(
        n_clusters=TRUE_K,
        max_iter=300,
        init="k-means++",
        n_init=300,
        verbose=False,
    )
    print("Number of clusters: {}".format(k))
    logger.info("Number of clusters: {}".format(k))

    km.fit(X_LSA_SVD)
    km_objects.append(km)

    labels = km.labels_

    sil_score = metrics.silhouette_score(X_LSA_SVD, km.labels_,
                                        sample_size=100000)
    sil_scores.append(sil_score)
    print("Silhouette Coefficient: %0.3f" % sil_score)
    logger.info("Silhouette Coefficient: {0: .3f}".format(sil_score))

    max_sil_score = max(sil_scores)
    best_cluster = sil_scores.index(max_sil_score)
    print("Best Cluster: ", best_cluster+2)


best_km = km_objects[best_cluster - 1]
original_space_centroids = svd.inverse_transform(best_km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]


# PRODUCTION OF TOPICS FOR THE NUMBER OF CLUSTERS WITH BEST SILHOUETTE SCORE
for i in range(best_cluster + 2):
    print("Cluster %d:" % i, end="")
    logger.info("Cluster %d: {}".format(i))
    for ind in order_centroids[i, :10]:
        print(" %s" % terms[ind], end="")
        logger.info(" {}".format(terms[ind]))
    print()


# t-SNE VISUALIZATION OF CLUSTERS
pca = PCA(64)  # Decreasing number of dimensions with PCA to 64
pca_projected = pca.fit_transform(X_LSA_SVD)
projection = TSNE(perplexity=PERPLEXITY).fit_transform(pca_projected)  # Application of TSNE on a matrix reduced by PCA
labels = best_km.labels_
n_labels = len(set(labels))
palette = np.array(sns.color_palette("hls", n_labels))  # Color palette with seaborn
f = plt.figure(figsize=(18, 12))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(*projection.T, linewidth=0, s=0.2, color=palette[labels])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.axis('tight')
plt.title('t-SNE visualization of {} clusters produced by K-Means'.format(best_cluster + 2), fontsize=16)
plt.savefig("kmmb_svd_{0}_min_{1}_max_{2}_perpl_{3}.png".format(N_COMP, MIN_DF, MAX_DF, PERPLEXITY))