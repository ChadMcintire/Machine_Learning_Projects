# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 13:26:11 2018

@author: Chad
"""
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
output_notebook()
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

filepath = "C:/Users/Chad/Desktop/450/Week 11/Try3.csv"
df1 = pd.read_csv(filepath,delimiter=',', encoding='cp1252', header = None)
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="https", value="")
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="que", value="")
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="por", value="")
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="para", value="")
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="com", value="")
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="like", value="")
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="don", value="")


df1.fillna(0, inplace=True)
#df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="?", value="")

#maybe get rid
df1.loc[:, 1] = df1.loc[:, 1].replace(regex=True, to_replace="just", value="")


#df1[1].replace("https", "", inplace=True)

NUM_TOPICS = 5
 
vectorizer = CountVectorizer(min_df=5, max_df=0.7, 
                             stop_words='english', lowercase=True, 
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

data_vectorized = vectorizer.fit_transform(df1[1].values.astype('U'))

lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)
print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print("LDA Model:")
print_topics(lda_model, vectorizer)
print("=" * 20)

svd = TruncatedSVD(n_components=2)
documents_2d = svd.fit_transform(data_vectorized)
 
df = pd.DataFrame(columns=['x', 'y', 'document'])
df['x'], df['y'], df['document'] = documents_2d[:,0], documents_2d[:,1], range(len(df1))
 
source = ColumnDataSource(ColumnDataSource.from_df(df))
labels = LabelSet(x="x", y="y", text="document", y_offset=8,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
 
plot = figure(plot_width=600, plot_height=600)
plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
plot.add_layout(labels)
show(plot)
