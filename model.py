import numpy as np
import pandas as pd
import metachange
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import spacy
import nltk
import re
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer
import igraph as ig
from nltk.corpus import stopwords
from tqdm import tqdm
nltk.download('stopwords')

# stp=stopwords.words('english')
stp=stopwords.words('french')

# nlp = spacy.load('en_core_web_md')
nlp = spacy.load('fr_core_news_sm')

class ChangePointIndicator():
    def __init__(self, sampling_size=1000, feature_size=1000, min_range=86400, max_depth=3):
        self.sampling_size = sampling_size
        self.feature_size = feature_size
        self.min_range = min_range
        self.max_depth = max_depth
        self.indicator = ig.Graph(directed=True)
        self.min_time = 9999999999999
        self.min_idx = -1
        self.max_time = -1
        self.max_idx = -1
        self.leaf_dict = {}


    def run(self, df):
        print("sampling dataset ...")
        sample_df = self.read_dataset(df)
        # sample_df.to_pickle("phase1b_6m_sample_data.p")
        print("preprocessing data ...")
        input_df = self.preprocess(sample_df)
        print("vectorizing data ...")
        tfidf_feature_df = self.vectorize(input_df)
        print("detecting change points ... (need some time)")
        res_multi = self.detect_changepoint(tfidf_feature_df)
        self.create_indicator(res_multi)


    def read_dataset(self, twitter_df):
        # sampling certain number for each date
        twitter_df["date"] = [datetime.fromtimestamp(time/1000).date() for time in twitter_df['timePublished'].values]
        date_list = twitter_df["date"].tolist()
        uni_date_list = list(set(date_list))
        uni_date_list.sort()
        sample_df_list = []
        date_num_dict = {}
        for date in uni_date_list:
            target_df = twitter_df[twitter_df["date"] == date]
            date_num_dict[date] = len(target_df)
            if len(target_df) > self.sampling_size:
                sample_df_list.append(target_df.sample(self.sampling_size))
            else:
                sample_df_list.append(target_df)
        sample_df = pd.concat(sample_df_list, ignore_index=True)
        sample_df = sample_df.drop(["date"], axis=1)
        return sample_df


    def preprocess(self, sample_df):
        structural_features_raw = []
        for row in tqdm(sample_df.itertuples(), total=len(sample_df)):
        # for row in sample_df.itertuples():
            contentText = row[1]
            lem_text = self.lemmatize(contentText)
            deep_clean_text = self.deep_clean(lem_text.strip())
            text_arr = deep_clean_text.split()
            clean_text_arr = [word for word in text_arr if word not in stp]
            clean_text = ' '.join(clean_text_arr)
            structural_features_raw.append({
                "clean_text": clean_text
            })
        structural_features_df = pd.DataFrame(structural_features_raw, index=sample_df.index)
        input_df = pd.concat([sample_df, structural_features_df], axis=1)
        return input_df


    def vectorize(self, input_df):
        processed_text_str_list = input_df['clean_text'].tolist()
        vectorizer = TfidfVectorizer(max_features=self.feature_size)
        vectorizer.fit(processed_text_str_list)
        features_raw = []
        for row in tqdm(input_df.itertuples(), total=len(input_df)):
        # for row in input_df.itertuples():
            text_str = row[-1]
            text_str_list = [text_str]
            vector = vectorizer.transform(text_str_list)
            vector = vector.toarray()
            features_raw.append({
                "tf_idf_vec": vector[0],
            })
        features_df = pd.DataFrame(features_raw, index=input_df.index)
        tfidf_feature_df = pd.concat([input_df, features_df], axis=1)
        return tfidf_feature_df


    def detect_changepoint(self, tfidf_feature_df):
        tfidf_feature_df['encoded_timestamp'] = [time / 1000 for time in tfidf_feature_df['timePublished'].values]
        vec_list = tfidf_feature_df['tf_idf_vec'].tolist()
        date_list = tfidf_feature_df['encoded_timestamp'].tolist()
        vec_array = np.array(vec_list)
        date_array = np.array(date_list)

        clf_rf = RandomForestClassifier(max_depth=32, criterion="entropy", random_state=0)
        res_multi, res_multi_result = metachange.change_point_tree(vec_array, date_array, clf_rf, min_range=self.min_range, max_d=self.max_depth)
        # print("============= detect_changepoint =============")
        # print(res_multi_result)
        return res_multi


    def create_indicator(self, res_multi):

        local_indicator = ig.Graph(directed=True)
        local_indicator.add_vertex(0)
        self.rec_tree(res_multi, 0, local_indicator, [0], self.make_node_text)
        num_vertex = local_indicator.vcount()
        for i in range(num_vertex):
            node_info = local_indicator.vs[i]["info"]
            if node_info[0] == 0:
                self.leaf_dict[i] = (node_info[1], node_info[2])
                if node_info[1] < self.min_time:
                    self.min_time = node_info[1]
                    self.min_idx = i
                if node_info[2] > self.max_time:
                    self.max_time = node_info[2]
                    self.max_idx = i

        # print("****************** create_indicator ********************")
        print(self.leaf_dict)
        self.indicator = local_indicator

    # # for test
    # def get_indicator(self):
    #     return self.indicator
    #
    # def get_max(self):
    #     return self.max_idx, self.max_time
    #
    # def get_min(self):
    #     return self.min_idx, self.min_time

    def annotate(self, input_time):
        time = input_time/1000
        within_span_idx = -1
        for leaf_idx, time_span in self.leaf_dict.items():
            if time >= time_span[0] and time <= time_span[1]:
                within_span_idx = leaf_idx
            elif time > self.max_time:
                within_span_idx = self.max_idx
            elif time < self.min_time:
                within_span_idx = self.min_idx
            else:
                continue

        parents = self.indicator.neighborhood(within_span_idx, order=10000000, mode='in', mindist=1)

        annotate_results = []
        for i in range(len(parents)):
            node_info = self.indicator.vs[parents[i]]["info"]
            # [change point date, alpha, depth]
            result = [datetime.fromtimestamp(node_info[3]).date(), node_info[4], len(parents) - i]
            annotate_results.append(result)
        return annotate_results


    def rec_tree(self, node_i, node_idx, dot, dot_info, make_node_text):
        node_label = node_idx
        node_text = make_node_text(node_i["data"])
        dot.vs[node_label]["info"] = node_text

        if (node_i["left"] != None):
            dot_info[0] += 1
            left_idx = dot_info[0]
            left_label = left_idx
            dot.add_vertex(left_label)
            dot.add_edge(node_label, left_label)
            self.rec_tree(node_i["left"], left_idx, dot, dot_info, make_node_text)
        if (node_i["right"] != None):
            dot_info[0] += 1
            right_idx = dot_info[0]
            right_label = right_idx
            dot.add_vertex(right_label)
            dot.add_edge(node_label, right_label)
            self.rec_tree(node_i["right"], right_idx, dot, dot_info, make_node_text)


    def make_node_text(self, data):
        t_left = data["t_left"]
        t_right = data["t_right"]
        if "t0" in data:
            header = 1
            changepoint = data["t0"]
            alpha = round(data["alpha"], 4)
        else:
            header = 0
            changepoint = -1
            alpha = -1
        return (header, t_left, t_right, changepoint, alpha)


    def lemmatize(self, sent):
        s = [token.lemma_ for token in nlp(sent)]
        s = ' '.join(s)
        return s


    def deep_clean(self, x):
        irrelevant_chars = "~?!./\:;+=&^%$#@(,)[]_*"
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        x = x.lower()
        x = re.sub(r'http\S+', '', x)
        remove_digits = str.maketrans('', '', digits)
        remove_chars = str.maketrans('', '', irrelevant_chars)
        x = x.translate(remove_digits)
        x = x.translate(remove_chars)
        x = emoji_pattern.sub(r'', x)
        x = x.replace('!', '')
        x = x.replace('?', '')
        x = x.replace('@', '')
        x = x.replace('&', '')
        x = x.replace('$', '')
        x = x.replace('``', '')
        x = x.replace("'s", '')
        x = x.replace("''", '')
        x = [t for t in x.split() if len(t) > 3]
        x = ' '.join(x)
        return x
