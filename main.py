# This is a sample Python script.
import datetime
import multiprocessing
import pickle
import random
import re
import string
from asyncio import sleep

import joblib
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import xgboost as xgb
from wordbatch.models import FM_FTRL
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from nltk import word_tokenize
import nltk
# nltk.download("all")
import stopwatch
import os
import openpyxl
from text_prepare import process_files
from matplotlib import pyplot as plt

# import pandas_ml as pdml
log_file_name = ("logs\\log_main_" + str(datetime.datetime.now()) + ".txt").replace(":", "-")

os.environ["MODIN_ENGINE"] = "ray"
with open(log_file_name, mode="a", encoding="utf8") as file:
    file.write("\n\n\n\n\t\tNew process --------------------------------------------------\n\n\n\n")


# process_files("I:\\projects\\PycharmProjects\\ai_text_detection\\data\\psych_texts")
def fprint(data=""):
    with open(log_file_name, mode="a", encoding="utf8") as file:
        if str(data) != "":
            file.write(str(datetime.datetime.now()) + ":\n")
            file.write(str(data).replace('.', ',') + "\n\n")
        else:
            file.write("\n")
    if str(data) != "":
        print(str(datetime.datetime.now()) + ":\n" + str(data))
    print()


atd = False
reload = False
re_vectorize = True
only_vectorize = False
short = False

per_class_cap = 100000


class FM_FTRL_sklearn(BaseEstimator):
    def __init__(self, X):
        self._estimator = FM_FTRL(alpha=0.02, beta=0.01, L1=0.0001, L2=1.0,
                                  D=X.shape[1], alpha_fm=0.03, L2_fm=0.005, init_fm=0.01,
                                  D_fm=20, weight_fm=10.0, e_noise=0.0001, e_clip=1.0,
                                  iters=5, inv_link="identity", threads=1,
                                  )

    def fit(self, X, y):
        self._estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = self._estimator.predict(X)
        probas_array = np.array([np.append(0, probas[i]) for i in range(len(probas))])
        return probas_array


def crop_data(per_class_count=50000):
    st.reset()
    st.start()
    generated_texts = pd.read_json("data/combined_data/generated_texts.json")
    generated_texts = generated_texts.head(min(per_class_count, generated_texts.shape[0]))
    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))
    generated_texts.to_json("data/combined_data/generated_texts_short.json")

    st.reset()
    st.start()
    human_texts = pd.read_json("data/combined_data/human_texts.json")
    human_texts = human_texts.head(min(per_class_count, human_texts.shape[0]))
    st.stop()
    fprint("loaded human texts, elapsed time = " + str(st.duration))
    human_texts.to_json("data/combined_data/human_texts_short.json")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    fprint(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


st = stopwatch.Stopwatch()
done = 0
part = 0
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default

exclude = set(string.punctuation)
exclude.update(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])


def preprocess(line):
    current = line
    current = current.replace(r"\n", r" ")
    current = current.replace(r"\r", r" ")
    # Drop puntuation
    current = ''.join(ch for ch in current if ch not in exclude)
    # Remove extra spaces from '😃  For' to '😃 For'
    current = re.sub(r"\s+", r" ", current)
    # Remove leading and trailing whitespace
    current = current.strip()
    return current
    tokenized = word_tokenize(current, language="russian")
    # cleaned = ""
    # for word in tokenized:
    #     if word not in stopwords:
    #         cleaned += word + " "
    # cleaned = cleaned.strip()
    return cleaned


# Press the green button in the gutter to run the script.


stopwords = list()

with open("stopwords-ru.txt", encoding="utf8") as file:
    lines = file.readlines()
    fprint()
    result = ""
    for line in lines:
        result += '"' + line.strip() + '",\n'
        stopwords.append(line.strip())
    fprint()


def text_splitter(x):
    return re.findall(r'[a-zA-Zа-яА-Я]+', x)


def rename_to_zeorones(x):
    if x == "H":
        return 0
    return 1


def load_from_json_256(sample=100000):
    # min_text_count = 100000000
    # st.reset()
    # st.start()
    # human1 = pd.read_csv("data\\vk\\train.csv")
    # human2 = pd.read_csv("data\\vk\\test.csv")
    # human1.drop('oid', axis=1, inplace=True)
    # human1.drop('category', axis=1, inplace=True)
    # human2.drop('oid', axis=1, inplace=True)
    #
    # human1.rename(columns={human1.columns[0]: 'text'}, inplace=True)
    # human2.rename(columns={human2.columns[0]: 'text'}, inplace=True)
    # human1["generated"] = np.zeros(shape=(len(human1), 1), dtype=np.int8)
    # human2["generated"] = np.zeros(shape=(len(human2), 1), dtype=np.int8)
    # human_ = pd.concat([human1, human2], ignore_index=True)
    #
    # fprint(str(human1.columns))
    # fprint(str(human2.columns))
    #
    # human_["clean"] = human_["text"]
    # # human_["generated"] = human1["generated"].apply(rename_to_zeorones)
    # _human_train = human_.head(sample)
    # st.stop()
    # fprint("loaded human texts, elapsed time = " + str(st.duration))

    min_text_count = 100000000
    st.reset()
    st.start()
    human1 = pd.read_json("content\\habr.json")
    # human1.drop('oid', axis=1, inplace=True)
    # human1.drop('category', axis=1, inplace=True)

    human1.rename(columns={human1.columns[0]: 'text'}, inplace=True)
    human1["generated"] = np.zeros(shape=(len(human1), 1), dtype=np.int8)
    human_ = human1

    fprint(str(human1.columns))

    human_["clean"] = human_["text"]
    # human_["generated"] = human1["generated"].apply(rename_to_zeorones)

    st.stop()
    fprint("loaded human texts, elapsed time = " + str(st.duration))

    st.reset()
    st.start()
    eval_human = pd.read_excel("data/psych_texts/texts_256.xlsx")
    eval_human["clean"] = eval_human[0]
    eval_human.rename(columns={eval_human.columns[0]: 'text'}, inplace=True)
    eval_human.rename(columns={"text": "text", "type": "generated"}, inplace=True)
    eval_human["generated"] = eval_human["generated"].apply(rename_to_zeorones)
    st.stop()
    fprint("loaded psych texts, elapsed time = " + str(st.duration))
    st.reset()
    st.start()
    gener_ = pd.read_json("content\\atd_gen.json")
    gener_.rename(columns={gener_.columns[0]: 'text'}, inplace=True)
    gener_["generated"] = gener_["generated"].apply(rename_to_zeorones)
    gener_["clean"] = gener_["text"]

    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))

    # st.reset()
    # st.start()
    # gener_ = pd.read_json("content\\generated.json")
    # gener_.rename(columns={gener_.columns[0]: 'text'}, inplace=True)
    # gener_["generated"] = gener_["generated"].apply(rename_to_zeorones)
    # gener_["clean"] = gener_["text"]
    # _generated_train = gener_.sample(n=min(sample, human_.shape[0]))
    # _generated_eval = gener_.tail(min(sample, eval_human.shape[0]))
    # st.stop()
    # fprint("loaded generated texts, elapsed time = " + str(st.duration))

    # st.reset()
    # st.start()
    # human = pd.read_json("content\\human.json")
    # human.rename(columns={human.columns[0]: 'text'}, inplace=True)
    # human["clean"] = human["text"]
    # human["generated"] = human["generated"].apply(rename_to_zeorones)
    # _human_train = human.head(sample)
    # st.stop()
    # fprint("loaded human texts, elapsed time = " + str(st.duration))

    _generated_eval = gener_.tail(min(sample, eval_human.shape[0]))
    _generated_train = gener_.sample(n=min(sample, human_.shape[0], gener_.shape[0]))
    _human_train = human_.head(min(sample, human_.shape[0], gener_.shape[0]))

    st.reset()
    st.start()
    _eval = pd.concat([_generated_eval, eval_human], ignore_index=True)
    _full = pd.concat([_generated_train, _human_train], ignore_index=True)
    _full = _full.sample(frac=1).reset_index()
    _eval = _eval.sample(frac=1).reset_index()
    fprint(_full.shape)
    fprint(_full.head(10))

    h = ""
    i = 0
    for row in _full.iterrows():
        h = h + str(row[1]["clean"]) + "\t"+str(row[1]["generated"])+"\n"
        i += 1
        if i > 30:
            break
    fprint("full")
    fprint(h)

    h = ""
    i = 0
    for row in _eval.iterrows():
        h = h + str(row[1]["clean"]) + "\t" + str(row[1]["generated"]) + "\n"
        i+=1
        if i>30:
            break
    fprint("eval")
    fprint(h)

    fprint(_eval.shape)
    fprint(_eval.head(10))
    fprint("shuffled data, elapsed time = " + str(st.duration))
    return _full, _eval


def load_data():
    st.reset()
    st.start()
    with open("full.pickle", mode="rb") as file:
        full = pickle.load(file)
    full = full.sample(frac=1, random_state=42).reset_index()
    st.stop()
    fprint("loaded texts, elapsed time = " + str(st.duration))
    return full


def load_data_RuATD():
    st.reset()
    st.start()
    with open("full_atd.pickle", mode="rb") as file:
        full = pickle.load(file)
    st.stop()
    fprint("loaded texts, elapsed time = " + str(st.duration))
    full = full.drop('Id', axis=1)
    return full


def load_new_data(per_class_count, load_cropped=False):
    # st.start()
    # generated_texts = pd.read_json("data/combined_data/generated_reduced.json")
    # fprint("loaded generated texts, elapsed time = " + str(st.duration))
    # st.reset()
    # st.start()
    # human_texts = pd.read_json("data/combined_data/human_reduced.json")
    # fprint("loaded human texts, elapsed time = " + str(st.duration))
    #

    st.reset()
    st.start()
    if not load_cropped:
        generated_texts = pd.read_json("data/combined_data/generated_texts.json")
    else:
        generated_texts = pd.read_json("data/combined_data/generated_texts_short.json")
    generated_texts = generated_texts.head(min(per_class_count, generated_texts.shape[0]))
    gen = generated_texts.tail(min(per_class_count, generated_texts.shape[0]))
    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))

    done = 0
    part = 0
    gen_clean = Parallel(n_jobs=cpus, verbose=60)(delayed(preprocess)(x) for x in generated_texts[0])
    fprint("finished preprocessing generated texts")

    st.reset()
    st.start()
    if not load_cropped:
        human_texts = pd.read_json("data/combined_data/human_texts.json")
    else:
        human_texts = pd.read_json("data/combined_data/human_texts_short.json")
    human_texts = human_texts.head(min(per_class_count, human_texts.shape[0]))
    st.stop()
    fprint("loaded human texts, elapsed time = " + str(st.duration))

    done = 0
    part = 0
    human_clean = Parallel(n_jobs=cpus, verbose=60)(delayed(preprocess)(x) for x in human_texts[0])
    fprint("finished preprocessing human texts")

    generated_texts["clean"] = gen_clean
    human_texts["clean"] = human_clean
    fprint(human_texts["clean"])

    # for i in range(0, generated_texts.shape[0]):
    #     current = generated_texts[0][i]
    #     current = word_tokenize(current, language="russian")
    #     cleaned = ""
    #     for word in current:
    #         if word not in stopwords:
    #             cleaned += word + " "
    #     cleaned = cleaned.strip()
    #     generated_texts.loc[i, "clean"] = cleaned
    #     part += 1
    #     if part >= generated_texts.shape[0] / 100 / 5:
    #         done += part
    #         fprint("done " + str(round(done / generated_texts.shape[0] * 100, 2)) + "%")
    #         part = 0
    # generated_texts["clean"][i] = cleaned

    # done = 0
    # part = 0
    # for i in range(0, human_texts.shape[0]):
    #     current = human_texts[0][i]
    #     current = word_tokenize(current, language="russian")
    #     cleaned = ""
    #     for word in current:
    #         if word not in stopwords:
    #             cleaned += word + " "
    #     cleaned = cleaned.strip()
    #     human_texts.loc[i, "clean"] = cleaned
    #     part += 1
    #     if part >= generated_texts.shape[0] / 100 / 5:
    #         done += part
    #         fprint("done " + str(round(done / generated_texts.shape[0] * 100, 2)) + "%")
    #         part = 0

    # gen_reduced = generated_texts[0:1000]
    # human_reduced = human_texts[0:1000]
    # gen_reduced.to_json("data/combined_data/generated_reduced.json")
    # human_reduced.to_json("data/combined_data/human_reduced.json")
    generated_texts["generated"] = np.ones(shape=(len(generated_texts), 1), dtype=np.int8)
    human_texts["generated"] = np.zeros(shape=(len(human_texts), 1), dtype=np.int8)
    generated_texts.rename(columns={generated_texts.columns[0]: "text"}, inplace=True)
    human_texts.rename(columns={human_texts.columns[0]: "text"}, inplace=True)
    full = pd.concat([generated_texts, human_texts], ignore_index=True)
    with open("full.pickle", mode="wb") as file:
        pickle.dump(full, file)
    return full


def save_as_txt():
    st.reset()
    st.start()
    atd_texts = pd.read_csv("content/posts.csv")
    atd_texts = atd_texts["text"]

    # atd_texts.drop('Id', axis=1, inplace=True)
    # atd_texts.rename(columns={atd_texts.columns[0]: "text"}, inplace=True)
    # atd_texts.rename(columns={atd_texts.columns[1]: "generated"}, inplace=True)
    with open("content\\habr.txt", 'w', encoding="utf8") as f:
        for item in atd_texts:
            f.write(str(item) + "\n******\n")
    # np.savetxt("content\\gen.txt", generated_texts["text"].values, fmt='%d', delimiter="*****")
    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))
    return 0

    st.reset()
    st.start()
    atd_texts = pd.read_csv("data/combined_data/ruatd.csv")
    atd_texts.drop('Id', axis=1, inplace=True)
    atd_texts.rename(columns={atd_texts.columns[0]: "text"}, inplace=True)
    atd_texts.rename(columns={atd_texts.columns[1]: "generated"}, inplace=True)
    with open("content\\atd_gen.txt", 'w', encoding="utf8") as f:
        for item in atd_texts["text"].values:
            f.write(str(item) + "\n******\n")
    # np.savetxt("content\\gen.txt", generated_texts["text"].values, fmt='%d', delimiter="*****")
    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))
    return 0

    st.reset()
    st.start()
    generated_texts = pd.read_json("data/combined_data/generated_texts.json")
    generated_texts.rename(columns={generated_texts.columns[0]: "text"}, inplace=True)
    with open("content\\gen.txt", 'w', encoding="utf8") as f:
        for item in generated_texts["text"].values:
            f.write(str(item) + "\n******\n")
    # np.savetxt("content\\gen.txt", generated_texts["text"].values, fmt='%d', delimiter="*****")
    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))
    st.reset()
    st.start()
    human_texts = pd.read_json("data/combined_data/human_texts.json")
    human_texts.rename(columns={human_texts.columns[0]: "text"}, inplace=True)
    with open("content\\hum.txt", 'w', encoding="utf8") as f:
        for item in human_texts["text"].values:
            f.write(str(item) + "\n******\n")
    # np.savetxt("content\\hum.txt", human_texts["text"].values, fmt='%d', delimiter="*****")
    st.stop()
    fprint("loaded human texts, elapsed time = " + str(st.duration))


def load_new_data_RuATD():
    # st.start()
    # generated_texts = pd.read_json("data/combined_data/generated_reduced.json")
    # fprint("loaded generated texts, elapsed time = " + str(st.duration))
    # st.reset()
    # st.start()
    # human_texts = pd.read_json("data/combined_data/human_reduced.json")
    # fprint("loaded human texts, elapsed time = " + str(st.duration))
    #
    st.reset()
    st.start()
    texts = pd.read_csv("data/combined_data/ruatd.csv")
    st.stop()
    fprint("loaded generated texts, elapsed time = " + str(st.duration))

    done = 0
    part = 0
    gen_clean = Parallel(n_jobs=cpus, verbose=60)(delayed(preprocess)(x) for x in texts["Text"])
    fprint("finished preprocessing texts")

    texts["clean"] = gen_clean
    texts.rename(columns={"Text": "text", "Class": "generated"}, inplace=True)
    texts["generated"] = texts["generated"].apply(rename_to_zeorones)
    full = texts
    with open("full_atd.pickle", mode="wb") as file:
        pickle.dump(full, file)
    # for i in range(0, generated_texts.shape[0]):
    #     current = generated_texts[0][i]
    #     current = word_tokenize(current, language="russian")
    #     cleaned = ""
    #     for word in current:
    #         if word not in stopwords:
    #             cleaned += word + " "
    #     cleaned = cleaned.strip()
    #     generated_texts.loc[i, "clean"] = cleaned
    #     part += 1
    #     if part >= generated_texts.shape[0] / 100 / 5:
    #         done += part
    #         fprint("done " + str(round(done / generated_texts.shape[0] * 100, 2)) + "%")
    #         part = 0
    # generated_texts["clean"][i] = cleaned

    # done = 0
    # part = 0
    # for i in range(0, human_texts.shape[0]):
    #     current = human_texts[0][i]
    #     current = word_tokenize(current, language="russian")
    #     cleaned = ""
    #     for word in current:
    #         if word not in stopwords:
    #             cleaned += word + " "
    #     cleaned = cleaned.strip()
    #     human_texts.loc[i, "clean"] = cleaned
    #     part += 1
    #     if part >= generated_texts.shape[0] / 100 / 5:
    #         done += part
    #         fprint("done " + str(round(done / generated_texts.shape[0] * 100, 2)) + "%")
    #         part = 0

    # gen_reduced = generated_texts[0:1000]
    # human_reduced = human_texts[0:1000]
    # gen_reduced.to_json("data/combined_data/generated_reduced.json")
    # human_reduced.to_json("data/combined_data/human_reduced.json")
    return full


import matplotlib.colors as colors

# colors_list = list(colors._colors_full_map.values())

colors_list = ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#bb3e03", "#ae2012",
               "#9b2226"]
colors_list = ["#6340bc", "#794ee6", "#090612", "#20153c", "#362367", "#4d3291"]


def draw_roc(TestY, lr_probs, model_name, id):
    fpr, tpr, treshold = roc_curve(TestY, lr_probs)
    roc_auc = auc(fpr, tpr)
    # строим график
    # plt.plot(fpr, tpr, color=colors_list[random.randint(0, len(colors_list)-1)],
    #          label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot(fpr, tpr, color=(random.randint(0, 100) / 255, random.randint(0, 100) / 255, random.randint(0, 100) / 255),
             label='ROC кривая (area = %0.2f)' % roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(model_name))  # + ", эксперимент:" + str(id) + "\n" + str(datetime.datetime.now()))
    plt.legend(loc="lower right")
    plt.savefig("images\\" + str(model_name) + " " + str(datetime.datetime.now()).replace(":", "-") + ".png")
    # plt.clf()


def regression(full, eval, X, eval_x):
    joblib_file = "regression_model.jbl"
    lr_model = LogisticRegression(max_iter=200)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []
    st.reset()
    st.start()
    max_auc_score = 0
    id = 1
    plt.clf()
    plt.figure(figsize=(7, 7))
    for train_idx, val_idx in cv.split(X[:full.shape[0]], full['generated']):
        X_train, X_val = X[:full.shape[0]][train_idx], X[:full.shape[0]][val_idx]
        y_train, y_val = full['generated'].iloc[train_idx], full['generated'].iloc[val_idx]

        # Train the model on the training data
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_val)

        # Print the classification report
        fprint(classification_report(y_val, y_pred))
        # Predict probabilities for the positive class on the validation data
        preds_val_lr = lr_model.predict_proba(X_val)[:, 1]

        draw_roc(y_val, preds_val_lr, "Logistic regression", id)
        accuracy = accuracy_score(y_val, y_pred)
        fprint("accuracy = "+str(accuracy))
        id += 1
        # Calculate ROC AUC score for the validation set
        auc_score = roc_auc_score(y_val, preds_val_lr)
        if auc_score > max_auc_score:
            max_auc_score = auc_score
            joblib.dump(lr_model, joblib_file)
            # lr_model = joblib.load(joblib_file)
        auc_scores.append(auc_score)
        fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))
    st.stop()
    plt.clf()
    fprint("logistic regression, elapsed time = " + str(st.duration))
    # Print the scores for each fold
    for i, score in enumerate(auc_scores, 1):
        fprint(f'ROC AUC for fold ' + str(i) + ': ' + str(round(score, 4)))

    fprint('Average ROC AUC:' + str(round(sum(auc_scores) / len(auc_scores), 4)))
    fprint('Standard deviation:' + str(
        round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4)))
    fprint()
    # load the best model
    lr_model = joblib.load(joblib_file)
    y_pred = lr_model.predict(eval_x)
    preds_val_lr = lr_model.predict_proba(eval_x)[:, 1]
    # Print the classification report
    fprint("Evaluation results of regression: " + str(classification_report(eval["generated"], y_pred)))
    accuracy = accuracy_score(eval["generated"], y_pred)
    fprint("accuracy = " + str(accuracy))
    draw_roc(eval["generated"], preds_val_lr, "Logistic regression", "Evaluation")
    fprint("------------------------------------------------")
    return lr_model


def XGB(full, eval, X, eval_x):
    joblib_file = "xgb_model.jbl"
    max_auc_score = 0

    xgb_model = XGBClassifier()
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    auc_scores = []
    id = 1
    plt.clf()
    # Split the data into training and validation for each fold
    for train_idx, val_idx in cv.split(X[:full.shape[0]], full['generated']):
        X_train, X_val = X[:full.shape[0]][train_idx], X[:full.shape[0]][val_idx]
        y_train, y_val = full['generated'].iloc[train_idx], full['generated'].iloc[val_idx]

        # Train the model on the training data
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)

        # Print the classification report
        fprint(classification_report(y_val, y_pred))
        # Predict probabilities for the positive class on the validation data
        preds_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
        draw_roc(y_val, preds_val_xgb, "XGB", id)
        id += 1
        # Calculate ROC AUC score for the validation set
        auc_score = roc_auc_score(y_val, preds_val_xgb)
        if auc_score > max_auc_score:
            max_auc_score = auc_score
            joblib.dump(xgb_model, joblib_file)
            # lr_model = joblib.load(joblib_file)
        auc_scores.append(auc_score)
        fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))
        accuracy = accuracy_score(y_val, y_pred)
        fprint("accuracy = " + str(accuracy))
    fprint("\nXGD\n")
    plt.clf()
    # Print the scores for each fold
    for i, score in enumerate(auc_scores, 1):
        fprint(f'ROC AUC for fold ' + str(i) + ': ' + str(round(score, 4)))

    fprint('Average ROC AUC:' + str(round(sum(auc_scores) / len(auc_scores), 4)))
    fprint('Standard deviation:' + str(
        round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4)))

    xgb_model = joblib.load(joblib_file)
    y_pred = xgb_model.predict(eval_x)
    preds_val_lr = xgb_model.predict_proba(eval_x)[:, 1]
    # Print the classification report
    fprint("Evaluation results of XGB: " + str(classification_report(eval["generated"], y_pred)))
    accuracy = accuracy_score(eval["generated"], y_pred)
    fprint("accuracy = " + str(accuracy))
    draw_roc(eval["generated"], preds_val_lr, "XGB", "Evaluation")
    fprint("------------------------------------------------")
    return xgb_model


def SGD(full, eval, x, eval_x):
    joblib_file = "sgd_model.jbl"
    max_auc_score = 0

    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, loss="modified_huber")
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    auc_scores = []
    id = 1
    plt.clf()
    # Split the data into training and validation for each fold
    for train_idx, val_idx in cv.split(x[:full.shape[0]], full['generated']):
        X_train, X_val = x[:full.shape[0]][train_idx], x[:full.shape[0]][val_idx]
        y_train, y_val = full['generated'].iloc[train_idx], full['generated'].iloc[val_idx]

        # Train the model on the training data
        sgd_model.fit(X_train, y_train)
        y_pred = sgd_model.predict(X_val)

        # Print the classification report
        fprint(classification_report(y_val, y_pred))
        # Predict probabilities for the positive class on the validation data
        preds_val_sgd = sgd_model.predict_proba(X_val)[:, 1]
        draw_roc(y_val, preds_val_sgd, "SGD", id)
        id += 1
        # Calculate ROC AUC score for the validation set
        auc_score = roc_auc_score(y_val, preds_val_sgd)
        if auc_score > max_auc_score:
            max_auc_score = auc_score
            joblib.dump(sgd_model, joblib_file)
        auc_scores.append(auc_score)
        fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))
        accuracy = accuracy_score(y_val, y_pred)
        fprint("accuracy = " + str(accuracy))
    fprint("\nSGD\n")
    plt.clf()
    # Print the scores for each fold
    for i, score in enumerate(auc_scores, 1):
        fprint(f'ROC AUC for fold ' + str(i) + ': ' + str(round(score, 4)))

    fprint('Average ROC AUC:' + str(round(sum(auc_scores) / len(auc_scores), 4)))
    fprint('Standard deviation:' + str(
        round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4)))

    sgd_model = joblib.load(joblib_file)
    y_pred = sgd_model.predict(eval_x)
    preds_val_lr = sgd_model.predict_proba(eval_x)[:, 1]
    accuracy = accuracy_score(eval["generated"], y_pred)
    fprint("accuracy = " + str(accuracy))
    # Print the classification report
    fprint("Evaluation results of SGD: " + str(classification_report(eval["generated"], y_pred)))

    draw_roc(eval["generated"], preds_val_lr, "SGD", "Evaluation")
    fprint("------------------------------------------------")
    return sgd_model


def ensemble_FM(full, eval, X):
    # Create the ensemble model
    fmf_model = FM_FTRL_sklearn(X)
    fmf_model._estimator_type = "classifier"

    lgr_model = LogisticRegression()

    ensemble = VotingClassifier(estimators=[
        ('fmf', fmf_model),
        ('lgr', lgr_model)
    ],
        weights=[0.01, 0.99],
        voting='soft'
    )

    cv = StratifiedKFold(n_splits=2, shuffle=True)
    auc_scores = []
    id = 1
    # Split the data into training and validation for each fold
    for train_idx, val_idx in cv.split(X[:full.shape[0]], full['generated']):
        X_train, X_val = X[:full.shape[0]][train_idx], X[:full.shape[0]][val_idx]
        y_train, y_val = full['generated'].iloc[train_idx], full['generated'].iloc[val_idx]

        ensemble.fit(X[:X_train.shape[0]], y_train)
        # Train the model on the training data
        preds_test = ensemble.predict_proba(X_val)[:, 1]
        fprint(preds_test)
        draw_roc(y_val, preds_test, "FM_FTRL + Logistic Regression (ensemble)", id)
        id += 1
        auc_score = roc_auc_score(y_val, preds_test)
        auc_scores.append(auc_score)
    for i, score in enumerate(auc_scores, 1):
        fprint(f'ROC AUC for fold {i}: {score:.4f}')

    fprint('Average ROC AUC:', round(sum(auc_scores) / len(auc_scores), 4))
    fprint('Standard deviation:',
           round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4))
    fprint()


def ensemble(model1, model2, full, X):
    # Create the ensemble model
    ensemble = VotingClassifier(estimators=[('model1', model1), ('model2', model2)], voting='soft')

    cv = StratifiedKFold(n_splits=2, shuffle=True)
    auc_scores = []

    # Split the data into training and validation for each fold
    for train_idx, val_idx in cv.split(X[:full.shape[0]], full['generated']):
        X_train, X_val = X[:full.shape[0]][train_idx], X[:full.shape[0]][val_idx]
        y_train, y_val = full['generated'].iloc[train_idx], full['generated'].iloc[val_idx]

        # Train the model on the training data
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_val)

        # Print the classification report
        fprint(classification_report(y_val, y_pred))

        # Predict probabilities for the positive class on the validation data
        preds_val_xgb = ensemble.predict_proba(X_val)[:, 1]

        # Calculate ROC AUC score for the validation set
        auc_score = roc_auc_score(y_val, preds_val_xgb)
        auc_scores.append(auc_score)

    # Print the scores for each fold
    for i, score in enumerate(auc_scores, 1):
        fprint(f'ROC AUC for fold {i}: {score:.4f}')

    fprint('Average ROC AUC:', round(sum(auc_scores) / len(auc_scores), 4))
    fprint('Standard deviation:',
           round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4))


def main():
    full, eval = load_from_json_256(sample=100000000000)

    # if atd:
    #     if reload:
    #         full = load_new_data_RuATD()
    #     else:
    #         full = load_data_RuATD()
    # else:
    #     if reload:
    #         full = load_new_data(per_class_cap, True)
    #     else:
    #         full = load_data()

    # full = load_new_data_RuATD()

    # full = load_new_data()

    # full = load_data()
    # full = load_data_RuATD()
    # gen = pd.DataFrame()
    # gen["text"] = None
    # gen["generated"] = None
    # gen["clean"] = None

    # hum = pd.DataFrame()
    # hum["text"] = None
    # hum["generated"] = None
    # hum["clean"] = None
    # max_items = 120000
    # for i in range(0, full.shape[0]):
    #     if full["generated"][i] == 0 and hum.shape[0] < max_items:
    #         hum.loc[hum.shape[0]] = full.iloc[i]
    #         #hum = pd.concat([hum, full.loc[i]])
    #     if full["generated"][i] == 1 and gen.shape[0] < max_items:
    #         gen.loc[gen.shape[0]] = full.iloc[i]
    #         #gen = pd.concat([gen, full.loc[i]])

    # negative_count = 0
    # positive_count = 0
    # msk_negative = full['generated'] == 0
    #
    # fprint("neg")
    # msk_positive = full['generated'] == 1
    # fprint("pos")
    # for item in msk_negative:
    #     if item:
    #         negative_count += 1
    # for item in msk_positive:
    #     if item:
    #         positive_count += 1
    # n_ = min(100000, min(negative_count, positive_count))
    # df_train_undersample = pd.concat(
    #     [full[msk_negative].sample(n=n_, random_state=888), full[msk_positive].sample(n=n_, random_state=888)])
    #
    # full = df_train_undersample

    fprint(full['generated'].value_counts())
    df = full["clean"]

    vectorizer = TfidfVectorizer(stop_words=None, max_features=500000,
                                 tokenizer=text_splitter,
                                 token_pattern=None)

    X = None
    eval_x = None
    transformer = TfidfTransformer()
    if atd:
        if re_vectorize:
            st.reset()
            st.start()
            vec = vectorizer.fit_transform(df)
            X = transformer.fit_transform(vec)
            st.stop()
            fprint("vectorized, elapsed time = " + str(st.duration))
            with open('X_atd.bin', 'wb') as fin:
                pickle.dump(X, fin)
            with open('vect_atd.bin', 'wb') as fin:
                joblib.dump(vectorizer, fin)
            with open('trans_atd.bin', 'wb') as fin:
                joblib.dump(transformer, fin)


        else:
            with open('vect_atd.bin', 'rb') as fin:
                vectorizer = joblib.load(fin)
            with open('trans_atd.bin', 'rb') as fin:
                transformer = joblib.load(fin)
            with open('X_atd.bin', 'rb') as fin:
                X = joblib.load(fin)


    else:
        if re_vectorize:
            st.reset()
            st.start()
            vec = vectorizer.fit_transform(df)
            X = transformer.fit_transform(vec)
            eval_vec = vectorizer.transform(eval["clean"])
            eval_x = transformer.transform(eval_vec)
            st.stop()
            fprint("vectorized, elapsed time = " + str(st.duration))
            # fprint("vectorized, elapsed time = " + str(st.duration))
            with open('X.bin', 'wb') as fin:
                joblib.dump(X, fin)
            with open('vect.bin', 'wb') as fin:
                joblib.dump(vectorizer, fin)
            with open('trans.bin', 'wb') as fin:
                joblib.dump(transformer, fin)
        else:
            with open('vect.bin', 'rb') as fin:
                vectorizer = joblib.load(fin)
            with open('trans.bin', 'rb') as fin:
                transformer = joblib.load(fin)
            with open('X.bin', 'rb') as fin:
                X = joblib.load(fin)

    if only_vectorize:
        exit(0)

    # ensemble_FM()
    lr_model = regression(full, eval, X, eval_x)
    sgd_model = SGD(full, eval, X, eval_x)
    xgb_model = XGB(full, eval, X, eval_x)



if __name__ == '__main__':
    # save_as_txt()
    main()
# regression + XGBClassifier


# ensemble_FM()
# ensemble(lr_model, sgd_model)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
