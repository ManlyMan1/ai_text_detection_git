# This is a sample Python script.
import multiprocessing
import pickle
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
from sklearn.metrics import roc_auc_score
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

st = stopwatch.Stopwatch()
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default

exclude = set(string.punctuation)
exclude.update(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])


def regression(X, Y):
    joblib_file = "regression_model.jbl"
    lr_model = joblib.load(joblib_file)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    auc_scores = []
    st.reset()
    st.start()
    max_auc_score = 0

    Y_pred = lr_model.predict(X)

    # Print the classification report
    print(classification_report(Y, Y_pred))
    # Predict probabilities for the positive class on the validation data
    preds_val_lr = lr_model.predict_proba(X)[:, 1]
    print(preds_val_lr)
    # Calculate ROC AUC score for the validation set
    auc_score = roc_auc_score(Y, preds_val_lr)
    print(f'ROC AUC for fold : {auc_score:.4f}')


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
def text_splitter(x):
    return re.findall(r'[a-zA-Zа-яА-Я]+', x)

stopwords = list()

with open("stopwords-ru.txt", encoding="utf8") as file:
    lines = file.readlines()
    print()
    result = ""
    for line in lines:
        result += '"' + line.strip() + '",\n'
        stopwords.append(line.strip())
    print()


def rename_to_zeorones(x):
    if x == "H":
        return 0
    return 1


def load_eval_data():
    # st.start()
    # generated_texts = pd.read_json("data/combined_data/generated_reduced.json")
    # print("loaded generated texts, elapsed time = " + str(st.duration))
    # st.reset()
    # st.start()
    # human_texts = pd.read_json("data/combined_data/human_reduced.json")
    # print("loaded human texts, elapsed time = " + str(st.duration))
    #
    st.reset()
    st.start()
    texts = pd.read_excel("data/psych_texts/texts_256.xlsx")
    st.stop()
    print("loaded human texts, elapsed time = " + str(st.duration))

    done = 0
    part = 0
    # gen_clean = Parallel(n_jobs=cpus, verbose=60)(delayed(preprocess)(x) for x in texts[texts.columns[0]])
    # print("finished preprocessing texts")

    # texts["clean"] = gen_clean
    texts["clean"] = texts[0]
    texts.rename(columns={texts.columns[0]: 'text'}, inplace=True)
    texts.rename(columns={"text": "text", "type": "generated"}, inplace=True)
    texts["generated"] = texts["generated"].apply(rename_to_zeorones)
    full = texts
    # with open("full_atd.pickle", mode="wb") as file:
    #     pickle.dump(full, file)
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
    #         print("done " + str(round(done / generated_texts.shape[0] * 100, 2)) + "%")
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
    #         print("done " + str(round(done / generated_texts.shape[0] * 100, 2)) + "%")
    #         part = 0

    # gen_reduced = generated_texts[0:1000]
    # human_reduced = human_texts[0:1000]
    # gen_reduced.to_json("data/combined_data/generated_reduced.json")
    # human_reduced.to_json("data/combined_data/human_reduced.json")
    st.reset()
    st.start()
    with open("full_atd.pickle", mode="rb") as file:
        gen = pickle.load(file)
    st.stop()
    print("loaded texts, elapsed time = " + str(st.duration))
    gen = gen.drop('Id', axis=1)
    gen = gen.drop(gen[gen.generated == 0].index)
    print()
    full = pd.concat([full, gen], ignore_index=True)
    return full

def main():
    full = load_eval_data()
    with open('vect.bin', 'rb') as fin:
        vectorizer = joblib.load(fin)
    with open('trans.bin', 'rb') as fin:
        transformer = joblib.load(fin)
    # transformer = TfidfTransformer()
    # vectorizer = TfidfVectorizer(decode_error="replace", vocabulary=vocab)

    # with open('X.bin', 'rb') as fin:
    #     X = pickle.load(fin)
    df = full["clean"]
    X = transformer.transform(vectorizer.transform(df))
    # X = transformer.fit_transform(vectorizer.fit_transform(df))
    Y = full["generated"]
    print()

    regression(X, Y)
