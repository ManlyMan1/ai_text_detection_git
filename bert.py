import datetime
import random

import joblib
import numpy as np
import stopwatch
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import pandas as pd

from bert_classifier import BertClassifier
from main import load_data, rename_to_zeorones
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc, accuracy_score, \
    classification_report

st = stopwatch.Stopwatch()
torch.cuda.empty_cache()


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
    fprint(_eval.shape)
    fprint(_eval.head(10))
    fprint("shuffled data, elapsed time = " + str(st.duration))
    return _full, _eval


log_file_name = ("logs\\log_BERT_" + str(datetime.datetime.now()) + ".txt").replace(":", "-")
with open(log_file_name, mode="a", encoding="utf8") as file:
    file.write("\n\n\n\n\t\tNew process --------------------------------------------------\n\n\n\n")


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


plt.figure(figsize=(7, 7))


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
    plt.title(str(model_name) + "\n" + str(datetime.datetime.now()))
    plt.legend(loc="lower right")
    plt.savefig("images\\" + str(model_name) + " " + str(datetime.datetime.now()).replace(":", "-") + ".png")
    # plt.clf()


def main():
    # train_data = pd.read_csv('/content/train.csv')
    # valid_data = pd.read_csv('/content/valid.csv')
    # test_data = pd.read_csv('/content/test.csv')
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny2',
        tokenizer_path='cointegrated/rubert-tiny2',
        n_classes=2,
        epochs=1,
        model_save_path='content/bert.pt'
    )
    train_data, eval = load_from_json_256(sample=130000)
    train_data = train_data.sample(frac=1).reset_index()
    train_data = train_data.drop('level_0', axis=1)
    train_data = train_data.drop('index', axis=1)
    train_data = train_data.drop('text', axis=1)
    train_data.rename(columns={'clean': "text"}, inplace=True)
    train_data.rename(columns={'generated': "label"}, inplace=True)
    plt.clf()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []
    st.reset()
    st.start()
    max_auc_score = 0
    id = 1
    for train_idx, val_idx in cv.split(train_data["text"], train_data['label']):
        X_train, X_val = train_data["text"][train_idx], train_data["text"][val_idx]
        y_train, y_val = train_data['label'].iloc[train_idx], train_data['label'].iloc[val_idx]

        classifier.preparation(
            X_train=list(X_train),
            y_train=list(y_train),
            X_valid=list(X_val),
            y_valid=list(y_val)
        )

        classifier.train()
        # Train the model on the training data
        #lr_model.fit(X_train, y_train)
        #y_pred = lr_model.predict(X_val)
        y_pred = [classifier.predict(t) for t in X_val]
        # Print the classification report
        fprint(classification_report(y_val, y_pred))
        # Predict probabilities for the positive class on the validation data
        probabilities = classifier.predict_proba(X_val)
        probabilities_ = list()
        for item in probabilities:
            probabilities_.append(item[1])
        accuracy = accuracy_score(y_val, y_pred)
        fprint("accuracy: " + str(accuracy))
        draw_roc(y_val, probabilities_, "RuBERT", id)

        id += 1
        # Calculate ROC AUC score for the validation set
        auc_score = roc_auc_score(y_val, probabilities_)
        if auc_score > max_auc_score:
            max_auc_score = auc_score
            torch.save(classifier.model, "content/bert_best.pt")
            # lr_model = joblib.load(joblib_file)
        auc_scores.append(auc_score)
        fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))
    st.stop()
    fprint("BERT, elapsed time = " + str(st.duration))
    # Print the scores for each fold
    for i, score in enumerate(auc_scores, 1):
        fprint(f'ROC AUC for fold ' + str(i) + ': ' + str(round(score, 4)))

    fprint('Average ROC AUC:' + str(round(sum(auc_scores) / len(auc_scores), 4)))
    fprint('Standard deviation:' + str(
        round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4)))
    fprint()
    # load the best model
    classifier.model = torch.load("content/bert_best.pt")

    #######################################################################################################  uncomment
    # n = int(0.8 * len(train_data))
    # fprint(train_data.head(10))
    # x_train = train_data["text"].values[:n]
    # y_train = train_data["label"].values[:n].astype(int)
    # x_test = train_data["text"].values[n:]
    # y_test = train_data["label"].values[n:].astype(int)

    # texts = pd.read_excel("data/psych_texts/texts_256.xlsx")
    #
    # done = 0
    # part = 0
    # # gen_clean = Parallel(n_jobs=cpus, verbose=60)(delayed(preprocess)(x) for x in texts[texts.columns[0]])
    # # fprint("finished preprocessing texts")
    #
    # # texts["clean"] = gen_clean
    # texts.rename(columns={texts.columns[0]: 'text'}, inplace=True)
    # texts.rename(columns={"text": "text", "type": "label"}, inplace=True)
    # texts["label"] = texts["label"].apply(rename_to_zeorones)


#######################################################################################################  uncomment
    # classifier.load_preparation(
    #     X_train=list(x_train),
    #     y_train=list(y_train),
    #     X_valid=list(x_test),
    #     y_valid=list(y_test)
    # )
    #
    # classifier.load()


    texts_ = list(eval['clean'])
    labels = list(eval['generated'])
    plt.clf()
    predictions = [classifier.predict(t) for t in texts_]

    precision, recall, f1score = precision_recall_fscore_support(labels, predictions, average='macro')[:3]
    probabilities = classifier.predict_proba(texts_)
    probabilities_ = list()
    for item in probabilities:
        probabilities_.append(item[1])
    auc_score = roc_auc_score(eval["generated"], probabilities_)
    draw_roc(eval["generated"], probabilities_, "RuBERT", "Evaluation")
    accuracy = accuracy_score(labels, predictions)
    fprint(f'precision: {precision}, recall: {recall}, f1score: {f1score}, accuracy: {accuracy}, auc score: {auc_score}')


main()
