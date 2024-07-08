import datetime
import random

import joblib
import numpy as np
import pandas as pd
import stopwatch
import torch
import torch.nn as nn
import sklearn
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from omnixai.data.text import Text
from omnixai.explainers.nlp import ShapText
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer, InputData, DataLoader
from omnixai.explainers.nlp.specific.ig import IntegratedGradientText
import nltk
from omnixai.visualization.dashboard import Dashboard
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from evaluation import load_eval_data
# nltk.download()
from main import load_new_data, crop_data, load_data, rename_to_zeorones
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from text_prepare import process_files_full

log_file_name = ("logs\\log_CNN_" + str(datetime.datetime.now()) + ".txt").replace(":", "-")
plt.figure(figsize=(7, 7))
st = stopwatch.Stopwatch()
with open(log_file_name, mode="a", encoding="utf8") as file:
    file.write("\n\n\n\n\t\tNew process --------------------------------------------------\n\n\n\n")

torch.cuda.empty_cache()
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


class CustomDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class BertClassifier:

    def __init__(self, model_path, tokenizer_path, n_classes=2, epochs=1, model_save_path='/content/bert.pt'):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fprint(self.device)
        self.model_save_path = model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=2, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=2, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for data in self.train_loader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            fprint(f'Epoch {epoch + 1}/{self.epochs}')

            train_acc, train_loss = self.fit()
            fprint(f'Train loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = self.eval()
            fprint(f'Val loss {val_loss} accuracy {val_acc}')
            fprint('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        self.model = torch.load(self.model_save_path)

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction


sigmoid = nn.Sigmoid()


class TextModel(nn.Module):

    def __init__(self, num_embeddings, num_classes, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = kwargs.get("embedding_size", 50)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.embedding.weight.data.normal_(mean=0.0, std=0.01)

        hidden_size = kwargs.get("hidden_size", 100)
        kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes]

        self.activation = nn.ReLU()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.embedding_size, hidden_size, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(len(kernel_sizes) * hidden_size, num_classes)

    def forward(self, inputs, masks):
        embeddings = self.embedding(inputs)
        x = embeddings * masks.unsqueeze(dim=-1)
        x = x.permute(0, 2, 1)
        x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
        outputs = self.output_layer(self.dropout(torch.cat(x, dim=1)))

        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(dim=1)
        return outputs

    def predict(self, inputs, masks):
        embeddings = self.embedding(inputs)
        x = embeddings * masks.unsqueeze(dim=-1)
        x = x.permute(0, 2, 1)
        x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
        outputs = self.output_layer(self.dropout(torch.cat(x, dim=1)))

        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(dim=1)
        return sigmoid(outputs)


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


def test_accuracy(model, data_x, data_y):
    model.eval()
    data = transform.transform(data_x)
    data_loader = DataLoader(
        dataset=InputData(data, [0] * len(data), max_length),
        batch_size=32,
        collate_fn=InputData.collate_func,
        shuffle=False
    )
    outputs = []
    for inputs in data_loader:
        value, mask, target = inputs
        y = model(value.to(device), mask.to(device))
        outputs.append(y.detach().cpu().numpy())
    # fprint(outputs)
    outputs = np.concatenate(outputs, axis=0)
    predictions = np.argmax(outputs, axis=1)
    # fprint('Accuracy: {}'.format(sklearn.metrics.f1_score(data_y, predictions, average='binary')))
    accuracy = sklearn.metrics.f1_score(data_y, predictions, average='macro')
    precision, recall, f1score = precision_recall_fscore_support(data_y, predictions, average='macro')[:3]
    fprint(f'precision: {precision}, recall: {recall}, f1score: {f1score}, accuracy: {accuracy}')


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
    plt.savefig("images\\" + str(model_name) + " " + str(id) + str(datetime.datetime.now()).replace(":", "-") + ".png")
    # plt.clf()


def predict_proba(model, data_x, data_y):
    model.eval()
    data = transform.transform(data_x)
    data_loader = DataLoader(
        dataset=InputData(data, [0] * len(data), max_length),
        batch_size=32,
        collate_fn=InputData.collate_func,
        shuffle=False
    )
    outputs = []
    probs = []
    # with torch.no_grad():
    for inputs in data_loader:
        value, mask, target = inputs
        y = model.predict(value.to(device), mask.to(device))
        outputs.append(y.detach().cpu().numpy())
        probs.append(y)
    # fprint(outputs)
    outputs_ = np.concatenate(outputs, axis=0)
    probs = torch.cat(probs, axis=0)
    probabilities = torch.nn.functional.softmax(probs, dim=1)
    predictions = np.argmax(outputs_, axis=1)
    # fprint('Accuracy: {}'.format(sklearn.metrics.f1_score(data_y, predictions, average='binary')))
    fprint('Accuracy: {}'.format(sklearn.metrics.f1_score(data_y, predictions, average='macro')))
    return probabilities.tolist()


def preprocess(X: Text):
    samples = transform.transform(X)
    max_len = 0
    for i in range(len(samples)):
        max_len = max(max_len, len(samples[i]))
    max_len = min(max_len, max_length)
    inputs = np.zeros((len(samples), max_len), dtype=int)
    masks = np.zeros((len(samples), max_len), dtype=np.float32)
    for i in range(len(samples)):
        x = samples[i][:max_len]
        inputs[i, :len(x)] = x
        masks[i, :len(x)] = 1
    return inputs, masks


max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

# process_files_full()
# exit(0)

# full_, eval_ = load_from_json_256()


train_data, eval_ = load_from_json_256(sample=140000)
# train_data = train_data.sample(frac=1).reset_index()

train_data = train_data[train_data['generated'].isin([0, 1])]
plt.clf()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []
st.reset()
st.start()
max_auc_score = 0
id = 1
class_names = ["human", "neuro"]
for i in range(0, 10):
    train_data = train_data.sample(frac=1)
    n = int(0.7 * len(train_data))
    fprint(train_data.head(10))
    x_train = Text(train_data["clean"].values[:n])
    y_train = train_data["generated"].values[:n].astype(int)
    x_val = Text(train_data["clean"].values[n:])
    y_val = train_data["generated"].values[n:].astype(int)
    class_names = ["Конструктивные", "Деструктивные"]
    transform = Word2Id().fit(x_train)
    joblib.dump(transform, "openxai_transformer.tranformer")
    model = TextModel(num_embeddings=transform.vocab_size, num_classes=len(class_names)).to(device)
    trainer = Trainer(
        optimizer_class=torch.optim.AdamW,
        learning_rate=1e-3,
        batch_size=16,
        num_epochs=10,
    )
    trainer.train(
        model=model,
        loss_func=nn.CrossEntropyLoss(),
        train_x=transform.transform(x_train),
        train_y=y_train,
        padding=True,
        max_length=max_length,
        verbose=True
    )

    joblib.dump(model, "openxai_model.model")

    id += 1

    probabilities = predict_proba(model, x_val, y_val)
    probabilities_ = list()
    for item in probabilities:
        probabilities_.append(item[1])
    test_accuracy(model, x_val, y_val)

    auc_score = roc_auc_score(y_val, probabilities_)
    draw_roc(y_val, probabilities_, "CNN", id)
    if auc_score > max_auc_score:
        max_auc_score = auc_score
        joblib.dump(model, "openxai_model_best.model")
        joblib.dump(transform, "openxai_transform_best.model")

        # lr_model = joblib.load(joblib_file)
    auc_scores.append(auc_score)
    fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))

#######################################################################################################  uncomment
# n = int(0.8 * len(train_data))
# fprint(train_data.head(10))
# x_train = Text(train_data["clean"].values[:n])
# y_train = train_data["generated"].values[:n].astype(int)
# x_test = Text(train_data["clean"].values[n:])
# y_test = train_data["generated"].values[n:].astype(int)
# class_names = ["Конструктивные", "Деструктивные"]

# fprint(x_train)
#######################################################################################################

#######################################################################################################  uncomment
# transform = Word2Id().fit(x_train)
# joblib.dump(transform, "openxai_transformer.tranformer")
model = joblib.load("openxai_model_best.model")
#joblib.dump(transform, "openxai_transform_best.model")
transform = joblib.load("openxai_transform_best.model")
#######################################################################################################


#######################################################################################################  uncomment
# model = TextModel(num_embeddings=transform.vocab_size, num_classes=len(class_names)).to(device)
# trainer = Trainer(
#     optimizer_class=torch.optim.AdamW,
#     learning_rate=1e-3,
#     batch_size=16,
#     num_epochs=10,
# )
# trainer.train(
#     model=model,
#     loss_func=nn.CrossEntropyLoss(),
#     train_x=transform.transform(x_train),
#     train_y=y_train,
#     padding=True,
#     max_length=max_length,
#     verbose=True
# )
#
# joblib.dump(model, "openxai_model.model")
#######################################################################################################
#model = joblib.load("openxai_model.model")

# full = load_eval_data()
x_test = Text(eval_["text"].values)
y_test = eval_["generated"].values.astype(int)
plt.clf()
probabilities = predict_proba(model, x_test, y_test)
probabilities_ = list()
for item in probabilities:
    probabilities_.append(item[1])

# x_test = Text(train_data["clean"].values[n:])
# y_test = train_data["generated"].values[n:].astype(int)
test_accuracy(model, x_test, y_test)
auc_score = roc_auc_score(y_test, probabilities_)
draw_roc(y_test, probabilities_, "CNN", "Evaluation")
#
# test_accuracy(model, x_test, y_test)
# test_accuracy(model, x_train, y_train)
#
# explainer = IntegratedGradientText(
#     model=model,
#     embedding_layer=model.embedding,
#     preprocess_function=preprocess,
#     id2token=transform.id_to_word
# )
#
# x = x_test[0]
# explanations = explainer.explain(x)
# # plt1=explanations.ipython_plot(class_names=class_names)
# # plt1.savefig('fig1.jpg')
# plt2 = explanations.plot(class_names=class_names)
# plt2.savefig('fig2.jpg')
