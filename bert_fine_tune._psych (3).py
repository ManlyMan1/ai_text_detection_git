#https://github.com/PradipNichite/Youtube-Tutorials/blob/main/FineTune_BERT_Model_Youtube.ipynb
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification

label3=False
data = pd.read_excel('D:\\GoogleDrive\\RNF_2024\\TurkLang2024\\texts_256.xlsx')
#data=data.head(500)
data=data[data['type'].isin(['k','d'])]
if label3:
    data=data[data['type'].isin(['k','d','i'])]
else:
     data=data[data['type'].isin(['k','d'])]   
data['type'].loc[data['type'] == 'k'] = 0
data['type'].loc[data['type'] == 'd'] = 1
if label3:
    data['type'].loc[data['type'] == 'i'] = 2


from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')
if label3:
    model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny',num_labels=3,id2label={0: 'k', 1: 'd', 2: 'i'})
else:
    model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny',num_labels=2,id2label={0: 'k', 1: 'd'},label2id={'k':0, 'd':1})
#model = model.to('cuda')


X = list(data["text"])
y = list(data["type"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)



# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    if label3:
        recall = recall_score(y_true=labels, y_pred=pred, average='macro')
        precision = precision_score(y_true=labels, y_pred=pred, average='macro')
        f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    else:

        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

    res="accuracy="+ str(accuracy)+ "; precision="+str(precision)+ "; recall="+ str(recall)+ "; f1="+ str(f1)
    print(res)
    with open("res_s2_256.txt", "a") as myfile:
        myfile.write(res+'\n')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



args = TrainingArguments(
    output_dir="output",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy ="epoch",
    logging_dir="logs",
    logging_strategy="epoch",

)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics)

trainer.train()
trainer.evaluate()

np.set_printoptions(suppress=True)

trainer.save_model('CustomModelPsych_s2_256')

