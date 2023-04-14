import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import StepLR

from transformers import BertTokenizer, BertModel

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device type: {device}\n')

# Constants ==================================================================
SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 32
MAX_LENGTH = 32

KEYWORDS_PATH = 'keywords.txt'
TRAIN_CSV_PATH = 'train.csv'
TEST_CSV_PATH = 'test.csv'
SUBMISSION_CSV_PATH = 'submission.csv'


# Classes ====================================================================
class DictDataset(Dataset):
    def __init__(self, list_of_pairs, labels=None):
        self.data = dict(list_of_pairs)
        self.keys = [pair[0] for pair in list_of_pairs]
        self.labels = labels
        self.len = len(list_of_pairs[0][1])

    def __getitem__(self, idx):
        if self.labels is not None:
            return {key: self.data[key][idx] for key in self.keys}, self.labels[idx]
        else:
            return {key: self.data[key][idx] for key in self.keys}

    def __len__(self):
        return self.len


class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, out_features),
        )

    def forward(self, X):
        return self.model(X)


class MyModel(nn.Module):
    def __init__(self, n_additional_features, out_features):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            output_attentions=False,
            output_hidden_states=True,
        )

        self.classifier = Classifier(768 * 4 + n_additional_features, out_features)

    def forward(self, **X):
        token_ids = X['token_ids']
        attention_mask = X['attention_mask']
        additional_features = X['additional_features']

        bert_output = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        all_hidden_states = torch.stack(bert_output[2])
        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )
        concatenate_pooling = concatenate_pooling[:, 0]
        combinded_features = torch.cat([concatenate_pooling, additional_features], dim=1)
        output = self.classifier(combinded_features)
        return output

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True


# Functions ==================================================================
def map_labels(y):
    mapper = dict()
    inverse_mapper = dict()
    y_numeric = []

    i = 0
    for label in y:
        if label not in mapper:
            mapper[label] = i
            inverse_mapper[i] = label
            i += 1
        y_numeric.append(mapper[label])
    return torch.Tensor(y_numeric), inverse_mapper


def preprocess(X):
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

    token_ids = []
    attention_masks = []

    for sample in tqdm(X, desc='Tokenization'):
        encoding_dict = tokenizer.encode_plus(
            sample,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        token_ids.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])

    token_ids = torch.cat(token_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return token_ids, attention_masks


def get_additional_features(X, keywords):
    """
    :param X: iterable collection with texts
    :param keywords: iterable collection with groups of keywords.
    Example: [['json'], ['pandas', 'pd], ['matplotlib', 'pyplot', 'plot', 'plt']]
    Presence of any word of group is treated as the same feature.
    :return: torch.Tensor with size [len(X), len(keywords)]
    """
    features = []
    for i, sentence in enumerate(X):
        containing_words = []
        sentence = sentence.lower()
        sentence = re.sub(r'[!?\'"()#.:]', ' ', sentence)
        sentence = sentence.split()
        for j, keywords_group in enumerate(keywords):
            for word in keywords_group:
                if word in sentence:
                    containing_words.append(j)
        features.append(containing_words)

    additional_features = torch.zeros(size=(len(X), len(keywords)), dtype=torch.long)
    for i, idxs in enumerate(features):
        for j in idxs:
            additional_features[i][j] = 1

    return additional_features


def train_epoch(model, optimizer, criterion, train_loader):
    loss_sum, scores_sum = 0, 0
    model.train()
    for X, y in tqdm(train_loader, desc='Train', total=len(train_loader), leave=False):
        X = {k: X[k].to(device) for k in X}
        y = y.to(device)
        output = model(**X)
        loss = criterion(output, y)
        loss_sum += loss.item()
        scores_sum += precision_score(y.cpu(), torch.argmax(output, dim=1).cpu(), average='micro')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = loss_sum / len(train_loader)
    score = scores_sum / len(train_loader)
    return loss, score


def validate(model, criterion, validation_loader):
    loss_sum, scores_sum = 0, 0
    model.eval()
    for X, y in tqdm(validation_loader, desc='Validation', total=len(validation_loader), leave=False):
        X = {k: X[k].to(device) for k in X}
        y = y.to(device)
        with torch.no_grad():
            output = model(**X)
            loss = criterion(output, y)
        loss_sum += loss.item()
        scores_sum += precision_score(y.cpu(), torch.argmax(output, dim=1).cpu(), average='micro')
    loss = loss_sum / len(validation_loader)
    score = scores_sum / len(validation_loader)
    return loss, score


def train_loop(model, optimizer, criterion, epochs, train_loader, validation_loader=None, scheduler=None):
    for epoch in range(1, epochs + 1):
        # ================== train ==================
        _, _ = train_epoch(model, optimizer, criterion, train_loader)

        # ================ validation ================
        if validation_loader is not None:
            val_loss, val_score = validate(model, criterion, validation_loader)
            print(f"epoch {epoch:<2} |loss {round(val_loss, 4):<6} |precision {round(val_score, 4):<6}")

        # ============== scheduler step ==============
        if scheduler is not None:
            scheduler.step()


def test(model, test_loader):
    model.eval()
    predictions = []
    for X in tqdm(test_loader, desc='Test', total=len(test_loader)):
        X = {k: X[k].to(device) for k in X}
        with torch.no_grad():
            test_output = model(**X)
        predictions.extend(test_output.cpu().numpy().argmax(axis=1).tolist())
    return predictions


def main():
    # Read data ==================================================================
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    X_train = train_df.title.values
    y_train, inverse_mapper = map_labels(train_df.lib.values)
    n_classes = len(inverse_mapper)
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_test = test_df.title.values
    test_ids = test_df.id.values

    # Preprocess =================================================================
    print('preprocessing started')
    train_token_ids, train_attention_masks = preprocess(X_train)
    test_token_ids, test_attention_masks = preprocess(X_test)

    with open(KEYWORDS_PATH) as f:
        keywords = [w.strip().split() for w in f.readlines()]
    n_additional_features = len(keywords)

    train_additional_features = get_additional_features(X_train, keywords)
    test_additional_features = get_additional_features(X_test, keywords)
    print('preprocessing finished\n')

    # Prepare dataloaders ========================================================
    train_idxs, val_idxs = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        shuffle=True,
        random_state=SEED,
        stratify=y_train)

    train_set = DictDataset(
        [
            ('token_ids', train_token_ids[train_idxs]),
            ('attention_mask', train_attention_masks[train_idxs].long()),
            ('additional_features', train_additional_features[train_idxs])
        ],
        labels=y_train[train_idxs].long()
    )

    val_set = DictDataset(
        [
            ('token_ids', train_token_ids[val_idxs]),
            ('attention_mask', train_attention_masks[val_idxs].long()),
            ('additional_features', train_additional_features[val_idxs])
        ],
        labels=y_train[val_idxs].long()
    )

    test_set = DictDataset(
        [
            ('token_ids', test_token_ids),
            ('attention_mask', test_attention_masks.long()),
            ('additional_features', test_additional_features)
        ]
    )

    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=BATCH_SIZE)
    validation_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Train classifier ===========================================================
    print('training classifier (3e-4)\n')
    model = MyModel(n_additional_features=n_additional_features, out_features=n_classes).to(device)
    model.freeze_bert()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_loop(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        epochs=10,
        train_loader=train_loader,
        validation_loader=validation_loader,
        scheduler=StepLR(optimizer, step_size=1, gamma=0.9)
    )

    # Fine-tune ==================================================================
    print('\nfine-tuning (2e-5)\n')
    model.unfreeze_bert()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-08)
    train_loop(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        epochs=2,
        train_loader=train_loader,
        validation_loader=validation_loader,
        scheduler=StepLR(optimizer, step_size=1, gamma=0.9)
    )

    # Fine-tune on validation set ================================================
    print('\nfine-tuning on validation set (1e-5)\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-08)
    train_loop(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        epochs=2,
        train_loader=validation_loader,
        scheduler=StepLR(optimizer, step_size=1, gamma=0.9)
    )

    # Submission =================================================================
    print('\nsubmission\n')
    predictions = test(model, test_loader)
    string_predictions = [inverse_mapper[pred] for pred in predictions]
    submission_df = pd.DataFrame(list(zip(test_ids, string_predictions)), columns=['id', 'lib'])
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)


if __name__ == '__main__':
    main()
