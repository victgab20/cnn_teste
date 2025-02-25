
import pandas as pd
import torch
import random
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset
import gensim

#import torchtext.vocab
import tqdm
import torch.nn as nn
import torch.nn.functional as F

import pickle
from nltk import tokenize
import torch.optim as optim

from sklearn.metrics import classification_report
import torchtext
 # Substitua pelo nome da classe do seu modelo
import torch

path = r"C:\Users\victo\Downloads\corpus\labeled"


# path = "/content/drive/MyDrive/dataset/UTLCorpus/"

# df = pd.read_pickle(path+'train_filmes.pkl')
# rus = RandomUnderSampler(random_state=2021)
# df = df.dropna()
# text = df[['text']]
# label = df[['helpfulness']]
# features = []

# text, label = rus.fit_resample(text, label)

# for t, l in zip(text, label):
#   d = {}
#   d['text'] = t[0]
#   d['helpfulness'] = l[0]
#   features.append(d)

# bdf = pd.DataFrame(df, columns=['text', 'helpfulness'])

# bdf.to_csv(path+'balanced_train_filmes.csv', index=False)




class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)  # Lê o CSV com pandas
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Recupera o texto da linha correspondente
        text = self.data.iloc[idx]['text']
        
        # Verifica se o texto é um float (ou NaN) e trata
        if isinstance(text, float) or pd.isna(text):  # Se for float ou NaN, converta para string vazia
            text = ""  # Ou use str(text) se preferir
        else:
            text = str(text)  # Certifique-se de que é uma string

        # Tokeniza o texto
        tokenized_text = self.tokenizer(text)

        # Recupera a ajuda (ou outra coluna que você precisa)
        label = torch.tensor(self.data.iloc[idx]['helpfulness'], dtype=torch.float)

        return tokenized_text, label


word_to_index = {"<pad>": 0}  # Inicializa o dicionário com o token de padding

def tokenizer(text):
    # Tokeniza o texto usando o NLTK
    tokenized = tokenize.word_tokenize(text, language='portuguese')
    
    # Atualiza o dicionário word_to_index com os novos tokens
    for token in tokenized:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)  # Atribui um índice incremental

    # Se o número de tokens for menor que 5, preenche com <pad>
    if len(tokenized) < 5:
        tokenized += ['<pad>'] * (5 - len(tokenized))
    
    return tokenized


def collate_batch(batch):
    texts = [item[0] for item in batch]  # Extrair textos do batch
    labels = [item[1] for item in batch]  # Extrair rótulos do batch

    # Converter tokens para índices usando o vocabulário
    texts = [[word_to_index.get(token, word_to_index["<pad>"]) for token in text] for text in texts]
    
    # Converter listas de índices para tensores
    texts = [torch.tensor(text, dtype=torch.long) for text in texts]

    # Padronizar o tamanho dos textos
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)

    return texts, labels

def process_splits(path, BATCH_SIZE, device, glove_path):
    nltk.download('punkt')  # Certifique-se de que o recurso 'punkt' está disponível
    # Usando o tokenizador para português
    tokenizer = lambda text: word_tokenize(text, language='portuguese')

    # Criando os datasets
    train_dataset = TextDataset(path + '/balanced_train_apps.csv', tokenizer)
    valid_dataset = TextDataset(path + '/balanced_dev_apps.csv', tokenizer)
    test_dataset = TextDataset(path + '/balanced_test_apps.csv', tokenizer)

    # Carregando os embeddings do GloVe (arquivo local)
    print("Carregando os embeddings do GloVe...")
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False)

    # Criando os DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    return train_loader, valid_loader, test_loader, embeddings

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
glove_path = f"{path}/glove_s300.txt"  # Defina o caminho correto

train_iterator, valid_iterator, test_iterator, embeddings = process_splits(path, BATCH_SIZE, device, glove_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# Carregar o modelo completo

# Colocar o modelo em modo de avaliação


train_iterator, valid_iterator, test_iterator, embeddings = process_splits(path, BATCH_SIZE, device, glove_path)
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, vectors):

        super().__init__()

        # 1 camada - embeddings
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

        # 3 camadas de convolução
        self.conv_0 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[2], embedding_dim))

        # A saída dos filtros alimenta uma mlp
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 32)

        self.fc1 = nn.Linear(32, output_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        #text = [batch size, sent len]

        embedded = self.embedding(text)

        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        out = F.relu(self.fc(cat))

        return self.fc1(out)
    
INPUT_DIM = len(embeddings.key_to_index)  # Tamanho do vocabulário
EMBEDDING_DIM = 300  # Dimensão dos vetores
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.7

# Adicionando token de padding
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
VECTORS = torch.tensor(embeddings.vectors, dtype=torch.float)

extra_vector = torch.randn(1, EMBEDDING_DIM)  # Vetor aleatório
VECTORS = torch.cat([VECTORS, extra_vector], dim=0)



model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, VECTORS)

# Carregar os pesos do modelo salvo
model.load_state_dict(torch.load("cnn.pt", map_location=device))
model.to(device)

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

def binary_accuracy(preds, y):

    #leva o valor para o inteiro mais próxio
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_predictions = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(iterator, desc='evaluating...'):
            # Acessa os elementos do batch
            texts = batch[0]  # O tensor de entrada
            helpfulness = batch[1]  # O tensor de rótulos

            predictions = model(texts.to(device)).squeeze(1)
            total_predictions.extend(p.item() for p in predictions)

            loss = criterion(predictions, helpfulness.to(device))
            acc = binary_accuracy(predictions, helpfulness.to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), total_predictions


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    valid_loss, valid_acc, _  = evaluate(model, valid_iterator, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')