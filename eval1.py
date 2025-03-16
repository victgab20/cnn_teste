import pandas as pd
import torch
import random
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

#import torchtext.vocab
import tqdm
import torch.nn as nn
import torch.nn.functional as F

import pickle
from nltk import tokenize
import torch.optim as optim

from sklearn.metrics import classification_report
import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset,BucketIterator
from torchtext.vocab import Vectors
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import precision_score, recall_score, f1_score

import gensim

path = r"C:\Users\victo\Downloads\corpus\labeled"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

BATCH_SIZE = 32

# Função para carregar e tokenizar os dados
class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.data = pd.read_csv(csv_path)
        self.texts = self.data['text']
        self.labels_polarity = self.data['stars']
        self.labels_utility = self.data['helpfulness']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        polarity = self.labels_polarity[idx]
        utility = self.labels_utility[idx]
        tokens = self.tokenizer(text)
        return tokens, polarity, utility

def tokenizer(text):
    return word_tokenize(text, language='portuguese')



# Função para carregar os embeddings do GloVe
def load_glove_embeddings(glove_path, cache_path):
    print(f"Carregando os embeddings do GloVe a partir de {glove_path}...")
    embeddings = Vectors(name=glove_path, cache=cache_path)
    print(f"Embeddings carregados com {len(embeddings.stoi)} palavras.")
    return embeddings

def process_splits(path, batch_size, tokenizer):
    print("Carregando os datasets...")
    
    text_field = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)

    stars_field = Field(sequential=False, use_vocab=False,dtype=torch.float)  # Rótulo para as estrelas (binário)
    helpfulness_field = Field(sequential=False, use_vocab=False,dtype=torch.float)  # Rótulo para a ajuda (binário)
    
    train_dataset = TabularDataset(
        path=f'{path}/balanced_test_apps.csv',
        format='csv',
        fields=[('text', text_field), ('stars', stars_field), ('helpfulness', helpfulness_field)],
        skip_header=True
    )

    # Dividir o dataset (o primeiro será os 5%, o segundo o restante)
    text_field.build_vocab(train_dataset, vectors=embeddings, max_size=25000)
    
    # Criando o iterador (BucketIterator) que pode lidar com o padding e batching
    train_loader, = BucketIterator.splits(
        (train_dataset,),
        batch_size=batch_size,
        device=device,  # Pode ser 'cuda' ou 'cpu'
        sort_within_batch=True,  # Para garantir que os textos sejam ordenados dentro de cada batch
        sort_key=lambda x: len(x.text),  # Ordena pelo comprimento do texto
        shuffle=False  # Embaralha os dados
    )
    
    return train_loader, text_field 

glove_path = f"{path}/glove_s300.txt"
tokenizer = lambda x: x.split()  # Simples tokenização por espaços

# Carregar os embeddings
cache_path = f"{path}/.vector_cache"  # Caminho para cache
embeddings = load_glove_embeddings(glove_path, cache_path)

# Carregar o dataset
train_loader,text_field  = process_splits(path, BATCH_SIZE, tokenizer)

vocab_size = len(text_field.vocab)
embedding_dim = 300  # Ajustado para GloVe 300D
n_filters = 100
filter_sizes = [3, 4, 5]
output_dim_polarity = 1  # Binário
output_dim_utility = 1  # Binário
dropout = 0.5
vectors = text_field.vocab.vectors


# Função de treinamento
def train_model(model, train_loader, num_epochs, learning_rate, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()  # Perda binária para ambas as tarefas
    
    for epoch in range(num_epochs):
        model.train()  # Coloca o modelo em modo de treinamento
        
        running_loss = 0.0
        correct_stars = 0
        correct_helpfulness = 0
        total = 0
        for batch in train_loader:
            text, _ = batch.text  # 'text' e 'text_lengths' são extraídos corretamente
            text = text.to(device)

            # Se o tensor text estiver transposto, você precisa ajustá-lo para [batch_size, seq_len]
            text = text.t()  # Agora a forma será [batch_size, seq_len]
            
            # Aqui, as labels (stars e helpfulness) já são tensores, então podemos tratá-los diretamente
            stars_labels = batch.stars.to(device).float()  # Conversão para float se necessário
            helpfulness_labels = batch.helpfulness.to(device).float()  # Tornar float para BCEWithLogitsLoss
            # Zero gradientes
            optimizer.zero_grad()
            
            # Forward pass
            stars_labels = stars_labels.unsqueeze(1)
            helpfulness_labels = helpfulness_labels.unsqueeze(1) 
            stars_output, helpfulness_output = model(text)
            
            # Calcular a perda para cada tarefa
            loss_stars = criterion(stars_output, stars_labels)  # Remover a dimensão extra
            loss_helpfulness = criterion(helpfulness_output, helpfulness_labels)
            
            # Total da perda
            loss = loss_stars + loss_helpfulness
            loss.backward()  # Backpropagation
            
            # Atualizar pesos
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calcular acurácia para stars e helpfulness

            pred_stars = torch.round(stars_output)
            pred_helpfulness = torch.round(helpfulness_output)

            correct_stars += (pred_stars == stars_labels).sum().item()
            correct_helpfulness += (pred_helpfulness == helpfulness_labels).sum().item()
            total += text.size(0)
        # Calcular a acurácia por época
        accuracy_stars = 100 * correct_stars / total
        accuracy_helpfulness = 100 * correct_helpfulness / total
        
        # Média de perda por época
        epoch_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Stars Accuracy: {accuracy_stars:.2f}%, Helpfulness Accuracy: {accuracy_helpfulness:.2f}%")
        
# Função de validação

def validate_model(model, val_loader, device):
    model.eval()  # Modo de validação
    with torch.no_grad():  
        total = 0

        all_stars_labels = []
        all_stars_preds = []
        all_helpfulness_labels = []
        all_helpfulness_preds = []

        for batch in val_loader:
            text, _ = batch.text  # 'text' e 'text_lengths' são extraídos corretamente
            text = text.to(device)

            # Se o tensor text estiver transposto, você precisa ajustá-lo para [batch_size, seq_len]
            text = text.t()  # Agora a forma será [batch_size, seq_len]
            
            # Aqui, as labels (stars e helpfulness) já são tensores, então podemos tratá-los diretamente
            stars_labels = batch.stars.to(device).float()  # Conversão para float se necessário
            helpfulness_labels = batch.helpfulness.to(device).float()  # Tornar float para BCEWithLogitsLoss

            # Forward pass
            stars_labels = stars_labels.unsqueeze(1)
            helpfulness_labels = helpfulness_labels.unsqueeze(1) 
            stars_output, helpfulness_output = model(text)
            
            # Gerar predições
            pred_stars = torch.round(torch.sigmoid(stars_output))
            pred_helpfulness = torch.round(torch.sigmoid(helpfulness_output))

            total += text.size(0)

            # Armazena os valores para métricas
            all_stars_labels.extend(stars_labels.cpu().numpy())
            all_stars_preds.extend(pred_stars.cpu().numpy())
            all_helpfulness_labels.extend(helpfulness_labels.cpu().numpy())
            all_helpfulness_preds.extend(pred_helpfulness.cpu().numpy())


        # Cálculo das métricas
        precision_stars = precision_score(all_stars_labels, all_stars_preds, zero_division=0)
        recall_stars = recall_score(all_stars_labels, all_stars_preds, zero_division=0)
        f1_stars = f1_score(all_stars_labels, all_stars_preds, zero_division=0)

        precision_helpfulness = precision_score(all_helpfulness_labels, all_helpfulness_preds, zero_division=0)
        recall_helpfulness = recall_score(all_helpfulness_labels, all_helpfulness_preds, zero_division=0)
        f1_helpfulness = f1_score(all_helpfulness_labels, all_helpfulness_preds, zero_division=0)
        print(f"Validation - Stars: Precision: {precision_stars:.2f}, Recall: {recall_stars:.2f}, F1: {f1_stars:.2f}")
        print(f"Validation - Helpfulness: Precision: {precision_helpfulness:.2f}, Recall: {recall_helpfulness:.2f}, F1: {f1_helpfulness:.2f}")


class CNN_Multitask(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, dropout, vectors):
        super().__init__()

        # Camada de embeddings
        if vectors is None:
            vectors = torch.rand((vocab_size, embedding_dim))
        else:
            vectors = torch.tensor(vectors, dtype=torch.float)

        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

        # Múltiplas camadas de convolução com diferentes tamanhos de filtro
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])

        # Camada totalmente conectada após convoluções
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 32)

        # Cabeças de saída
        self.fc_polarity = nn.Linear(32, 1)  # Saída para polarity
        self.fc_helpfulness = nn.Linear(32, 1)  # Saída para helpfulness

        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)  
        embedded = embedded.unsqueeze(1)  

        max_filter_size = max([conv.kernel_size[0] for conv in self.convs])
        if embedded.shape[2] < max_filter_size:
            pad_amount = max_filter_size - embedded.shape[2]
            embedded = F.pad(embedded, (0, 0, pad_amount, 0))  

        conved = [
            F.relu(conv(embedded).squeeze(3)) 
            for conv in self.convs if conv.kernel_size[0] <= embedded.shape[2]
        ]
        
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  
        out = F.relu(self.fc(cat))

        polarity = self.fc_polarity(out)
        helpfulness = self.fc_helpfulness(out)

        return polarity, helpfulness

model = CNN_Multitask(vocab_size, 300, n_filters, filter_sizes, dropout, vectors)

num_epochs = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('cnn_apps_1.pt', map_location=device))
model.to(device)

validate_model(model, train_loader, device)


#train_model(model, train_loader, num_epochs, learning_rate, device)