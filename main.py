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
    
    # Definindo os campos para os dados
    text_field = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)

    stars_field = Field(sequential=False, use_vocab=False,dtype=torch.float)  # Rótulo para as estrelas (binário)
    helpfulness_field = Field(sequential=False, use_vocab=False,dtype=torch.float)  # Rótulo para a ajuda (binário)
    
    # Carregar o dataset de treino
    train_dataset = TabularDataset(
        path=f'{path}/balanced_train_apps.csv',
        format='csv',
        fields=[('text', text_field), ('stars', stars_field), ('helpfulness', helpfulness_field)]
    )
    text_field.build_vocab(train_dataset, vectors=embeddings, max_size=25000)
    
    # Criando o iterador (BucketIterator) que pode lidar com o padding e batching
    train_loader, = BucketIterator.splits(
        (train_dataset,),
        batch_size=batch_size,
        device=device,  # Pode ser 'cuda' ou 'cpu'
        sort_within_batch=True,  # Para garantir que os textos sejam ordenados dentro de cada batch
        sort_key=lambda x: len(x.text),  # Ordena pelo comprimento do texto
        shuffle=True  # Embaralha os dados
    )
    
    return train_loader, text_field 


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
            print(batch)
            text, _ = batch.text  # 'text' e 'text_lengths' são extraídos corretamente
            text = text.to(device)

            print(f"Forma do tensor de entrada text: {text.shape}")  # [seq_len, batch_size]

            # Se o tensor text estiver transposto, você precisa ajustá-lo para [batch_size, seq_len]
            text = text.t()  # Agora a forma será [batch_size, seq_len]
            
            print(f"Forma do tensor text após transposição: {text.shape}")
            # Aqui, as labels (stars e helpfulness) já são tensores, então podemos tratá-los diretamente
            stars_labels = batch.stars.to(device).float()  # Conversão para float se necessário
            helpfulness_labels = batch.helpfulness.to(device).float()  # Tornar float para BCEWithLogitsLoss
            # Zero gradientes
            optimizer.zero_grad()
            
            # Forward pass
            stars_labels = stars_labels.unsqueeze(1) 
            stars_output, helpfulness_output = model(text)
            print("polarity_output shape:", stars_output.shape)  # Deve ser [32, 1]

            # Rótulos
            print("stars_labels shape:", stars_labels.shape)  # Deve ser [32]
            
            # Calcular a perda para cada tarefa
            loss_stars = criterion(stars_output, stars_labels)  # Remover a dimensão extra
            loss_helpfulness = criterion(helpfulness_output.squeeze(1), helpfulness_labels)
            
            # Total da perda
            loss = loss_stars + loss_helpfulness
            loss.backward()  # Backpropagation
            
            # Atualizar pesos
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calcular acurácia para stars e helpfulness
            pred_stars = torch.round(stars_output)  # Remover o sigmoid e o round
            pred_helpfulness = torch.round(helpfulness_output)  # Aplicar sigmoid e arredondar para binário
            
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
        torch.save(model.state_dict(), "cnn_apps.pt")
        
# Função de validação
def validate_model(model, val_loader, device):
    model.eval()  # Coloca o modelo em modo de validação
    with torch.no_grad():  # Desliga o cálculo do gradiente
        correct_stars = 0
        correct_helpfulness = 0
        total = 0
        
        for batch in val_loader:
            text, stars_labels, helpfulness_labels = batch.text
            text = text.to(device)
            stars_labels = stars_labels.to(device).float()  # Tornar float para BCEWithLogitsLoss
            helpfulness_labels = helpfulness_labels.to(device).float()  # Tornar float para BCEWithLogitsLoss
            
            # Forward pass
            stars_output, helpfulness_output = model(text)
            
            # Calcular as predições
            pred_stars = torch.round(torch.sigmoid(stars_output))
            pred_helpfulness = torch.round(torch.sigmoid(helpfulness_output))
            
            correct_stars += (pred_stars == stars_labels).sum().item()
            correct_helpfulness += (pred_helpfulness == helpfulness_labels).sum().item()
            total += text.size(0)
        
        # Acurácia de validação
        accuracy_stars = 100 * correct_stars / total
        accuracy_helpfulness = 100 * correct_helpfulness / total
        
        print(f"Validation - Stars Accuracy: {accuracy_stars:.2f}%, "
              f"Helpfulness Accuracy: {accuracy_helpfulness:.2f}%")

# Caminho para os dados e embeddings
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

# Treinar o modelo

# class CNN_Multitask(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
#                  dropout, vectors):
#         super().__init__()

#         # Embedding layer
#         self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

#         # Adjusting the convolutional layers to handle smaller sequences
#         # Reducing kernel sizes for better compatibility with short sequences
#         self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(2, embedding_dim))
#         self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(2, embedding_dim))
#         self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(2, embedding_dim))

#         # Fully connected layer
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, 32)

#         # Output layers for the two tasks
#         self.fc1_stars = nn.Linear(32, output_dim)  # For stars (polarity)
#         self.fc2_helpfulness = nn.Linear(32, output_dim)  # For helpfulness

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, text):
#         # text: [batch_size, sent_len]

#         embedded = self.embedding(text)  # [batch_size, sent_len, emb_dim]
#         embedded = embedded.unsqueeze(1)  # [batch_size, 1, sent_len, emb_dim]

#         # Convolution operations
#         conved_0 = F.relu(self.conv_0(embedded).squeeze(3))  # [batch_size, n_filters, sent_len - kernel_size[0] + 1]
#         conved_1 = F.relu(self.conv_1(embedded).squeeze(3))  # [batch_size, n_filters, sent_len - kernel_size[1] + 1]
#         conved_2 = F.relu(self.conv_2(embedded).squeeze(3))  # [batch_size, n_filters, sent_len - kernel_size[2] + 1]

#         # Max pooling operations
#         pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)  # [batch_size, n_filters]
#         pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)  # [batch_size, n_filters]
#         pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)  # [batch_size, n_filters]

#         # Concatenating the pooled features
#         cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))  # [batch_size, n_filters * len(filter_sizes)]

#         # Feed forward through the fully connected layer
#         out = F.relu(self.fc(cat))  # [batch_size, 32]

#         # Output for stars (polarity)
#         out_polarity = self.fc1_stars(out)  # [batch_size, 1]
        
#         # Output for helpfulness
#         out_helpfulness = self.fc2_helpfulness(out)  # [batch_size, 1]

#         return out_polarity, out_helpfulness
class CNN_Multitask(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 dropout, vectors):

        super().__init__()

        # 1 camada - embeddings
        if vectors is None:
            vectors = torch.rand((vocab_size, embedding_dim))  # Exemplo de valor padrão
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

        # 3 camadas de convolução
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(1, embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(1, embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(1, embedding_dim))

        # A saída dos filtros alimenta uma mlp
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 32)

        # Cabeças de saída
        self.fc_polarity = nn.Linear(32, 1)  # Cabeça para polarity
        self.fc_helpfulness = nn.Linear(32, 1)  # Cabeça para helpfulness

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

        # Cabeças de saída
        polarity = torch.sigmoid(self.fc_polarity(out))  # Resultado binário para polarity
        helpfulness = torch.sigmoid(self.fc_helpfulness(out))  # Resultado binário para helpfulness

        return polarity, helpfulness
model = CNN_Multitask(vocab_size, 300, n_filters, filter_sizes, dropout, vectors)

num_epochs = 5
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_model(model, train_loader, num_epochs, learning_rate, device)