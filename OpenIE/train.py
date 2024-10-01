'''This file trains the supervised model for Open IE task.'''

# Imports
import warnings
warnings.filterwarnings("ignore")

import helper
import model

import os
import spacy
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LR = 0.001
NUM_EPOCHS = 100
TRAIN_DATASET_SIZE = 4500
MODEL_FILENAME = "model_checkpoint.pth.tar"

# Model parameters
INPUT_DIM = 768  # BERT Embedding size
HIDDEN_DIM = 256
LOAD_MODEL = True

# load the dataset into a pandas dataframe
sentences, labels = helper.load_dataset('./Dataset/original_cleaned')
df = pd.DataFrame({
    'Sentence': sentences,
    'Labels': labels
})

# Training on a subset of dataset
# as loading entire dataset into memory
# causes kernel crash
df = df[:TRAIN_DATASET_SIZE]

# Tokenize sentences using spacy
nlp = spacy.load('en_core_web_sm')

def check_token_label_length(row):
    doc = nlp(row['Sentence'])
    spacy_sentence = helper.remerge_sent(doc)
    tokens = [token.text for token in spacy_sentence]
    labels = row['Labels'].split()

    is_match = len(tokens) == len(labels)
    return is_match, len(tokens), len(labels), tokens

df[['Token_Label_Match', 'Num_Tokens', 'Num_Labels', 'Tokens']] = df.apply(
    check_token_label_length,
    axis=1,
    result_type="expand",
)

# Load the pre-trained BERT tokenizer and model
# Load the fast version of the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
bert_model = AutoModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

embeddings_list = helper.generate_embeddings(df, tokenizer, bert_model)

# pad the embeddings to ensure uniformity across dataset as model will be trained in batches
padded_embeddings = pad_sequence(embeddings_list, batch_first=True)
df['Embeddings'] = [padded_embeddings[i] for i in range(padded_embeddings.shape[0])]

# encode the labels
label_encoder = LabelEncoder()
labels_list = ['ARG1', 'ARG2', 'REL', 'TIME', 'LOC', 'NONE', 'PADDING'] # Adding a new label for padding
label_encoder.fit(labels_list)

df['Encoded_Labels'] = df['Labels'].apply(lambda x: label_encoder.transform(x.split()))

padding_label = label_encoder.transform(['PADDING'])[0]
max_len = max(len(label) for label in df['Encoded_Labels'])

padded_labels = [helper.pad_labels(label, max_len, padding_label) for label in df['Encoded_Labels']]
df['Padded_Labels'] = padded_labels

df['Padded_Label_Length'] = df['Padded_Labels'].apply(len)

# Split the data into training and validation sets
train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
    df['Embeddings'].tolist(),  
    df['Padded_Labels'].tolist(),
    test_size=0.1,
    random_state=42
)

train_dataset = TensorDataset(
    torch.stack(train_embeddings),
    torch.stack(train_labels),
)
val_dataset = TensorDataset(
    torch.stack(val_embeddings),
    torch.stack(val_labels),
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

label_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}

output_dim = len(labels_list)

model = model.BiLSTM_CRF(INPUT_DIM, HIDDEN_DIM, output_dim)

loss_fn = nn.CrossEntropyLoss(ignore_index=label_to_idx['PADDING'])
optimizer = optim.Adam(model.parameters(), lr=LR)

if LOAD_MODEL and os.path.exists(MODEL_FILENAME):
    print("Loading checkpoint")
    model.load_state_dict(torch.load(MODEL_FILENAME)["state_dict"])
    optimizer.load_state_dict(torch.load(MODEL_FILENAME)["optimizer"])

# List to store the average loss per epoch
epoch_losses = []

# Training the model
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    # Saving checkpoint after every 2 epochs
    if epoch %2 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        print("Saving checkpoint")
        torch.save(checkpoint, MODEL_FILENAME)

    for batch in train_loader:
        inputs, labels = batch

        # Create mask for valid tokens (non-padding)
        # will set valid labels to True and padding labels to False
        mask = labels.ne(label_to_idx['PADDING'])  # Mask of shape [batch_size, seq_len]

        # Forward pass through the model to get logits
        loss = model(inputs, tags=labels, mask=mask)  #Shape: [batch_size, max_seq_len, num_classes]
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)  # Store the average loss for this epoch

    
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}")

helper.plot_training_loss(
    NUM_EPOCHS,
    epoch_losses,
)
