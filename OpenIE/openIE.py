'''This file tests the OpenIE model on val and test set.'''

# Imports
import warnings
warnings.filterwarnings("ignore")

import helper
import model

import os
import spacy
import torch

from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim


MODEL_FILENAME = "model_checkpoint.pth.tar"
# EXTRACTION_FILE_PATH = '../CaRB/system_outputs/test/extractions.txt'
EXTRACTION_FILE_PATH = 'extractions.txt'

# Model parameters
LR = 0.001
INPUT_DIM = 768  # BERT Embedding size
HIDDEN_DIM = 256

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
bert_model = AutoModel.from_pretrained('bert-base-uncased')

label_encoder = LabelEncoder()
labels_list = ['ARG1', 'ARG2', 'REL', 'TIME', 'LOC', 'NONE', 'PADDING']
label_encoder.fit(labels_list)

label_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
output_dim = len(labels_list)

openIEModel = model.BiLSTM_CRF(INPUT_DIM, HIDDEN_DIM, output_dim)
optimizer = optim.Adam(openIEModel.parameters(), lr=LR)

if os.path.exists(MODEL_FILENAME):
    print("Loading checkpoint")
    openIEModel.load_state_dict(torch.load(MODEL_FILENAME)["state_dict"])
    optimizer.load_state_dict(torch.load(MODEL_FILENAME)["optimizer"])

# avg_val_loss, val_accuracy = helper.evaluate_bilstm_crf(
#     openIEModel,
#     train.val_loader,
#     train.label_to_idx,
# )

# Set up logging
# logging.basicConfig(
#     filename='bilstm_crf_val_predictions.log', 
#     level=logging.INFO, 
#     format='%(asctime)s - %(message)s',
# )

# helper.evaluate_bilstm_crf_with_text_output(
#     openIEModel,
#     train.val_embeddings,
#     train.val_labels,
#     train.label_encoder,
#     logging,
# )

df_test = helper.load_test_dataset('./Dataset/test.txt')

df_test['Tokens'] = df_test.apply(helper.generate_test_token, axis=1)

embeddings_list_test = helper.generate_embeddings(
    df_test,
    tokenizer,
    bert_model,
)

# pad the embeddings to ensure uniformity across dataset as model will be trained in batches
padded_embeddings_test = pad_sequence(embeddings_list_test, batch_first=True)
df_test['Embeddings'] = [padded_embeddings_test[i] for i in range(padded_embeddings_test.shape[0])]

# Create the reverse mapping from index to label
idx_to_label = {v: k for k, v in label_to_idx.items()}

extractions, skipped_count = helper.generate_extractions_bilstm_crf_strict(openIEModel, padded_embeddings_test, df_test)
print(f"Total skipped sentences: {skipped_count}")

# Write output extractions to a tab separated file
with open('../CaRB/system_outputs/test/extractions_strict.txt', 'w') as f:
    for extraction in extractions:
        f.write(extraction + '\n')

print("Test Extraction Strict file has been created.")

extractions, skipped_count = helper.generate_extractions_bilstm_crf_with_confidence(openIEModel, padded_embeddings_test, df_test)
print(f"Total skipped sentences: {skipped_count}")

# Write output extractions to a tab separated file
with open('../CaRB/system_outputs/test/extractions_confidence.txt', 'w') as f:
    for extraction in extractions:
        f.write(extraction + '\n')

print("Test Extraction Confidence file has been created.")

# error analysis
nlp = spacy.load('en_core_web_sm')
helper.analyze_extractions(
    nlp,
    './Dataset/test_gold.txt',
    './Dataset/extractions.txt',
)
