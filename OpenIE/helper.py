'''Helper functions used in open information extraction task.'''

import spacy
import torch
import pandas as pd
import matplotlib.pyplot as plt


# function to load the datatset
def load_dataset(file_path):
    sentences = []
    labels = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        sentence = []
        
        for line in lines:
            line = line.strip() # remove leading/trailing whitespaces
            if line:
                if not line.startswith('ARG1') and not line.startswith('ARG2') and not line.startswith('REL') and not line.startswith('LOC') and not line.startswith('TIME') and not line.startswith('NONE'):
                    sentence = line
                else:
                    current_label = line
                    sentences.append(sentence)
                    labels.append(current_label)
                    
    return sentences, labels


def remerge_sent(sent):
    # merges tokens which are not separated by white-space
    # does this recursively until no further changes
    changed = True
    while changed:
        changed = False
        i = 0
        while i < sent.__len__() - 1:
            tok = sent[i]
            if not tok.whitespace_:
                ntok = sent[i + 1]
                # in-place operation.
                with sent.retokenize() as retokenizer:
                    retokenizer.merge(sent[i: i + 2])
                changed = True
            i += 1
    return sent


# Retrieves BERT embeddings for a list of tokens using a fast tokenizer, enabling accurate aggregation of subword embeddings into their original token representations.
def get_bert_embeddings(tokens, tokenizer, bert_model):
    inputs = tokenizer(tokens, return_tensors='pt', is_split_into_words=True, padding=False, truncation=True)

    # Get BERT embeddings from the model
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Get the embeddings for each subword
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (sequence_length, hidden_size)
    # Get word_ids to align subword tokens with the original tokens
    word_ids = inputs.word_ids()

    # Aggregate subword embeddings back to their original tokens
    aggregated_embeddings = []
    current_token_embeddings = []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if len(current_token_embeddings) > 0 and word_id != word_ids[idx - 1]:
            aggregated_embeddings.append(torch.mean(torch.stack(current_token_embeddings), dim=0))
            current_token_embeddings = []
        current_token_embeddings.append(token_embeddings[idx])
    
    if len(current_token_embeddings) > 0:
        aggregated_embeddings.append(torch.mean(torch.stack(current_token_embeddings), dim=0))

    return torch.stack(aggregated_embeddings)


# Function to generate BERT embeddings for dataFrame
def generate_embeddings(df, tokenizer, bert_model):
    embeddings_list = []
    for _, row in df.iterrows():
        tokenized_sentence = row['Tokens']
        embeddings = get_bert_embeddings(tokenized_sentence, tokenizer, bert_model)
        embeddings_list.append(embeddings)
    return embeddings_list


# Function to pad labels to max length
def pad_labels(labels, max_len, padding_label):
    # Initialize a tensor with the padding label (assuming integer encoding)
    padded_labels = torch.full((max_len,), padding_label, dtype=torch.long)
    padded_labels[:len(labels)] = torch.tensor(labels, dtype=torch.long)

    return padded_labels


def plot_training_loss(num_epochs, epoch_losses):
    # Plotting the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.xticks(range(1, num_epochs + 1))  # Set x-ticks to each epoch
    plt.show()


def evaluate_bilstm_crf(model, val_loader, label_to_idx):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_predicted_tags = []  # List to collect all predicted tags for analysis

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Create mask for valid tokens (non-padding)
            mask = labels.ne(label_to_idx['PADDING'])  # Mask of shape [batch_size, seq_len]

            # Forward pass through the model to get predicted tags (for CRF)
            predicted_tags = model(inputs)  # No need to pass the mask; CRF decodes the valid outputs

            # Since predicted_tags is a list of lists, convert it to a tensor for indexing
            predicted_tags = torch.tensor(predicted_tags, dtype=torch.long, device=labels.device)

            # Calculate loss only for valid tokens (masking padding labels)
            loss = model(inputs, tags=labels)  # This returns the loss with CRF
            total_loss += loss.item()

            # Mask needs to be applied on predicted_tags and labels
            correct += (predicted_tags[mask] == labels[mask]).sum().item()  # Compare valid tokens
            total += mask.sum().item()  # Count only valid tokens

    avg_loss = total_loss / len(val_loader)  # Average loss per batch
    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


def evaluate_bilstm_crf_with_text_output(model, embeddings, labels, label_encoder):
    model.eval()
    with torch.no_grad():
        for emb, true_labels in zip(embeddings, labels):
            # Add batch dimension
            emb = emb.unsqueeze(0)  # Shape: [1, seq_length, input_dim]
            true_labels = true_labels.unsqueeze(0)  # Shape: [1, seq_length]

            # Forward pass through the model to get predictions
            # Ensure the model returns the predicted tags from CRF
            predicted_tags = model(emb)  # Get predicted labels from CRF
            
            # Since `predicted_tags` is a list of predicted sequences, you may need to convert it to a tensor
            # If you want to use just the first sequence for evaluation
            if isinstance(predicted_tags, list):
                predicted_indices = predicted_tags[0]  # Get the first sequence
            else:
                predicted_indices = predicted_tags  # Otherwise, use it directly

            # Convert predicted and true label indices to label names
            predicted_labels = label_encoder.inverse_transform(predicted_indices)  # No need for squeeze() here
            true_labels_text = label_encoder.inverse_transform(true_labels.squeeze(0).cpu().numpy())

            # Display the predicted and true labels
            print("Predicted Labels: ", predicted_labels)
            print("True Labels:      ", true_labels_text)
            print("-" * 50)


def load_test_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    sentences = [sentence.strip() for sentence in sentences]
    df_test = pd.DataFrame(sentences, columns=['Sentence'])
    return df_test


def generate_test_token(row):
    # Process the sentence with SpaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(row['Sentence'])
    spacy_sentence = remerge_sent(doc)
    tokens = [token.text for token in spacy_sentence]

    return tokens


def generate_extractions_bilstm_crf(model, padded_embeddings, df_test, labels, idx_to_label):
    model.eval()
    
    with torch.no_grad():
        # Forward pass through the model to get predicted tags (for CRF)
        predicted_tags = model(padded_embeddings)  # No need to pass the mask; CRF decodes the valid outputs
        # Since predicted_tags is a list of lists, convert it to a tensor for indexing
        predicted_tags = torch.tensor(predicted_tags, dtype=torch.long, device=labels.device)

    results = []
    
    for i, row in df_test.iterrows():
        sentence = row['Sentence']
        tokens = row['Tokens']
        n = len(tokens)  # Number of tokens in the current sentence

        predicted_labels = predicted_tags[i][:n]
        
        # Get valid tokens and their corresponding predictions
        valid_tokens = tokens[:len(predicted_labels)]
        sentence_results = []
        
        current_rel = []  # Store current relation tokens
        current_time, current_loc = '', ''  # Track TIME and LOC
        arg1_list, arg2_list = [], []  # Collect all possible ARG1 and ARG2 terms
        arg1_pos, arg2_pos, rel_pos = -1, -1, -1  # Track positions of ARG1, ARG2, REL
        
        for token_index in range(len(predicted_labels)):
            if token_index >= len(valid_tokens):
                print(f"Warning: token_index {token_index} is out of bounds for valid_tokens.")
                break  # Prevent out of range access
            
            predicted_class = predicted_labels[token_index]
            
            label = idx_to_label[predicted_class.item()]  # Convert index to label
            
            token = valid_tokens[token_index]  # Use the valid token
            
            if label == 'REL':
                current_rel.append(token)
                rel_pos = token_index
            elif label == 'ARG1':
                arg1_list.append(token)
                arg1_pos = token_index
            elif label == 'ARG2':
                arg2_list.append(token)
                arg2_pos = token_index
            elif label == 'TIME':
                current_time = token
            elif label == 'LOC':
                current_loc = token
        
        # Form relations based on different ARG1/ARG2 and REL patterns
        if current_rel:
            rel = ' '.join(current_rel)
            
            # Standard pattern: ARG1 before REL, ARG2 after REL
            if arg1_pos != -1 and arg2_pos != -1 and arg1_pos < rel_pos < arg2_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{current_time}\t{current_loc}")
            
            # Inverted pattern: ARG2 before REL, ARG1 after REL
            if arg1_pos != -1 and arg2_pos != -1 and arg2_pos < rel_pos < arg1_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg2_list)}\t{' '.join(arg1_list)}\t{current_time}\t{current_loc}")
            
            # Both ARG1 and ARG2 before REL
            if arg1_pos != -1 and arg2_pos != -1 and arg1_pos < arg2_pos < rel_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{current_time}\t{current_loc}")
            
            # Both ARG1 and ARG2 after REL
            if arg1_pos != -1 and arg2_pos != -1 and rel_pos < arg1_pos < arg2_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{current_time}\t{current_loc}")
        
        # Check for incomplete relations (only ARG1 or ARG2 found)
        if current_rel:
            if arg1_list and not arg2_list:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t\t{current_time}\t{current_loc}")
            elif arg2_list and not arg1_list:
                sentence_results.append(f"{sentence}\t1\t{rel}\t\t{' '.join(arg2_list)}\t{current_time}\t{current_loc}")
        
        results.extend(sentence_results)

    return results


