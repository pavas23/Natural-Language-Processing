'''Helper functions used in Open Information Extraction task.'''

# Imports
import spacy
import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
labels_list = ['ARG1', 'ARG2', 'REL', 'TIME', 'LOC', 'NONE', 'PADDING']
label_encoder.fit(labels_list)

label_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
idx_to_label = {v: k for k, v in label_to_idx.items()}

# Function to load the datatset
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

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Create mask for valid tokens (non-padding)
            mask = labels.ne(label_to_idx['PADDING'])  # Mask of shape [batch_size, seq_len]

            # Forward pass through the model to get predicted tags (for CRF)
            predicted_tags = model(inputs)

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


def evaluate_bilstm_crf_with_text_output(model, embeddings, labels, label_encoder, logging):
    model.eval()
    with torch.no_grad():
        for emb, true_labels in zip(embeddings, labels):
            # Add batch dimension
            emb = emb.unsqueeze(0)  # Shape: [1, seq_length, input_dim]
            true_labels = true_labels.unsqueeze(0)  # Shape: [1, seq_length]

            # Forward pass through the model to get predictions
            # Ensure the model returns the predicted tags from CRF
            predicted_tags = model(emb) 
            
            if isinstance(predicted_tags, list):
                predicted_indices = predicted_tags[0]  # Get the first sequence
            else:
                predicted_indices = predicted_tags 

            # Convert predicted and true label indices to label names
            predicted_labels = label_encoder.inverse_transform(predicted_indices)  # No need for squeeze() here
            true_labels_text = label_encoder.inverse_transform(true_labels.squeeze(0).cpu().numpy())

            # Log the predicted and true labels
            logging.info(f"Predicted Labels: {predicted_labels}")
            logging.info(f"True Labels:      {true_labels_text}")
            logging.info("-" * 50)


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


def generate_extractions_bilstm_crf_strict(model, padded_embeddings, df_test):
    model.eval()
    
    with torch.no_grad():
        predicted_tags = model(padded_embeddings)
        predicted_tags = torch.tensor(predicted_tags, dtype=torch.long, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    results = []
    skipped_count = 0
    
    for i, row in df_test.iterrows():
        sentence = row['Sentence']
        tokens = row['Tokens']
        n = len(tokens) 

        predicted_labels = predicted_tags[i][:n]
        valid_tokens = tokens[:len(predicted_labels)]
        sentence_results = []
        
        current_rel = []
        current_time = []
        current_loc = []
        arg1_list, arg2_list = [], []
        arg1_pos, arg2_pos, rel_pos = -1, -1, -1
        
        found_rel = False  
        found_arg1 = False 
        found_arg2 = False  
        found_time = False 
        found_loc = False

        for token_index in range(len(predicted_labels)):
            if token_index >= len(valid_tokens):
                print(f"Warning: token_index {token_index} is out of bounds for valid_tokens.")
                break
            
            predicted_class = predicted_labels[token_index]
            label = idx_to_label[predicted_class.item()] 
            token = valid_tokens[token_index]
            
            if label == 'REL':
                found_rel = True  
                if current_rel and token_index == current_rel[-1][-1] + 1:
                    current_rel[-1][0].append(token)  # Continue building multi-word relation
                    current_rel[-1][-1] = token_index
                else:
                    current_rel.append([[token], token_index])  # Start a new relation phrase
                rel_pos = token_index

            elif label == 'ARG1':
                found_arg1 = True  
                arg1_list.append(token)
                arg1_pos = token_index

            elif label == 'ARG2':
                found_arg2 = True  
                arg2_list.append(token)
                arg2_pos = token_index

            elif label == 'TIME':
                found_time = True 
                if current_time and token_index == current_time[-1][-1] + 1:
                    current_time[-1][0].append(token)  # Continue building multi-token time phrase
                    current_time[-1][-1] = token_index
                else:
                    current_time.append([[token], token_index])  # Start a new time phrase

            elif label == 'LOC':
                found_loc = True  
                if current_loc and token_index == current_loc[-1][-1] + 1:
                    current_loc[-1][0].append(token)  # Continue building multi-token location phrase
                    current_loc[-1][-1] = token_index
                else:
                    current_loc.append([[token], token_index])  # Start a new location phrase

        if current_rel:
            rel = ' '.join([' '.join(r[0]) for r in current_rel])  # Multi-word relation
            time_str = ' '.join([' '.join(t[0]) for t in current_time]) if current_time else ''
            loc_str = ' '.join([' '.join(l[0]) for l in current_loc]) if current_loc else ''
            
            # Standard pattern: ARG1 before REL, ARG2 after REL
            if found_arg1 and found_arg2 and arg1_pos < rel_pos < arg2_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
            
            # Inverted pattern: ARG2 before REL, ARG1 after REL
            if found_arg1 and found_arg2 and arg2_pos < rel_pos < arg1_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg2_list)}\t{' '.join(arg1_list)}\t{time_str}\t{loc_str}")
            
            # Both ARG1 and ARG2 before REL
            if found_arg1 and found_arg2 and arg1_pos < arg2_pos < rel_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
            
            # Both ARG1 and ARG2 after REL
            if found_arg1 and found_arg2 and rel_pos < arg1_pos < arg2_pos:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
            
            # Loosened: Capture relations with only ARG1 or ARG2
            if found_arg1 and not found_arg2:
                sentence_results.append(f"{sentence}\t1\t{rel}\t{' '.join(arg1_list)}\t\t{time_str}\t{loc_str}")
            if found_arg2 and not found_arg1:
                sentence_results.append(f"{sentence}\t1\t{rel}\t\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
        
        # Handle case where no relations or arguments are found, but time or location might be present
        if not found_rel and not found_arg1 and not found_arg2:
            if found_time or found_loc:
                # If time or location found, include in results even without relations or arguments
                time_str = ' '.join([' '.join(t[0]) for t in current_time]) if current_time else ''
                loc_str = ' '.join([' '.join(l[0]) for l in current_loc]) if current_loc else ''
                sentence_results.append(f"{sentence}\t0.5\t\t\t{time_str}\t{loc_str}")
            else:
                skipped_count += 1  # Increment skipped count
                print(f"Skipped sentence {i}: {sentence} with predicted labels: {predicted_labels}")
        
        results.extend(sentence_results)

    return results, skipped_count  # Return both results and skipped count


def extract_confidence_features(sentence, rel, arg1, arg2, tokens):
    features = {}

    # Check if (x, r, y) covers all words in the sentence
    features['covers_all_words'] = len(arg1.split()) + len(rel.split()) + len(arg2.split()) == len(tokens)

    # Check for specific prepositions in relation
    last_token_rel = rel.split()[-1] if rel.split() else ''
    features['last_prep_for'] = last_token_rel == 'for'
    features['last_prep_on'] = last_token_rel == 'on'
    features['last_prep_of'] = last_token_rel == 'of'
    features['last_prep_to'] = last_token_rel == 'to'
    features['last_prep_in'] = last_token_rel == 'in'

    # Sentence length
    features['short_sentence'] = len(tokens) <= 10
    features['medium_sentence'] = 10 < len(tokens) <= 20
    features['long_sentence'] = len(tokens) > 20

    # Check for WH-word to the left of the relation
    wh_words = ['who', 'what', 'where', 'when', 'why', 'how']
    features['wh_word_left'] = any(word in tokens[:tokens.index(rel.split()[0])] for word in wh_words if rel)

    # Check if r matches specific patterns (e.g., VW*P)
    vw_p_pattern = True  # Implement specific pattern-matching logic based on your extraction rules
    v_pattern = True  # Implement specific pattern-matching logic based on your extraction rules
    features['r_matches_vw_p'] = vw_p_pattern
    features['r_matches_v'] = v_pattern

    # Check if sentence starts with ARG1
    features['sentence_starts_with_x'] = sentence.startswith(arg1)

    # Proper noun checks for ARG1 and ARG2 (a more advanced check might use POS tagging)
    features['arg1_is_proper_noun'] = arg1.istitle()
    features['arg2_is_proper_noun'] = arg2.istitle()

    return features


def calculate_confidence(features):
    # Feature weights with adjustments to boost confidence for correct predictions
    weights = {
        'covers_all_words': 2.5,  # Further increased weight for covering all words
        'last_prep_for': 1.0,
        'last_prep_on': 0.9,
        'last_prep_of': 0.8,
        'short_sentence': 0.7,
        'wh_word_left': 0.6,
        'r_matches_vw_p': 0.7,
        'last_prep_to': 0.6,
        'last_prep_in': 0.5,
        'medium_sentence': 0.5,
        'sentence_starts_with_x': 0.5,
        'arg2_is_proper_noun': 0.4,
        'arg1_is_proper_noun': 0.2,
        'long_sentence': -0.3,
        'r_matches_v': -0.4,
        'np_left_of_x': -0.5,
        'prep_left_of_x': -0.5,
        'np_right_of_y': -0.7,
        'coord_conj_left_of_r': -0.9,
        'verb_in_rel': 0.6,  
        'valid_structure': 0.8  
    }

    # Compute the weighted sum of the feature values
    confidence_score = sum(weights[key] * features[key] for key in features if key in weights)

    bias = 3.0  
    confidence_score += bias

    # Normalize the confidence score to be between 0 and 1
    min_score = 0  
    max_score = 10 
    normalized_score = (confidence_score - min_score) / (max_score - min_score) * 1

    return max(0, min(normalized_score, 1))


def generate_extractions_bilstm_crf_with_confidence(model, padded_embeddings, df_test):
    model.eval()
    
    with torch.no_grad():
        predicted_tags = model(padded_embeddings)
        predicted_tags = torch.tensor(predicted_tags, dtype=torch.long, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    results = []
    skipped_count = 0
    
    for i, row in df_test.iterrows():
        sentence = row['Sentence']
        tokens = row['Tokens']
        n = len(tokens) 

        predicted_labels = predicted_tags[i][:n]
        valid_tokens = tokens[:len(predicted_labels)]
        sentence_results = []
        
        current_rel = []
        current_time = []
        current_loc = []
        arg1_list, arg2_list = [], []
        arg1_pos, arg2_pos, rel_pos = -1, -1, -1
        
        found_rel = False 
        found_arg1 = False  
        found_arg2 = False  
        found_time = False 
        found_loc = False 

        for token_index in range(len(predicted_labels)):
            if token_index >= len(valid_tokens):
                print(f"Warning: token_index {token_index} is out of bounds for valid_tokens.")
                break
            
            predicted_class = predicted_labels[token_index]
            label = idx_to_label[predicted_class.item()] 
            token = valid_tokens[token_index]
            
            if label == 'REL':
                found_rel = True
                if current_rel and token_index == current_rel[-1][-1] + 1:
                    current_rel[-1][0].append(token)
                    current_rel[-1][-1] = token_index
                else:
                    current_rel.append([[token], token_index])
                rel_pos = token_index

            elif label == 'ARG1':
                found_arg1 = True
                arg1_list.append(token)
                arg1_pos = token_index

            elif label == 'ARG2':
                found_arg2 = True
                arg2_list.append(token)
                arg2_pos = token_index

            elif label == 'TIME':
                found_time = True
                if current_time and token_index == current_time[-1][-1] + 1:
                    current_time[-1][0].append(token)
                    current_time[-1][-1] = token_index
                else:
                    current_time.append([[token], token_index])

            elif label == 'LOC':
                found_loc = True
                if current_loc and token_index == current_loc[-1][-1] + 1:
                    current_loc[-1][0].append(token)
                    current_loc[-1][-1] = token_index
                else:
                    current_loc.append([[token], token_index])

        if current_rel:
            rel = ' '.join([' '.join(r[0]) for r in current_rel])
            time_str = ' '.join([' '.join(t[0]) for t in current_time]) if current_time else ''
            loc_str = ' '.join([' '.join(l[0]) for l in current_loc]) if current_loc else ''

            # Extract confidence features
            features = extract_confidence_features(sentence, rel, ' '.join(arg1_list), ' '.join(arg2_list), tokens)
            confidence_score = calculate_confidence(features)  # Get confidence score for each extraction
            
            # Standard pattern: ARG1 before REL, ARG2 after REL
            if found_arg1 and found_arg2 and arg1_pos < rel_pos < arg2_pos:
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
            
            # Inverted pattern: ARG2 before REL, ARG1 after REL
            if found_arg1 and found_arg2 and arg2_pos < rel_pos < arg1_pos:
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t{rel}\t{' '.join(arg2_list)}\t{' '.join(arg1_list)}\t{time_str}\t{loc_str}")
            
            # Both ARG1 and ARG2 before REL
            if found_arg1 and found_arg2 and arg1_pos < arg2_pos < rel_pos:
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
            
            # Both ARG1 and ARG2 after REL
            if found_arg1 and found_arg2 and rel_pos < arg1_pos < arg2_pos:
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t{rel}\t{' '.join(arg1_list)}\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
            
            # Loosened: Capture relations with only ARG1 or ARG2
            if found_arg1 and not found_arg2:
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t{rel}\t{' '.join(arg1_list)}\t\t{time_str}\t{loc_str}")
            if found_arg2 and not found_arg1:
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t{rel}\t\t{' '.join(arg2_list)}\t{time_str}\t{loc_str}")
        
        # Handle case where no relations or arguments are found, but time or location might be present
        if not found_rel and not found_arg1 and not found_arg2:
            if found_time or found_loc:
                time_str = ' '.join([' '.join(t[0]) for t in current_time]) if current_time else ''
                loc_str = ' '.join([' '.join(l[0]) for l in current_loc]) if current_loc else ''
                sentence_results.append(f"{sentence}\t{confidence_score:.2f}\t\t\t{time_str}\t{loc_str}")
            else:
                skipped_count += 1
                print(f"Skipped sentence {i}: {sentence} with predicted labels: {predicted_labels}")
        
        results.extend(sentence_results)

    return results, skipped_count


def read_extractions(file_path):
    extractions = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # skip empty lines
                parts = line.split("\t")
                
                sentence = parts[0] 
                confidence = parts[1]  # Ignored confidence value
                rel = parts[2] 
                arg1 = parts[3]  
                arg2 = parts[4] 
        
                if sentence not in extractions:
                    extractions[sentence] = []
                extractions[sentence].append((rel, arg1, arg2))  # store relation, arg1, arg2 as tuple
                
    return extractions


def pos_tagging(nlp, text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def is_passive_voice(pos_tags):
    """ Check if the relation phrase is in passive voice """
    # Look for auxiliary verbs like 'was', 'is', etc., followed by a past participle verb (e.g., 'was eaten')
    aux_verb = None
    for i, (token, pos) in enumerate(pos_tags):
        if pos == 'AUX':
            aux_verb = True
        elif pos == 'VERB' and aux_verb:
            if token.endswith('ed') or token.endswith('en'):
                return True
        aux_verb = False  # Reset after verb
    return False


def verb_object_mismatch(relation_pos_tags, object_phrase):
    """ Check for verb-object mismatch """
    # Expanded list of verbs that typically do not take direct objects
    verbs_that_cannot_have_direct_objects = ["is", "seems", "become", "feel", "exist", "occur", "belong", "appear", "happen"]
    
    # Find verb in the relation and check if it can take a direct object
    for token, pos in relation_pos_tags:
        if pos == 'VERB' and token in verbs_that_cannot_have_direct_objects:
            return True
    return False


def is_non_contiguous(pos_tags):
    """ Check if POS tags for relation are non-contiguous (e.g., interrupted by punctuation or conjunctions) """
    gap_found = False
    for i, (token, pos) in enumerate(pos_tags):
        if pos == 'PUNCT' or pos == 'CCONJ':  # Punctuation or conjunctions indicate a gap
            gap_found = True
        if gap_found and pos == 'VERB':  # If a verb appears after the gap, relation is split
            return True
    return False


def is_imperative_verb(nlp, pos_tags):
    """ Check if the first word is a verb in imperative form """
    if len(pos_tags) > 0 and pos_tags[0][1] == 'VERB':
        # Use dependency parsing to check if this verb is the root (indicating an imperative sentence)
        doc = nlp(pos_tags[0][0])  # Parse the verb with spaCy
        for token in doc:
            if token.dep_ == 'ROOT' and token.tag_ in ['VB', 'VBP']:  # Imperative is often tagged as VB or VBP
                return True
    return False


def is_overspecified(pos_tags):
    """ Check if relation contains more than necessary details (e.g., prepositional phrases, modifiers) """
    overspecification_words = {'ADJ', 'DET', 'ADP', 'AUX'}
    unnecessary_words = []
    
    for tag in pos_tags:
        if tag[1] in overspecification_words:
            unnecessary_words.append(tag)

    # If more than 2 unnecessary words or a long relation, flag it as overspecified
    if len(unnecessary_words) > 2 or len(pos_tags) > 4:
        return True
    return False


def lexical_constraint_filter(relation):
    """ Check if a relation was filtered out due to lexical constraints (rare word, non-standard verb, etc.) """
    rare_words = ['defenestration', 'quixotic', 'anathema']
    filtered_phrases = ['rare_word', 'filtered_relation']
    
    return relation in rare_words or relation in filtered_phrases


def is_more_specific_relation(gold_extraction, predicted_extractions):
    """ Check if the predicted relation is a general form of the gold relation """
    gold_relation = gold_extraction[1]
    for pred in predicted_extractions:
        pred_relation = pred[1]
        # Check if the predicted relation is a generalization of the gold relation (subset of words)
        if gold_relation in pred_relation or pred_relation in gold_relation:
            return True
    return False


def calculate_percentage(errors, total):
    return {error: (count / total) * 100 for error, count in errors.items()}


def categorize_extraction_errors(nlp, gold_extractions, predicted_extractions):
    error_categories = Counter({
        "Correct relation, incorrect arguments": 0,
        "Correct relation, incorrect argument order": 0,
        "Correct relation, missing arguments": 0,
        "N-ary relation": 0,
        "Non-contiguous relation phrase": 0,
        "Imperative verb": 0,
        "Overspecified relation phrase": 0,
        "Other (POS/chunking errors)": 0
    })

    for sentence, gold_set in gold_extractions.items():
        if sentence in predicted_extractions:
            predicted_set = predicted_extractions[sentence]
            for gold in gold_set:
                if gold not in predicted_set:
                    # This handles incorrect extractions when the sentence is present
                    error_type = categorize_error_type(nlp, gold, predicted_set)
                    error_categories[error_type] += 1
        else:
            pass

    return error_categories


def categorize_error_type(nlp, gold_extraction, predicted_extractions):
    gold_relation = gold_extraction[1]
    gold_arg1 = gold_extraction[0]
    gold_arg2 = gold_extraction[2]
    
    for pred in predicted_extractions:
        pred_relation = pred[1]
        pred_arg1 = pred[0]
        pred_arg2 = pred[2]
        
        # Correct relation but incorrect arguments
        if pred_relation == gold_relation:
            # Check for incorrect argument order
            if pred_arg1 != gold_arg1 or pred_arg2 != gold_arg2:
                if pred_arg1 == gold_arg2 and pred_arg2 == gold_arg1:
                    return "Correct relation, incorrect argument order"
                if not pred_arg1 or not pred_arg2:
                    return "Correct relation, missing arguments"
                return "Correct relation, incorrect arguments"
        
        # N-ary relation: more than two arguments
        if len(pred) > 3:
            return "N-ary relation"
        
        # Non-contiguous relation phrase: check if relation is split
        pred_relation_pos = pos_tagging(nlp, pred_relation)
        if is_non_contiguous(pred_relation_pos):
            return "Non-contiguous relation phrase"
        
        # Imperative verb check: if relation starts with a verb in imperative mood
        if is_imperative_verb(nlp, pred_relation_pos):
            return "Imperative verb"
        
        # Overspecified relation phrase: relation contains unnecessary words
        if is_overspecified(pred_relation_pos):
            return "Overspecified relation phrase"
        
        # Check for passive voice
        if is_passive_voice(pred_relation_pos):
            return "Passive voice relation"
        
        # Check for verb-object mismatch
        if verb_object_mismatch(pred_relation_pos, pred_arg2):
            return "Verb-object mismatch"
    
    return "Other (POS/chunking errors)"


def categorize_missed_extractions(gold_extractions, predicted_extractions):
    missed_categories = Counter({
        "Could not identify correct arguments": 0,
        "Relation filtered out by lexical constraint": 0,
        "Identified a more specific relation": 0,
        "POS/chunking error": 0
    })

    for sentence, gold_set in gold_extractions.items():
        if sentence not in predicted_extractions:
            for gold in gold_set:
                if lexical_constraint_filter(gold[1]):
                    missed_categories["Relation filtered out by lexical constraint"] += 1
                elif is_more_specific_relation(gold, predicted_extractions.get(sentence, [])):
                    missed_categories["Identified a more specific relation"] += 1
                else:
                    missed_categories["Could not identify correct arguments"] += 1

    return missed_categories


# Compare and calculate error percentages
def analyze_extractions(nlp, gold_file, extraction_file):
    gold_extractions = read_extractions(gold_file)
    predicted_extractions = read_extractions(extraction_file)

    missed_errors = categorize_missed_extractions(gold_extractions, predicted_extractions)
    total_missed = sum(missed_errors.values())
    missed_percentages = calculate_percentage(missed_errors, total_missed)
    
    incorrect_errors = categorize_extraction_errors(nlp, gold_extractions, predicted_extractions)
    total_incorrect = sum(incorrect_errors.values())
    incorrect_percentages = calculate_percentage(incorrect_errors, total_incorrect)
    
    print("Missed Extractions:")
    for category, percentage in missed_percentages.items():
        print(f"{category}: {percentage:.2f}%")
    
    print("\nIncorrect Extractions:")
    for category, percentage in incorrect_percentages.items():
        print(f"{category}: {percentage:.2f}%")
