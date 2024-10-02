import warnings
warnings.filterwarnings("ignore")

import time
import matplotlib.pyplot as plt


from datasets import load_dataset
import numpy as np
import re
from collections import Counter

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']


train_text = train_data["text"]


test_text = test_data["text"]


def preprocess_text(sentences, min_freq=5):
    # Remove non-English characters and words, roman numerals, and less frequent words
    def is_english(word):
        return re.match(r'^[a-zA-Z]+$', word) is not None

    # Tokenizing sentences
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    
    # Flatten list and count word frequencies
    all_words = [word for sentence in tokenized_sentences for word in sentence if is_english(word)]
    
    # Remove infrequent words
    word_freq = Counter(all_words)
    vocab = {word for word, freq in word_freq.items() if freq >= min_freq}

    # Add <UNK> token for out-of-vocabulary words
    vocab.add('<UNK>')

    # Filter sentences
    tokenized_sentences = [
        [word if word in vocab else '<UNK>' for word in sentence] 
        for sentence in tokenized_sentences
    ]
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return tokenized_sentences, word2idx, idx2word


def generate_cbow_batches(sentences, word2idx, context_size=4, batch_size=32):
    while True:
        batch_targets = []
        batch_contexts = []
        for sentence in sentences:
            sentence_idx = [word2idx[word] for word in sentence]
            for i, target in enumerate(sentence_idx):
                context = sentence_idx[max(0, i-context_size): i] + sentence_idx[i+1: min(len(sentence_idx), i+context_size+1)]
                if len(context) == context_size * 2:
                    batch_targets.append(target)
                    batch_contexts.append(context)
                
                if len(batch_targets) == batch_size:
                    yield np.array(batch_contexts), np.array(batch_targets)
                    batch_targets = []
                    batch_contexts = []


def generate_test_batches(sentences, word2idx, context_size=4, batch_size=32):
    while True:
        batch_targets = []
        batch_contexts = []
        for sentence in sentences:
            sentence_idx = [word2idx.get(word, word2idx['<UNK>']) for word in sentence]
            for i, target in enumerate(sentence_idx):
                context = sentence_idx[max(0, i-context_size): i] + sentence_idx[i+1: min(len(sentence_idx), i+context_size+1)]
                if len(context) == context_size * 2:
                    batch_targets.append(target)
                    batch_contexts.append(context)
                
                if len(batch_targets) == batch_size:
                    yield np.array(batch_contexts), np.array(batch_targets)
                    batch_targets = []
                    batch_contexts = []


class CBOWModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.rand(vocab_size, embedding_dim) * 0.01  # Input-Embedding Weights
        self.W2 = np.random.rand(embedding_dim, vocab_size) * 0.01  # Output-Embedding Weights
        # self.context = context
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0)
    
    def forward(self, context_words):
        # Mean of context embeddings (CBOW)
        h = np.mean(self.W1[context_words], axis=0)  # shape: (embedding_dim,)
        u = np.dot(h, self.W2)  # shape: (vocab_size,)
        y_pred = self.softmax(u)  # shape: (vocab_size,)
        return y_pred, h
    
    def backward(self, context_words, target_word, y_pred, h, lr=0.01):
        # Compute gradients
        e = y_pred
        e[target_word] -= 1  # Error
        dW2 = np.outer(h, e)  # Gradient of W2
        dW1 = np.mean(np.dot(self.W2, e), axis=0)  # Gradient of W1

        # Update weights
        self.W1[context_words] -= lr * dW1
        self.W2 -= lr * dW2

    def train(self, sentences, word2idx, epochs=10, batch_size=32, context_size=4, lr=0.01):
        num_batches = len(sentences) // batch_size
        generator = generate_cbow_batches(sentences, word2idx, context_size, batch_size)
        
        for epoch in range(epochs):
            total_loss = 0
            for _ in range(num_batches):
                contexts, targets = next(generator)
                batch_loss = 0
                
                for i in range(batch_size):
                    y_pred, h = self.forward(contexts[i])
                    target_word = targets[i]
                    
                    # Calculate loss (cross-entropy)
                    batch_loss -= np.log(y_pred[target_word])
                    
                    # Backpropagation
                    self.backward(contexts[i], target_word, y_pred, h, lr)
                
                total_loss += batch_loss

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches}")


def mean_reciprocal_rank(predictions, targets):

    reciprocal_ranks = []
    
    for i in range(len(predictions)):
        # Get predicted probabilities for the current query
        pred_probs = predictions[i]
        
        # Sort predictions in descending order and get the indices
        sorted_indices = np.argsort(-pred_probs)
        
        # Find the rank of the true target word
        target_index = targets[i]
        rank = np.where(sorted_indices == target_index)[0][0] + 1  # +1 because rank starts from 1
        
        # Calculate reciprocal rank
        reciprocal_rank = 1 / rank
        reciprocal_ranks.append(reciprocal_rank)
    
    # Calculate Mean Reciprocal Rank
    return np.mean(reciprocal_ranks)


def calculate_mrr(model, sentences, word2idx, context_size=4):
    generator = generate_test_batches(sentences, word2idx, context_size, batch_size=1)
    predictions = []
    targets = []
    
    for _ in range(len(sentences)):
        context, target = next(generator)
        y_pred, _ = model.forward(context[0])
        
        # Store prediction and target for MRR calculation
        predictions.append(y_pred)
        targets.append(target[0])
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate MRR using the custom function
    return mean_reciprocal_rank(predictions, targets)


sentences_train = train_text[:10000]
sentences_test = test_text[:1000]


sentences_train


tokenized_train, word2idx, idx2word = preprocess_text(sentences_train)


word2idx


len(word2idx)


vocab_size = len(word2idx)
embedding_dim = 50  # Example dimension


tokenized_test, _, _ = preprocess_text(sentences_test)


# Variables to store results
window_sizes = [2, 3, 5]  # Different window sizes to test
mrr_scores = []
times_taken = []

# Run the model with different window sizes
for window_size in window_sizes:
    cbow_model = CBOWModel(vocab_size, embedding_dim)

    # Measure time for training
    start_time = time.time()
    cbow_model.train(tokenized_train, word2idx, epochs=10, batch_size=64, context_size=window_size, lr=0.01)
    end_time = time.time()
    
    # Calculate time taken and store it
    time_taken = end_time - start_time
    times_taken.append(time_taken)
    
    # Calculate MRR on test data
    mrr = calculate_mrr(cbow_model, tokenized_test, word2idx)
    mrr_scores.append(mrr)

    print(f"Window size: {window_size}, Time: {time_taken:.2f} seconds, MRR: {mrr:.4f}")

# Plotting Time and MRR vs Window Size
fig, ax1 = plt.subplots()

# Plot Time on left y-axis
ax1.set_xlabel('Window Size (Context Size)')
ax1.set_ylabel('Time (s)', color='tab:red')
ax1.plot(window_sizes, times_taken, color='tab:red', label='Time')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for MRR
ax2 = ax1.twinx()
ax2.set_ylabel('MRR', color='tab:blue')
ax2.plot(window_sizes, mrr_scores, color='tab:blue', label='MRR')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add legends and show the plot
fig.tight_layout()
plt.title("Time and MRR vs Window Size")
plt.show()
