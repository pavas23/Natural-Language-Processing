import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import nltk
import string 
import re
import random
import time

from nltk.corpus import stopwords
from collections import Counter
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
RemoveWords = set(stopwords.words('english'))

TRAIN_SET_SIZE_SOFTMAX = 1000
TRAIN_SET_SIZE_NEG = 3000

# Helper functions
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess_sentences(corpus, min_word_freq=5, min_sentence_len=7):
    # Remove punctuation and numbers (Roman numerals and others)
    corpus = re.sub(r'\b[MDCLXVI]+\b|\d+', '', corpus)
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the corpus
    sentences = [sent.strip().split() for sent in corpus.lower().split('.')]
    
    # Filter out short sentences
    sentences = [sent for sent in sentences if len(sent) >= min_sentence_len]
    
    # Filter out rare words and short sentences
    word_freq = Counter([word for sent in sentences for word in sent])
    sentences = [[word for word in sent if word_freq[word] >= min_word_freq] for sent in sentences]
    
    return [sent for sent in sentences if len(sent) >= min_sentence_len]


class SkipGramModel(object):
    def __init__(self, embedding_dim=20, learning_rate=0.0005):
        self.Neuron = embedding_dim
        self.lr = learning_rate
        self.initial_lr = learning_rate

        self.X_train = []
        self.y_train = []
        self.words = []
        self.word_index = {}
        self.vocab = {}
  
    def InitializeWeights(self, V, data):
        self.V = V
        self.W = np.random.uniform(-0.4, 0.4, (self.V, self.Neuron))
        self.W1 = np.random.uniform(-0.4, 0.4, (self.Neuron, self.V))
          
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
            
    def train(self, mytol, maxepochs=20000):
        # Initialize loss
        self.loss = 0
        self.loss1 = 1  # Random number to start with
        itr = 1
        
        while abs(self.loss1 - self.loss) >= mytol and itr <= maxepochs:
            self.loss1 = self.loss
            self.loss = 0
            for j in range(len(self.X_train)):
                # Implementing feedforward 
                self.h = np.dot(self.W.T, self.X_train[j]).reshape(self.Neuron, 1)
                self.u = np.dot(self.W1.T, self.h)
                self.y = softmax(self.u)
                
                # Backpropagation
                error = self.y - np.asarray(self.y_train[j]).reshape(self.V, 1)
                dLdW1 = np.dot(self.h, error.T)
                X = np.array(self.X_train[j]).reshape(self.V, 1)
                dLdW = np.dot(X, np.dot(self.W1, error).T)

                self.W1 = self.W1 - self.lr * dLdW1
                self.W = self.W - self.lr * dLdW

                # Loss function (cross-entropy)
                for m in range(self.V):
                    if self.y_train[j][m]:
                        self.loss += -1 * self.u[m][0]
                self.loss += np.log(np.sum(np.exp(self.u)))

            print(f"Loss at itr {itr}: {self.loss}")

            # Update adaptive learning rate
            self.lr = self.initial_lr / (1 + 0.01 * itr)  # 0.01 is decay rate
            itr = itr + 1
             
    def predict(self, word, number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = np.zeros(self.V)
            X[index] = 1

            self.h = np.dot(self.W.T, X).reshape(self.Neuron, 1)
            self.u = np.dot(self.W1.T, self.h)
            self.y = softmax(self.u)
            prediction = self.y

            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
              
            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.words[output[k]])
                if len(top_context_words) >= number_of_predictions:
                    break
      
            return top_context_words
        else:
            print("Word not found")
    
    def get_embedding_matrix(self):
        return self.W 
    
    def get_word_embedding(self, word):
        if word in self.word_index:
            word_idx = self.word_index[word]
            return self.W[word_idx]  # The embedding for the word
        else:
            print(f"Word '{word}' not in vocabulary.")
            return None


class SkipGramModelNeg(object):
    def __init__(self, vocab_size, embedding_dim=20, negative_samples=5, learning_rate=0.0005):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.lr = learning_rate
        
        # Initialize weights for input (W) and output (W_out) layers
        self.W = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))  # Input layer weights
        self.W_out = np.random.uniform(-0.5, 0.5, (embedding_dim, vocab_size))  # Output layer weights
        
        # Initialize word index mapping
        self.word_index = {}

    def train(self, sentences, vocab, window_size=2, epochs=1, batch_size=64):
        # Populate word_index
        self.word_index = {word: idx for idx, word in enumerate(vocab)}
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0

            for sentence in sentences:
                for i, target_word in enumerate(sentence):
                    if target_word not in vocab:
                        continue

                    target_word_idx = self.word_index[target_word]

                    # Context words within the window
                    context_words = []
                    for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                        if j != i and sentence[j] in vocab:
                            context_words.append(sentence[j])

                    for context_word in context_words:
                        context_word_idx = self.word_index[context_word]

                        # Negative sampling (random words from vocabulary)
                        negative_samples = random.sample(
                            [idx for idx in range(self.vocab_size) if idx != target_word_idx],
                            self.negative_samples
                        )

                        # Update model using the target-context pair and negative samples
                        loss = self.train_skipgram_neg_sampling(target_word_idx, context_word_idx, negative_samples)
                        total_loss += loss

            print(f"Loss at epoch {epoch + 1}: {total_loss}")

    def train_skipgram_neg_sampling(self, target_word_idx, context_word_idx, negative_samples):
        # Positive sample (target-context)
        h = self.W[target_word_idx] 
        u_pos = np.dot(h, self.W_out[:, context_word_idx]) 
        pos_loss = -np.log(sigmoid(u_pos)) 

        # Negative samples
        neg_loss = 0
        for neg_word_idx in negative_samples:
            u_neg = np.dot(h, self.W_out[:, neg_word_idx])  
            neg_loss += -np.log(sigmoid(-u_neg)) 

        total_loss = pos_loss + neg_loss

        # Gradients for positive context word
        grad_out_pos = (sigmoid(u_pos) - 1) * h
        grad_in_pos = (sigmoid(u_pos) - 1) * self.W_out[:, context_word_idx]

        for neg_word_idx in negative_samples:
            u_neg = np.dot(h, self.W_out[:, neg_word_idx])  # Calculate u_neg for gradient
            grad_out_neg = (sigmoid(-u_neg)) * h
            grad_in_neg = (sigmoid(-u_neg)) * self.W_out[:, neg_word_idx]

            self.W_out[:, neg_word_idx] -= self.lr * grad_out_neg
            self.W[target_word_idx] -= self.lr * grad_in_neg

        self.W_out[:, context_word_idx] -= self.lr * grad_out_pos
        self.W[target_word_idx] -= self.lr * grad_in_pos

        return total_loss

    def get_embedding_matrix(self):
        return self.W

    def get_word_embedding(self, word):
        if word in self.word_index:
            return self.W[self.word_index[word]]
        else:
            print(f"Word '{word}' not in vocabulary.")
            return None
        

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

corpus_softmax = ""
for i in range(TRAIN_SET_SIZE_SOFTMAX):
  corpus_softmax = corpus_softmax + "." + train_data[i]["text"]


corpus_neg = ""
for i in range(TRAIN_SET_SIZE_NEG):
  corpus_neg = corpus_neg + "." + train_data[i]["text"]


corpus_softmax_set = preprocess_sentences(corpus_softmax)
print("Number of Sentences in softmax corpus set :", len(corpus_softmax_set))

corpus_neg_set = preprocess_sentences(corpus_neg)
print("Number of Sentences in neg corpus set :", len(corpus_neg_set))

def prepare_data_in_batches(sentences, window_size, vocab, batch_size):
    X_batch = []
    y_batch = []
    current_batch_size = 0

    for sentence in sentences:
        for i, target_word in enumerate(sentence):
            if target_word not in vocab:
                continue

            target_word_idx = vocab[target_word]
            X = np.zeros(len(vocab))
            X[target_word_idx] = 1  # One-hot encode the target word

            # Get the context words within the window size
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if j != i and sentence[j] in vocab:
                    context_word_idx = vocab[sentence[j]]
                    y = np.zeros(len(vocab))
                    y[context_word_idx] = 1  # One-hot encode the context word

                    X_batch.append(X)
                    y_batch.append(y)
                    current_batch_size += 1

                    # If the current batch size matches the batch_size, yield the batch
                    if current_batch_size == batch_size:
                        yield np.array(X_batch), np.array(y_batch)
                        X_batch = []
                        y_batch = []
                        current_batch_size = 0

    # Yield the final batch if there are any remaining pairs
    if current_batch_size > 0:
        yield np.array(X_batch), np.array(y_batch)

sentences_softmax = corpus_softmax_set
sentences_neg = corpus_neg_set

vocab_softmax = {word: idx for idx, word in enumerate(set([word for sentence in sentences_softmax for word in sentence]))}
vocab_neg = {word: idx for idx, word in enumerate(set([word for sentence in sentences_neg for word in sentence]))}

window_sizes = [2, 3, 4]
batch_size = 500
num_epochs = 5
tol = 1e-4

def train_skipgram_model(model, window_size, maxepochs_per_batch=1):
    model.InitializeWeights(len(vocab_softmax), list(vocab_softmax.keys()))
    model.vocab = vocab_softmax

    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Generate word-context pairs in batches
        batch_generator = prepare_data_in_batches(sentences_softmax, window_size, vocab_softmax, batch_size)
        
        for X_batch, y_batch in batch_generator:
            model.X_train = X_batch
            model.y_train = y_batch
            
            # Perform one epoch of training on this batch
            model.train(mytol=tol, maxepochs=maxepochs_per_batch)

        print(f"Finished epoch {epoch + 1}/{num_epochs}")
    
    total_time = time.time() - start_time
    
    return model, total_time

model_pos_2, time_pos_2 = train_skipgram_model(SkipGramModel(), window_sizes[0])
model_pos_3, time_pos_3 = train_skipgram_model(SkipGramModel(), window_sizes[1])
model_pos_4, time_pos_4 = train_skipgram_model(SkipGramModel(), window_sizes[2])

num_epochs = 3

model_neg_2 = SkipGramModelNeg(vocab_size=len(vocab_neg), embedding_dim=20, negative_samples=7)
start_time = time.time()
model_neg_2.train(sentences_neg, vocab_neg, window_size=window_sizes[0], epochs=num_epochs)
time_neg_2 = time.time() - start_time

model_neg_3 = SkipGramModelNeg(vocab_size=len(vocab_neg), embedding_dim=20, negative_samples=7)
start_time = time.time()
model_neg_3.train(sentences_neg, vocab_neg, window_size=window_sizes[1], epochs=num_epochs)
time_neg_3 = time.time() - start_time

model_neg_4 = SkipGramModelNeg(vocab_size=len(vocab_neg), embedding_dim=20, negative_samples=7)
start_time = time.time()
model_neg_4.train(sentences_neg, vocab_neg, window_size=window_sizes[2], epochs=num_epochs)
time_neg_4 = time.time() - start_time


def compute_similarity(embedding_matrix, target_vector):
    """
    Compute cosine similarity between the target vector and all word embeddings.
    """
    similarities = cosine_similarity(embedding_matrix, target_vector.reshape(1, -1)).flatten()
    return similarities

def get_rank(similarity_scores, true_index):
    """
    Get the rank of the true context word in the similarity list.
    """
    sorted_indices = np.argsort(-similarity_scores)  # Sort in descending order of similarity
    rank = np.where(sorted_indices == true_index)[0][0] + 1  # Get the rank of the true word
    return rank

def calculate_mrr_for_window(context_indices, target_embedding, embedding_matrix):
    """
    Calculate MRR for a single context window.
    """
    mrr = 0.0
    for context_idx in context_indices:
        similarity_scores = compute_similarity(embedding_matrix, target_embedding)
        rank = get_rank(similarity_scores, context_idx)
        mrr += 1 / rank
    mrr /= len(context_indices)  # Average over the context window
    return mrr


def calculate_mrr_for_dataset(test_data, embedding_matrix, word_index):
    """
    Calculate the overall MRR for the entire test dataset.
    """
    total_mrr = 0.0
    for t in test_data:
        print(t)
        target_word, context_words = t[0], t[1]
        if target_word not in word_index:
            continue  # Skip words that are not in the vocabulary
        
        target_idx = word_index[target_word]
        target_embedding = embedding_matrix[target_idx]
        
        # Get indices for all context words
        context_indices = [word_index[word] for word in context_words if word in word_index]
        
        if len(context_indices) > 0:
            mrr_window = calculate_mrr_for_window(context_indices, target_embedding, embedding_matrix)
            total_mrr += mrr_window
    
    avg_mrr = total_mrr / len(test_data)
    return avg_mrr

# This function create One hot encoding for Input word and the context words
def prepare_test_data(sentences, window_size, vocab):
    test_data = []
    for sentence in sentences:
        
        # Iterate over each word in the sentence
        for i in range(window_size, len(sentence)-window_size):
            target_word = sentence[i]
            if target_word not in vocab:
                continue  # Skip words not in the vocabulary
            
            # Get the context words within the window size
            context_words = []
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if j != i and sentence[j] in vocab:  # Exclude the target word itself
                    context_words.append(sentence[j])
            
            if len(context_words) > 0:
                print(target_word, context_words)
                test_data.append([target_word, context_words])
    
    return test_data

corpus_test = test_data["text"][:1000]
corpus_text_tokenized = ""
for i in corpus_test:
  corpus_text_tokenized = corpus_text_tokenized + "." + i

corpus_text_tokenized = preprocess_sentences(corpus_text_tokenized)

test_context_size_2 = prepare_test_data(corpus_text_tokenized, 2, vocab_softmax)

embedding_matrix_pos_2 = model_pos_2.get_embedding_matrix()
word_index_pos_2 = model_pos_2.word_index  # word to index mapping from the model

avg_mrr_pos_2 = calculate_mrr_for_dataset(test_context_size_2, embedding_matrix_pos_2, word_index_pos_2)
print(f"Mean Reciprocal Rank for test data (Positive, 2): {avg_mrr_pos_2:.4f}")

embedding_matrix_neg_2 = model_neg_2.get_embedding_matrix()
word_index_neg_2 = model_neg_2.word_index  # word to index mapping from the model

avg_mrr_neg_2 = calculate_mrr_for_dataset(test_context_size_2, embedding_matrix_neg_2, word_index_neg_2)
print(f"Mean Reciprocal Rank for test data (Negative, 2): {avg_mrr_neg_2:.4f}")

test_context_size_3 = prepare_test_data(corpus_text_tokenized, 3, vocab_softmax)

embedding_matrix_pos_3 = model_pos_3.get_embedding_matrix()
word_index_pos_3 = model_pos_3.word_index  # word to index mapping from the model

avg_mrr_pos_3 = calculate_mrr_for_dataset(test_context_size_3, embedding_matrix_pos_3, word_index_pos_3)
print(f"Mean Reciprocal Rank for test data (Positive, 3): {avg_mrr_pos_3:.4f}")

embedding_matrix_neg_3 = model_neg_3.get_embedding_matrix()
word_index_neg_3 = model_neg_3.word_index  # word to index mapping from the model

avg_mrr_neg_3 = calculate_mrr_for_dataset(test_context_size_3, embedding_matrix_neg_3, word_index_neg_3)
print(f"Mean Reciprocal Rank for test data (Negative, 3): {avg_mrr_neg_3:.4f}")

test_context_size_4 = prepare_test_data(corpus_text_tokenized, 4, vocab_softmax)

embedding_matrix_pos_4 = model_pos_4.get_embedding_matrix()
word_index_pos_4 = model_pos_4.word_index  # word to index mapping from the model

avg_mrr_pos_4 = calculate_mrr_for_dataset(test_context_size_4, embedding_matrix_pos_4, word_index_pos_4)
print(f"Mean Reciprocal Rank for test data (Positive, 4): {avg_mrr_pos_4:.4f}")

embedding_matrix_neg_4 = model_neg_4.get_embedding_matrix()
word_index_neg_4 = model_neg_4.word_index  # word to index mapping from the model

avg_mrr_neg_4 = calculate_mrr_for_dataset(test_context_size_4, embedding_matrix_neg_4, word_index_neg_4)
print(f"Mean Reciprocal Rank for test data (Negative, 4): {avg_mrr_neg_4:.4f}")

# avg mrr for skip-gram with softmax
avg_mrr__pos = (avg_mrr_pos_2 + avg_mrr_pos_3 + avg_mrr_pos_4)/3

# avg mrr for skip-gram with negative samples
avg_mrr_neg = (avg_mrr_neg_2 + avg_mrr_neg_3 + avg_mrr_neg_4)/3

print("MRR for SkipGram With Softmax: ", avg_mrr__pos)
print("MRR for SkipGram With Negative Samples: ", avg_mrr_neg)

labels = ['Skip-Gram with Softmax', 'Skip-Gram with Negative Sampling']
times = [avg_mrr__pos, avg_mrr_neg]

plt.figure(figsize=(6, 6))
plt.bar(labels, times, color=['blue', 'orange'])

plt.title('Average MRR Comparison')
plt.xlabel('Model Type')
plt.ylabel('Average MRR')
plt.show()

# time taken for training
avg_time_softmax = (time_pos_2 + time_pos_3 + time_pos_4)/3
print("Avg time taken for training skip gram with softmax: ", avg_time_softmax)

avg_time_neg_sampling = (time_neg_2 + time_neg_3 + time_neg_4)/3
print("Avg time taken for training skip gram with negative sampling: ", avg_time_neg_sampling)

labels = ['Skip-Gram with Softmax', 'Skip-Gram with Negative Sampling']
times = [avg_time_softmax, avg_time_neg_sampling]

plt.figure(figsize=(6, 6))
plt.bar(labels, times, color=['blue', 'orange'])

plt.title('Average Training Time Comparison')
plt.xlabel('Model Type')
plt.ylabel('Average Training Time (seconds)')
plt.show()
