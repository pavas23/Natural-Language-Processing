import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string 
import re

from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from collections import Counter
from collections import defaultdict
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
RemoveWords = set(stopwords.words('english'))

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def preprocess_sentences(corpus, min_word_freq=10, min_sentence_len=8):
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
    def __init__(self):
        self.Neuron = 20
        self.lr = 0.0005
        self.Context_Window = 2

        self.X_train = []
        self.y_train = []
        

        self.words = []
        self.word_index = {}
        self.vocab = {}
  
    def InitializeWeights(self,V,data):
        self.V = V
        self.W = np.random.uniform(-0.4, 0.4, (self.V, self.Neuron))
        self.W1 = np.random.uniform(-0.4, 0.4, (self.Neuron, self.V))
          
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
            
    def train(self,mytol,maxepochs=20000):
      	#initialize loss
        self.loss =0
        self.loss1 = 1 #random number 1
        itr=1
        
        while abs(self.loss1 - self.loss)>= mytol and itr <= maxepochs:
            self.loss1 = self.loss
            self.loss = 0
            for j in range(len(self.X_train)):

             		# implementing feedforward 
                self.h = np.dot(self.W.T,self.X_train[j]).reshape(self.Neuron,1)
                self.u = np.dot(self.W1.T,self.h)
                self.y = softmax(self.u)
                
                # implementation of back propogration
                error = self.y - np.asarray(self.y_train[j]).reshape(self.V,1)
                dLdW1 = np.dot(self.h,error.T)
                X = np.array(self.X_train[j]).reshape(self.V,1)
                dLdW = np.dot(X, np.dot(self.W1,error).T)

                self.W1 = self.W1 - self.lr*dLdW1
                self.W = self.W - self.lr*dLdW

                #loss Function
                C = 0
                for m in range(self.V):
                    if(self.y_train[j][m]):
                        self.loss += -1*self.u[m][0]
                        C += 1
                self.loss += C*np.log(np.sum(np.exp(self.u)))
            #Print loss
            # print("epoch ",itr, " loss = ",self.loss)
            #update adaptive alpha
            self.lr *= 1/( (1+self.lr*itr) )
            itr=itr+1
             
    def predict(self,word,number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1


            #prediction = self.feed_forward(X)
            self.h = np.dot(self.W.T,X).reshape(self.Neuron,1)
            self.u = np.dot(self.W1.T,self.h)
            self.y = softmax(self.u)
            prediction=self.y


            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
              
            top_context_words = []
            for k in sorted(output,reverse=True):
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=number_of_predictions):
                    break
      
            return top_context_words
        else:
            print("Word not found")
    
    def get_embedding_matrix(self):
        """
        Returns the word embedding matrix where each row corresponds
        to the embedding of a word in the vocabulary.
        """
        return self.W 
    
    def get_word_embedding(self, word):
        """
        Returns the embedding of the given word.
        """
        if word in self.word_index:
            word_idx = self.word_index[word]
            return self.W[word_idx]  # The embedding for the word
        else:
            print(f"Word '{word}' not in vocabulary.")
            return None


class SkipGramModelNeg(object):
    def __init__(self, negative_samples=5):
        self.Neuron = 20
        self.lr = 0.005
        self.negative_samples = negative_samples
        self.words = []
        self.word_index = {}
        self.V = 0
        self.W = None
        self.W1 = None

    def InitializeWeights(self, V, data):
        self.V = V
        self.W = np.random.uniform(-0.4, 0.4, (self.V, self.Neuron))
        self.W1 = np.random.uniform(-0.4, 0.4, (self.Neuron, self.V))
        self.words = data
        for i, word in enumerate(data):
            self.word_index[word] = i

    def train(self, mytol, maxepochs=20000):
        itr = 1

        while itr <= maxepochs:
            total_loss = 0
            
            for j in range(len(self.X_train)):
                # Feedforward
                h = np.dot(self.W.T, self.X_train[j]).reshape(self.Neuron, 1)
                u = np.dot(self.W1.T, h)
                y_pred = softmax(u)

                # Prepare positive samples
                positive_indices = np.where(self.y_train[j] == 1)[0]

                # Negative sampling
                negative_samples = np.random.choice(
                    [i for i in range(self.V) if i not in positive_indices],
                    self.negative_samples,
                    replace=False
                )

                # Create labels: 1 for positive, 0 for negative
                labels = np.zeros(self.V)
                labels[positive_indices] = 1
                labels[negative_samples] = 0
                
                # Calculate error
                error = y_pred - labels.reshape(self.V, 1)

                # Gradients
                dLdW1 = np.dot(h, error.T)
                X = np.array(self.X_train[j]).reshape(self.V, 1)
                dLdW = np.dot(X, np.dot(self.W1, error).T)

                # Update weights
                self.W1 -= self.lr * dLdW1
                self.W -= self.lr * dLdW

                # Loss calculation
                positive_loss = -np.sum(labels[positive_indices] * np.log(y_pred[positive_indices] + 1e-10))
                negative_loss = -np.sum((1 - labels[negative_samples]) * np.log(1 - y_pred[negative_samples] + 1e-10))
                total_loss += positive_loss + negative_loss

            # Print loss for monitoring
            print("Epoch ", itr, " Loss = ", total_loss)
            
            # Update learning rate
            self.lr *= 1 / (1 + self.lr * itr)
            itr += 1

    def predict(self, word, number_of_predictions):
        if word in self.word_index:
            index = self.word_index[word]
            X = np.zeros(self.V)
            X[index] = 1

            h = np.dot(self.W.T, X).reshape(self.Neuron, 1)
            u = np.dot(self.W1.T, h)
            y_pred = softmax(u)

            output = {y_pred[i][0]: i for i in range(self.V)}
            top_context_words = sorted(output, reverse=True)[:number_of_predictions]
            return [self.words[i] for i in top_context_words]
        else:
            print("Word not found")
            return []

    def get_embedding_matrix(self):
        return self.W 

    def get_word_embedding(self, word):
        if word in self.word_index:
            return self.W[self.word_index[word]]
        else:
            print(f"Word '{word}' not in vocabulary.")
            return None


class CBOWModel(object):
    def __init__(self):
        self.Neuron = 50  # Number of hidden units
        self.lr = 0.005   # Learning rate
        self.Context_Window = 2  # Context window size

        self.X_train = []  # One-hot encoded context words (input)
        self.y_train = []  # One-hot encoded target word (output)

        self.words = []
        self.word_index = {}
        self.vocab = {}

    def softmax(self, z):
        """Softmax function to normalize the output probabilities."""
        exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
        return exp_z / np.sum(exp_z)

    def InitializeWeights(self, V, data):
        """Initialize weights of the neural network."""
        self.V = V  # Vocabulary size
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.Neuron))  # Input -> Hidden
        self.W1 = np.random.uniform(-0.8, 0.8, (self.Neuron, self.V))  # Hidden -> Output

        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
            
    def train(self, mytol, maxepochs=20000):
        # Initialize loss
        self.loss = 0
        self.loss1 = 1  # Random initial number
        itr = 1
        
        while abs(self.loss1 - self.loss) >= mytol and itr <= maxepochs:
            self.loss1 = self.loss
            self.loss = 0
            for j in range(len(self.X_train)):

                # Sum the one-hot encoded context vectors
                context_vector = np.sum(self.X_train[j], axis=0).reshape(self.V, 1)

                # Feedforward: context vector -> hidden -> output
                self.h = np.dot(self.W.T, context_vector).reshape(self.Neuron, 1)  # Input to hidden
                self.u = np.dot(self.W1.T, self.h)  # Hidden to output
                self.y = self.softmax(self.u)  # Softmax output (probability distribution over words)

                # Backpropagation (output -> hidden -> input)
                error = self.y - np.asarray(self.y_train[j]).reshape(self.V, 1)  # Error (predicted vs true target)
                dLdW1 = np.dot(self.h, error.T)  # Gradient for W1
                dLdW = np.dot(context_vector, np.dot(self.W1, error).T)  # Gradient for W

                # Update weights using gradient descent
                self.W1 = self.W1 - self.lr * dLdW1
                self.W = self.W - self.lr * dLdW

                # Loss calculation (cross-entropy loss)
                target_index = np.argmax(self.y_train[j])  # Get the target word's index
                self.loss += -np.log(self.y[target_index][0] + 1e-9)  # To prevent log(0)

            # Print loss
            print(f"epoch {itr}, loss = {self.loss}")

            # Update adaptive learning rate
            self.lr *= 1 / (1 + self.lr * itr)
            itr += 1

    def predict(self, context_words, number_of_predictions):
        # Create a context vector (sum of the one-hot encoded context words)
        context_vector = np.zeros(self.V)
        for word in context_words:
            if word in self.words:
                index = self.word_index[word]
                context_vector[index] = 1
        context_vector = context_vector.reshape(self.V, 1)

        # Feedforward pass to predict the target word
        self.h = np.dot(self.W.T, context_vector).reshape(self.Neuron, 1)  # Context to hidden
        self.u = np.dot(self.W1.T, self.h)  # Hidden to output
        self.y = self.softmax(self.u)  # Softmax output (probability distribution over words)

        # Get the top predictions
        top_indices = np.argsort(-self.y.flatten())[:number_of_predictions]
        top_context_words = [self.words[i] for i in top_indices]
        
        return top_context_words


# importing corpus
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']


corpus = ""
for i in range(5000):
  corpus = corpus +"."+ train_data[i]["text"]

corpus_Set = preprocess_sentences(corpus)

def prepare_data_in_batches(sentences, window_size, vocab, batch_size):
    """
    Generator to prepare context-target word pairs in batches.
    
    Args:
    - sentences: List of tokenized sentences.
    - window_size: Size of the context window.
    - vocab: Dictionary mapping words to indices (vocabulary).
    - batch_size: Number of word-context pairs to return in each batch.
    
    Yields:
    - X_batch: A batch of one-hot encoded target words.
    - y_batch: A batch of one-hot encoded context words.
    """
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


sentences = corpus_Set

vocab = {word: idx for idx, word in enumerate(set([word for sentence in sentences for word in sentence]))}


window_size = 2
batch_size = 1000
V = len(vocab)

model = SkipGramModel()
model.InitializeWeights(V, list(vocab.keys()))
model.vocab = vocab

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    # Generate word-context pairs in batches
    batch_generator = prepare_data_in_batches(sentences, window_size, vocab, batch_size)
    
    for X_batch, y_batch in batch_generator:
        model.X_train = X_batch
        model.y_train = y_batch
        
        # Perform one epoch of training on this batch
        model.train(mytol=1e-4, maxepochs=1)  # Use a small maxepochs value for batch-wise training

    print(f"Finished epoch {epoch + 1}/{num_epochs}")


model1 = SkipGramModelNeg()
model1.InitializeWeights(V, list(vocab.keys()))
model1.vocab = vocab

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    # Generate word-context pairs in batches
    batch_generator = prepare_data_in_batches(sentences, window_size, vocab, batch_size)
    
    for X_batch, y_batch in batch_generator:
        model1.X_train = X_batch
        model1.y_train = y_batch
        
        # Perform one epoch of training on this batch
        model1.train(mytol=1e-4, maxepochs=1)  # Use a small maxepochs value for batch-wise training

    print(f"Finished epoch {epoch + 1}/{num_epochs}")


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


embedding_matrix = model1.get_embedding_matrix()
word_index = model1.word_index  # word to index mapping from the model

#This function create One hot encoding for Input word and the context words
def prepare_test_data(sentences, window_size, vocab):
    """
    Prepare test data from sentences for MRR calculation.

    Args:
    - sentences: List of sentences (each sentence is a string).
    - window_size: The context window size (c).
    - vocab: Set of valid words (usually, your model's vocabulary).

    Returns:
    - test_data: A list of (target_word, [context_words]) tuples.
    """
    test_data = []
    print(vocab)
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


c_t = test_data["text"][:100]

c_t = ""
for i in c_t:
  c_t_t= c_t_t +"."+ i

c_t_t = preprocess_sentences(c_t_t)

test_try = prepare_test_data(c_t_t, 2, vocab)

avg_mrr = calculate_mrr_for_dataset(test_try, embedding_matrix, word_index)
print(f"Mean Reciprocal Rank for test data: {avg_mrr:.4f}")
