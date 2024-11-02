# Open Information Extraction and Word2Vec

## Open Information Extraction

Open Information Extraction (OpenIE) is a crucial task in natural language processing (NLP) that involves ```extracting tuples``` of structured information from ```unstructured text```. These tuples are typically in the form ```〈subject, relation, object, time, location>```, with the goal of capturing the key relationships and entities within a sentence.

This task presents a supervised learning approach to OpenIE, where the goal is to design a model that extracts multiple relation tuples from a sentence. Sentences may yield more than one extraction depending on the presence of multiple relationships or entities.

### Word Embeddings

We have used BERT (Bidirectional Encoder Representations from Transformers) for generating embeddings for openIE. Unlike traditional embeddings like Word2Vec or GloVe, which generate a single vector for each word regardless of context, BERT captures the meaning of a word in relation to its surrounding words.

![bert](https://github.com/user-attachments/assets/0787093d-44c0-4dea-844f-ad462fce7173)

BERT embeddings are often fine-tuned for specific tasks, and in OpenIE, BERT can be used to generate token-level embeddings, which are then passed through models like BiLSTM-CRF to label tokens appropriately. This results in more accurate and contextsensitive extractions.

### BiLSTM Model

A ```BiLSTM (Bidirectional Long Short-Term Memory)``` model is an extension of the LSTM architecture designed to capture dependencies in both directions within sequential data. LSTMs, by design, are capable of ```learning long-range dependencies``` by using a memory cell and gating mechanisms (input, output, and forget gates) to control the flow of information, making them highly effective in handling sequential data with varying time dependencies, such as text or speech.

The architecture consists of an input layer ```(usually embeddings like BERT)```, the ```BiLSTM layers```, and often a ```final layer (like CRF)``` for sequence labeling or classification. This combined architecture excels at handling complex dependencies within the data, improving the model’s ability to generate accurate predictions.

A ```Conditional Random Field (CRF)``` layer is often added on top of a BiLSTM model in sequence labeling tasks to improve the overall prediction by considering the dependencies between output labels. While the BiLSTM layer captures context from both directions of the input sequence, the CRF layer helps ensure that the predicted sequence of labels is globally optimal by considering relationships between neighboring labels.

<p align="center">
<img src="https://github.com/user-attachments/assets/3199113e-e78e-4826-9816-d97a3332ebb1" height="350" width="500" align="centre">
</p>

## Word2Vec

Word2Vec is a popular technique for generating word embeddings, which are dense vector representations of words in a continuous vector space. Word2Vec comes in two primary models: ```Skip-Gram``` and ```Continuous Bag of Words (CBOW)```. In the Skip-Gram model, the task is to predict the surrounding context words given a target word, while in CBOW, the goal is to predict the target word based on its surrounding words. Both models use a ```neural network with a single hidden layer```.

### Skip-Gram

Skip-Gram works by maximizing the probability of correctly predicting the surrounding words within a certain window size for each target word. A sliding window is moved across the text, and for each word (target), the model attempts to predict nearby words. The training objective is to adjust the word vectors in such a way that words occurring in similar contexts are close to each other in the vector space.

<p align="center">
<img src="https://github.com/user-attachments/assets/8e3e78ad-8c1a-445b-b3db-81c291ee74d5" height="350" width="350" align="centre">
</p>

### CBOW

CBOW works by taking the average of the vectors for the surrounding words (context) and using that to predict the vector for the target word. This method aims to maximize the likelihood of the target word appearing in the given context. Unlike Skip-Gram, which is more focused on predicting multiple context words for a single target, CBOW is designed to predict the target word from the entire context.

<p align="center">
<img src="https://github.com/user-attachments/assets/02a28b5d-eb55-456c-921d-c0247f2b8aa4" height="350" width="350" align="centre">
</p>







