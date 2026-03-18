import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig, GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config, T5ForSequenceClassification, T5Config
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import json
import nltk
from collections import Counter
import torch.nn as nn
import random

def set_seed(seed_value=42):
    """Set random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nltk.download('punkt')
nltk.download('punkt_tab')

def load_jsonl_to_df(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    if 'gold_label_encoded' in df.columns:
        df['gold_label_encoded'] = df['gold_label_encoded'].astype(int)
    return df

nli_df_loaded = load_jsonl_to_df('data.jsonl')
batch_size = 32

# BERT Model
bert_model_name = 'bert-base-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_config = BertConfig.from_pretrained(bert_model_name, num_labels=3, problem_type="single_label_classification")
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, config=bert_config)
bert_model.load_state_dict(torch.load('bert_nli_model.pt', map_location=device))
bert_model.to(device)
bert_model.eval()

# GPT-2 Model
gpt2_model_name = 'gpt2'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_config = GPT2Config.from_pretrained(gpt2_model_name, num_labels=3, problem_type="single_label_classification")
gpt2_model = GPT2ForSequenceClassification.from_pretrained(gpt2_model_name, config=gpt2_config)
gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
gpt2_model.load_state_dict(torch.load('gpt2_nli_model.pt', map_location=device))
gpt2_model.to(device)
gpt2_model.eval()

# RNN Model
all_words_loaded = []
for sentence in pd.concat([nli_df_loaded['sentence1'], nli_df_loaded['sentence2']]).dropna():
    all_words_loaded.extend(nltk.word_tokenize(sentence.lower()))

word_counts_loaded = Counter(all_words_loaded)
vocabulary_loaded = {word: i + 2 for i, (word, _) in enumerate(word_counts_loaded.most_common())} # +2 for <pad> and <unk>
vocabulary_loaded['<pad>'] = 0
vocabulary_loaded['<unk>'] = 1
max_seq_len_rnn = 128

glove_file = 'glove.6B.100d.txt'
embedding_dim_glove = 100

def load_glove_embeddings(file_path, word_index, embedding_dim):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items():
        if i < len(word_index):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
    return torch.tensor(embedding_matrix, dtype=torch.float)

pretrained_weights_rnn = load_glove_embeddings(glove_file, vocabulary_loaded, embedding_dim_glove)

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=vocabulary_loaded['<pad>'])
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocabulary_loaded['<pad>'])

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden_pooled = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_pooled = hidden[-1,:,:]
        return self.dropout(hidden_pooled)


class NLIRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pretrained_embeddings=None):
        super().__init__()
        self.encoder = RNNEncoder(vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, pretrained_embeddings)
        classifier_input_dim = (hidden_dim * 2 if bidirectional else hidden_dim) * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, premise, hypothesis):
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        combined_representation = torch.cat((u, v), dim=1)
        return self.classifier(combined_representation)

VOCAB_SIZE_RNN = len(vocabulary_loaded)
EMBEDDING_DIM_RNN = 100
HIDDEN_DIM_RNN = 256
OUTPUT_DIM_RNN = 3
N_LAYERS_RNN = 2
BIDIRECTIONAL_RNN = True
DROPOUT_RNN = 0.5

rnn_model = NLIRNNModel(
    VOCAB_SIZE_RNN, EMBEDDING_DIM_RNN, HIDDEN_DIM_RNN, OUTPUT_DIM_RNN,
    N_LAYERS_RNN, BIDIRECTIONAL_RNN, DROPOUT_RNN, pretrained_embeddings=pretrained_weights_rnn
)
rnn_model.load_state_dict(torch.load('rnn_nli_model.pt', map_location=device))
rnn_model.to(device)
rnn_model.eval()

# T5 Model
t5_model_name = 't5-small'
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_config = T5Config.from_pretrained(t5_model_name, num_labels=3, problem_type="single_label_classification")
t5_model = T5ForSequenceClassification.from_pretrained(t5_model_name, config=t5_config)
t5_model.load_state_dict(torch.load('t5_nli_model.pt', map_location=device))
t5_model.to(device)
t5_model.eval()

def tokenize_bert(sentence1_list, sentence2_list):
    return bert_tokenizer(sentence1_list, sentence2_list, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

def tokenize_gpt2(sentence1_list, sentence2_list):
    input_texts = [f"{p} {gpt2_tokenizer.eos_token} {h}" for p, h in zip(sentence1_list, sentence2_list)]
    return gpt2_tokenizer(input_texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

def text_to_sequence_rnn(text, vocab, max_len):
    if pd.isna(text) or not isinstance(text, str):
        return [vocab['<pad>']] * max_len
    tokens = nltk.word_tokenize(text.lower())
    sequence = [vocab.get(word, vocab['<unk>']) for word in tokens]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence.extend([vocab['<pad>']] * (max_len - len(sequence)))
    return sequence

def tokenize_t5(sentence1_list, sentence2_list):
    input_texts = [f"mnli premise: {p} hypothesis: {h}" for p, h in zip(sentence1_list, sentence2_list)]
    return t5_tokenizer(input_texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

def evaluate_model(model, tokenizer_func, df, s1_col, s2_col, labels_col, model_type='bert'):
    filtered_df = df.dropna(subset=[s1_col, s2_col, labels_col]).copy()
    if filtered_df.empty:
        return np.nan

    sentence1_data = filtered_df[s1_col].tolist()
    sentence2_data = filtered_df[s2_col].tolist()
    labels_data = filtered_df[labels_col].tolist()

    if model_type == 'rnn':
        input_sequences_1 = torch.tensor([text_to_sequence_rnn(s, vocabulary_loaded, max_seq_len_rnn) for s in sentence1_data], dtype=torch.long)
        input_sequences_2 = torch.tensor([text_to_sequence_rnn(s, vocabulary_loaded, max_seq_len_rnn) for s in sentence2_data], dtype=torch.long)
        dataset = TensorDataset(input_sequences_1, input_sequences_2, torch.tensor(labels_data, dtype=torch.long))
    else:
        tokenized_inputs = tokenizer_func(sentence1_data, sentence2_data)
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        dataset = TensorDataset(input_ids, attention_mask, torch.tensor(labels_data, dtype=torch.long))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'rnn':
                b_seq1, b_seq2, b_labels = batch
                b_seq1, b_seq2, b_labels = b_seq1.to(device), b_seq2.to(device), b_labels.to(device)
                outputs = model(b_seq1, b_seq2)
            else:
                b_input_ids, b_attention_mask, b_labels = batch
                b_input_ids = b_input_ids.to(device)
                b_attention_mask = b_attention_mask.to(device)
                b_labels = b_labels.to(device)
                outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)

            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = torch.argmax(logits, dim=1).flatten()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

    return accuracy_score(true_labels, predictions)

perf_results = []
perturbation_methods = {
    'Original': ('sentence1', 'sentence2'),
    'P1_AdjectiveToRelativeClause': ('sentence1_perturbed', 'sentence2_perturbed'),
    'P2_AppositiveInsertion': ('sentence1_perturbed2', 'sentence2_perturbed2'),
    'P3_AdverbialPhraseInsertion': ('sentence1_perturbed3', 'sentence2_perturbed3')
}

models_to_evaluate = {
    'BERT': {'model': bert_model, 'tokenizer_func': tokenize_bert, 'type': 'bert'},
    'GPT-2': {'model': gpt2_model, 'tokenizer_func': tokenize_gpt2, 'type': 'gpt2'},
    'RNN': {'model': rnn_model, 'tokenizer_func': None, 'type': 'rnn'},
    'T5': {'model': t5_model, 'tokenizer_func': tokenize_t5, 'type': 't5'}
}

for model_name_str, model_info in models_to_evaluate.items():
    print(f"Evaluating {model_name_str}...")
    for p_method_name, (s1_col, s2_col) in perturbation_methods.items():
        accuracy = evaluate_model(model_info['model'], model_info['tokenizer_func'],
                                  nli_df_loaded, s1_col, s2_col, 'gold_label_encoded', model_info['type'])
        perf_results.append({
            'model': model_name_str,
            'perturbation_method': p_method_name,
            'performance': accuracy
        })

perf_df = pd.DataFrame(perf_results)
perf_df.to_csv('perf.csv', index=False)
print("perf.csv created successfully.")

# Complexity Metrics Calculation
complex_results = []
def get_spacy_doc_tree_depth(doc):
    if doc is None or not list(doc.sents):
        return 0
    max_depth = 0
    for token in doc:
        depth = 0
        current_token = token
        while current_token != current_token.head:
            depth += 1
            current_token = current_token.head
        max_depth = max(max_depth, depth)
    return max_depth

def get_max_subj_verb_dep_distance(doc_text, nlp_instance=None):
    if not doc_text or not nlp_instance:
        return np.nan
    doc = nlp_instance(doc_text)
    if not list(doc.sents):
        return np.nan
    max_dist = 0
    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ == 'nsubj':
                subj_idx = token.i
                head_token = token.head
                if head_token != token and head_token.i >= sentence.start and head_token.i < sentence.end:
                    verb_idx = head_token.i
                    dist = abs(subj_idx - verb_idx)
                    max_dist = max(max_dist, dist)
    return max_dist

def get_mean_dep_distance(doc_text, nlp_instance=None):
    if not doc_text or not nlp_instance:
        return np.nan
    doc = nlp_instance(doc_text)
    if not list(doc.sents):
        return np.nan
    all_sentence_mdds = []
    for sentence in doc.sents:
        if not sentence:
            continue
        total_dist = 0
        num_words = 0
        for token in sentence:
            if token.head != token:
                dist = abs(token.i - token.head.i)
                total_dist += dist
                num_words += 1
        if num_words > 0:
            all_sentence_mdds.append(total_dist / num_words)

    if all_sentence_mdds:
        return np.mean(all_sentence_mdds)
    else:
        return np.nan

# Initialize spaCy for complexity calculation
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"spaCy model loading failed for complexity calculation. Error: {e}")
    nlp = None

# Calculate complexity for original sentences
def calculate_original_complexity_metrics(df, nlp_instance):
    if nlp_instance is None: return None, None, None
    df['sentence1_doc_temp'] = df['sentence1'].apply(lambda x: nlp_instance(x) if pd.notna(x) else None)
    df['sentence2_doc_temp'] = df['sentence2'].apply(lambda x: nlp_instance(x) if pd.notna(x) else None)
    
    df['sentence1_dep_tree_depth_temp'] = df['sentence1_doc_temp'].apply(get_spacy_doc_tree_depth)
    df['sentence2_dep_tree_depth_temp'] = df['sentence2_doc_temp'].apply(get_spacy_doc_tree_depth)
    max_dep_tree_mean = df[['sentence1_dep_tree_depth_temp', 'sentence2_dep_tree_depth_temp']].max(axis=1).mean()

    df['sentence1_subj_verb_dep_dist_temp'] = df['sentence1'].apply(lambda x: get_max_subj_verb_dep_distance(x, nlp_instance))
    df['sentence2_subj_verb_dep_dist_temp'] = df['sentence2'].apply(lambda x: get_max_subj_verb_dep_distance(x, nlp_instance))
    max_subj_verb_dep_dist_mean = df[['sentence1_subj_verb_dep_dist_temp', 'sentence2_subj_verb_dep_dist_temp']].max(axis=1).mean()

    df['sentence1_mdd_temp'] = df['sentence1'].apply(lambda x: get_mean_dep_distance(x, nlp_instance))
    df['sentence2_mdd_temp'] = df['sentence2'].apply(lambda x: get_mean_dep_distance(x, nlp_instance))
    max_mdd_mean = df[['sentence1_mdd_temp', 'sentence2_mdd_temp']].max(axis=1).mean()
    df.drop(columns=[col for col in df.columns if col.endswith('_temp')], inplace=True)
    
    return max_dep_tree_mean, max_subj_verb_dep_dist_mean, max_mdd_mean

def calculate_perturbed_complexity_metrics(df, s1_col, s2_col, nlp_instance):
    if nlp_instance is None: return np.nan, np.nan, np.nan
    
    temp_df = df.dropna(subset=[s1_col, s2_col]).copy()
    if temp_df.empty: return np.nan, np.nan, np.nan

    temp_df['sentence1_doc_p'] = temp_df[s1_col].apply(lambda x: nlp_instance(x) if pd.notna(x) else None)
    temp_df['sentence2_doc_p'] = temp_df[s2_col].apply(lambda x: nlp_instance(x) if pd.notna(x) else None)

    temp_df['dep_tree_depth_s1'] = temp_df['sentence1_doc_p'].apply(get_spacy_doc_tree_depth)
    temp_df['dep_tree_depth_s2'] = temp_df['sentence2_doc_p'].apply(get_spacy_doc_tree_depth)
    max_dep_tree_mean = temp_df[['dep_tree_depth_s1', 'dep_tree_depth_s2']].max(axis=1).mean()

    temp_df['subj_verb_dist_s1'] = temp_df[s1_col].apply(lambda x: get_max_subj_verb_dep_distance(x, nlp_instance))
    temp_df['subj_verb_dist_s2'] = temp_df[s2_col].apply(lambda x: get_max_subj_verb_dep_distance(x, nlp_instance))
    max_subj_verb_dep_dist_mean = temp_df[['subj_verb_dist_s1', 'subj_verb_dist_s2']].max(axis=1).mean()

    temp_df['mdd_s1'] = temp_df[s1_col].apply(lambda x: get_mean_dep_distance(x, nlp_instance))
    temp_df['mdd_s2'] = temp_df[s2_col].apply(lambda x: get_mean_dep_distance(x, nlp_instance))
    max_mdd_mean = temp_df[['mdd_s1', 'mdd_s2']].max(axis=1).mean()

    return max_dep_tree_mean, max_subj_verb_dep_dist_mean, max_mdd_mean

# Original complexity metrics
orig_dep_tree_mean, orig_subj_verb_dist_mean, orig_mdd_mean = calculate_original_complexity_metrics(nli_df_loaded, nlp)

complex_results.append({'perturbation_method': 'Original', 'metric_type': 'DepTreeDepth', 'value': orig_dep_tree_mean})
complex_results.append({'perturbation_method': 'Original', 'metric_type': 'SubjVerbDepDist', 'value': orig_subj_verb_dist_mean})
complex_results.append({'perturbation_method': 'Original', 'metric_type': 'MeanDepDistance', 'value': orig_mdd_mean})

# Perturbation 1 complexity metrics
p1_dep_tree_mean, p1_subj_verb_dist_mean, p1_mdd_mean = calculate_perturbed_complexity_metrics(nli_df_loaded, 'sentence1_perturbed', 'sentence2_perturbed', nlp)

complex_results.append({'perturbation_method': 'P1_AdjectiveToRelativeClause', 'metric_type': 'DepTreeDepth', 'value': p1_dep_tree_mean})
complex_results.append({'perturbation_method': 'P1_AdjectiveToRelativeClause', 'metric_type': 'SubjVerbDepDist', 'value': p1_subj_verb_dist_mean})
complex_results.append({'perturbation_method': 'P1_AdjectiveToRelativeClause', 'metric_type': 'MeanDepDistance', 'value': p1_mdd_mean})

# Perturbation 2 complexity metrics
p2_dep_tree_mean, p2_subj_verb_dist_mean, p2_mdd_mean = calculate_perturbed_complexity_metrics(nli_df_loaded, 'sentence1_perturbed2', 'sentence2_perturbed2', nlp)

complex_results.append({'perturbation_method': 'P2_AppositiveInsertion', 'metric_type': 'DepTreeDepth', 'value': p2_dep_tree_mean})
complex_results.append({'perturbation_method': 'P2_AppositiveInsertion', 'metric_type': 'SubjVerbDepDist', 'value': p2_subj_verb_dist_mean})
complex_results.append({'perturbation_method': 'P2_AppositiveInsertion', 'metric_type': 'MeanDepDistance', 'value': p2_mdd_mean})

# Perturbation 3 complexity metrics
p3_dep_tree_mean, p3_subj_verb_dist_mean, p3_mdd_mean = calculate_perturbed_complexity_metrics(nli_df_loaded, 'sentence1_perturbed3', 'sentence2_perturbed3', nlp)

complex_results.append({'perturbation_method': 'P3_AdverbialPhraseInsertion', 'metric_type': 'DepTreeDepth', 'value': p3_dep_tree_mean})
complex_results.append({'perturbation_method': 'P3_AdverbialPhraseInsertion', 'metric_type': 'SubjVerbDepDist', 'value': p3_subj_verb_dist_mean})
complex_results.append({'perturbation_method': 'P3_AdverbialPhraseInsertion', 'metric_type': 'MeanDepDistance', 'value': p3_mdd_mean})

complex_df = pd.DataFrame(complex_results)
complex_df.to_csv('complex.csv', index=False)
print("complex.csv created successfully.")

print("All requested files (data.jsonl, perf.csv, complex.csv) have been generated.")