import os
import random
import time

from data_modules import get_datamodule
import datasets
from transformers import MarianMTModel, MarianTokenizer
import random

from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
import spacy

# Load German language model for SpaCy
nlp = spacy.load("de_core_news_sm")
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
model.eval()

import pandas as pd
def relabel_dataset():
    original_data = {
    'sentence': ['This is sentence 1', 'Another sentence here', 'And one more sentence','Another sentence here1','Another sentence here2'],
    'label': [0, 1, 2,2,0]
    }
    original_df = pd.DataFrame(original_data)

# Function to augment dataset by removing specific labels
    def augment_dataset(original_df, labels_to_remove):
     # Remove rows with the specified labels
        augmented_df = original_df[~original_df['label'].isin(labels_to_remove)].copy()
        # Re-label remaining rows
        label_mapping = {label: idx for idx, label in enumerate(sorted(set(augmented_df['label'])))}
        augmented_df['label'] = augmented_df['label'].map(label_mapping)
        return augmented_df

    # Specify the labels you want to remove
    labels_to_remove = [1, 2]

    # Augment the dataset
    augmented_df = augment_dataset(original_df, labels_to_remove)

    # Display the augmented dataset
    print("Original Dataset:")
    print(original_df)
    print("\nAugmented Dataset with labels {} removed:".format(labels_to_remove))
    print(augmented_df)


# Function to generate augmented text using BERT with augmentation constraints
def generate_augmented_text(text):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    # Mask multiple tokens in the text
    masked_indices = [i for i, token in enumerate(tokens) if random.random() < 0.7 and token != '[CLS]' and token != '[SEP]']
    masked_tokens = [token for i, token in enumerate(tokens) if i in masked_indices]
    for i in masked_indices:
        tokens[i] = '[MASK]'
    
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids_tensor = torch.tensor([input_ids])
    
    # Predict masked tokens
    with torch.no_grad():
        outputs = model(input_ids_tensor)
        predictions = outputs[0][0]
    
    # Get predicted words for masked tokens
    predicted_words = []
    for i in masked_indices:
        predicted_token_id = torch.argmax(predictions[i]).item()
        predicted_word = tokenizer.convert_ids_to_tokens(predicted_token_id)
        predicted_words.append(predicted_word)
    
    # Replace the masked tokens with predicted words
    augmented_texts = []
    for i, masked_index in enumerate(masked_indices):
        original_token = tokens[masked_index]
        predicted_word = predicted_words[i]
        
        # Apply part-of-speech constraint
        original_pos = nlp(original_token)[0].pos_
        predicted_pos = nlp(predicted_word)[0].pos_
        
        # Apply semantic similarity constraint
        original_similarity = nlp(original_token)[0].similarity(nlp(predicted_word))
        
        if original_pos == predicted_pos and original_similarity > 0.5:
            tokens[masked_index] = predicted_word
            augmented_texts.append(' '.join(tokens))
    
    return augmented_texts
    



def sampler_augmentation():


    # Original text dataset
    text_dataset_german = [
    "Die schnelle braune Füchsin springt über den faulen Hund.",
    "Maschinelles Lernen revolutioniert verschiedene Branchen.",
    "Künstliche Intelligenz hat das Potenzial, das Gesundheitswesen zu transformieren."
    ]

    # Generate 5 augmented texts for each input
    for original_text in text_dataset_german:
        print("Original:", original_text)
        for _ in range(5):
            augmented_texts = generate_augmented_text(original_text)
            for i, augmented_text in enumerate(augmented_texts):
              print(f"Augmented {i + 1}:", augmented_text)
        print()   

if __name__ == "__main__":
    dataset_list = ['Bundestag-v2', 'tagesschau', 'german_argument_mining', 
                    'mlsum', 'hatecheck-german', 'financial_phrasebank_75agree_german', 
                    'x_stance', 'swiss_judgment_prediction', 'miam']
    data_folder = "Bundestag-v2"
    data_dir = os.path.join(os.getcwd(),"cleaned_data")
    # relabel_dataset()

    sampler_augmentation()
    # augment_labels(data_folder,data_dir)