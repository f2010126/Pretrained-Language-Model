from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import os
import torch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def convert_entire_set():
    dataset_name = 'oscar'
    dataset = load_dataset('oscar', 'unshuffled_deduplicated_de')
    text_data = []
    for sample in tqdm(dataset['train']):
        sample = (sample['review_body'] + ' ' +sample['review_title']).replace('\n', '')
        text_data.append(sample)
    # write everything to a single file
    with open(f'data/text_{dataset_name}.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))


def amazon_hf_txt():
    text_data = []
    file_count = 0
    dataset_name = 'amazon_reviews_multi'
    dataset = load_dataset(dataset_name, 'de')

    for sample in tqdm(dataset['train']):
        sample = (sample['review_body'] + ' ' +sample['review_title']).replace('\n', '')
        text_data.append(sample)
        if len(text_data) == 10_000:
            # once we get the 10K mark, save to file
            save_path= os.path.join(os.getcwd(),f'data/{dataset_name}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path,f'text_{file_count}.txt'), 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    # Leftover data
    if len(text_data) > 0:
        with open(f'data/{dataset_name}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))

def build_tokenizer():
    dataset_name = 'amazon_reviews_multi'
    data_path = os.path.join(os.getcwd(), f'data/{dataset_name}')
    paths = [str(x) for x in Path(data_path).glob('**/*.txt')]
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK],'<s>', '<pad>', '</s>', '<unk>', '<mask>'"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(paths, trainer)

    save_path = os.path.join(os.getcwd(), f'data')
    tokenizer.save(os.path.join(save_path, "TinyBERT_de.json"))


def batch_data():
    dataset_name = 'amazon_reviews_multi'
    data_path = os.path.join(os.getcwd(), f'data/{dataset_name}')
    paths = [str(x) for x in Path(data_path).glob('**/*.txt')]
    save_path = os.path.join(os.getcwd(), f'data')
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(save_path, "TinyBERT_de.json"))
    fast_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for path in paths:
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')
        batch = fast_tokenizer(lines, max_length=512, padding='max_length', truncation=True)

    labels = torch.tensor([x.ids for x in batch])
    mask = torch.tensor([x.attention_mask for x in batch])
    # make copy of labels tensor, this will be input_ids
    input_ids = labels.detach().clone()
    # create random array of floats with equal dims to input_ids
    rand = torch.rand(input_ids.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for i in range(input_ids.shape[0]):
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids
        input_ids[i, selection] = 3  # our custom [MASK] token == 3

def train_loop():
    save_path = os.path.join(os.getcwd(), f'data')
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(save_path, "TinyBERT_de.json"))
    print(fast_tokenizer.tokenize("Hallo Welt!"))

if __name__ == '__main__':
    amazon_hf_txt()
    build_tokenizer()
    batch_data()

    batch_data()
   #amazon_hf_txt()

