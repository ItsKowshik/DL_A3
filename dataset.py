"""
dataset.py — Data Loading, Tokenization & Vocabulary
DA6401 Assignment 3: "Attention Is All You Need"

Dataset : Multi30k (de→en)
Source  : https://huggingface.co/datasets/bentrevett/multi30k
Tokenization : spaCy only (de_core_news_sm, en_core_web_sm)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset
import spacy
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

class Vocabulary:
    """
    Token to index mapping.

    Special tokens always occupy fixed indices:
        0: <unk>  1: <pad>  2: <sos>  3: <eos>
    """

    def __init__(self):
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}  # str → idx
        self.itos = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}  # idx → str

    def build(self, token_lists, min_freq: int = 2):
        """
        Build vocab from list-of-token-lists.
        Only tokens appearing >= min_freq times are added.
        """
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        for token, freq in sorted(counter.items()):
            if freq >= min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx]   = token

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, UNK_TOKEN)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def encode(self, tokens):
        """List[str] to List[int]"""
        return [self.lookup_index(t) for t in tokens]

class Multi30kDataset(Dataset):
    def __init__(self, split: str = "train", src_vocab=None, tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.

        Args:
            split     : 'train', 'validation', or 'test'
            src_vocab : Pre-built Vocabulary for German (None = build fresh)
            tgt_vocab : Pre-built Vocabulary for English (None = build fresh)
        """
        self.split = split
        raw = load_dataset("bentrevett/multi30k")
        self.data = raw[split]
        self.spacy_de = spacy.load("de_core_news_sm")   # German (source)
        self.spacy_en = spacy.load("en_core_web_sm")    # English (target)
        self.src_tokens = [self._tokenize_de(ex["de"]) for ex in self.data]
        self.tgt_tokens = [self._tokenize_en(ex["en"]) for ex in self.data]
        if src_vocab is None:
            self.src_vocab = Vocabulary()
            self.src_vocab.build(self.src_tokens, min_freq=2)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary()
            self.tgt_vocab.build(self.tgt_tokens, min_freq=2)
        else:
            self.tgt_vocab = tgt_vocab

        self.src_indices = self.process_data(self.src_tokens, self.src_vocab)
        self.tgt_indices = self.process_data(self.tgt_tokens, self.tgt_vocab)


    def _tokenize_de(self, text: str):
        """German text to list of lowercase tokens via spaCy."""
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def _tokenize_en(self, text: str):
        """English text to list of lowercase tokens via spaCy."""
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]


    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        self.src_vocab = Vocabulary()
        self.src_vocab.build(self.src_tokens, min_freq=2)

        self.tgt_vocab = Vocabulary()
        self.tgt_vocab.build(self.tgt_tokens, min_freq=2)


    def process_data(self, token_lists, vocab: Vocabulary):
        """
        Convert token lists to index tensors.
        Wraps each sentence with <sos> and <eos>.
        """
        processed = []
        for tokens in token_lists:
            indices = [SOS_IDX] + vocab.encode(tokens) + [EOS_IDX]
            processed.append(torch.tensor(indices, dtype=torch.long))
        return processed


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.src_indices[idx], self.tgt_indices[idx]



def collate_fn(batch):
    """
    Pads src and tgt sequences to the max length in the batch.

    Args:
        batch : list of (src_tensor, tgt_tensor) tuples

    Returns:
        src_batch : [batch_size, max_src_len]
        tgt_batch : [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def get_dataloaders(batch_size: int = 128):
    """
    Build train / val / test DataLoaders for Multi30k.

    Vocab built from train split only (no leakage).

    Returns:
        train_loader, val_loader, test_loader,
        src_vocab (Vocabulary), tgt_vocab (Vocabulary)
    """
    # Build vocab from train only
    train_dataset = Multi30kDataset(split="train")

    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    # Val and test reuse train vocab
    val_dataset  = Multi30kDataset(split="validation",
                                   src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset = Multi30kDataset(split="test",
                                   src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab



if __name__ == "__main__":
    print("Loading dataset...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=32)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")
    print(f"Src vocab size: {len(src_vocab)}")
    print(f"Tgt vocab size: {len(tgt_vocab)}")

    # Inspect one batch
    src, tgt = next(iter(train_loader))
    print(f"\nSample batch — src shape: {src.shape}, tgt shape: {tgt.shape}")
    print(f"PAD_IDX={PAD_IDX}, SOS_IDX={SOS_IDX}, EOS_IDX={EOS_IDX}")

    # Decode first sentence back to text
    src_sent = " ".join(src_vocab.lookup_token(i.item()) for i in src[0]
                        if i.item() not in (PAD_IDX,))
    tgt_sent = " ".join(tgt_vocab.lookup_token(i.item()) for i in tgt[0]
                        if i.item() not in (PAD_IDX,))
    print(f"\nDE: {src_sent}")
    print(f"EN: {tgt_sent}")