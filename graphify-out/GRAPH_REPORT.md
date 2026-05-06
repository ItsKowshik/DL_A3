# Graph Report - A3  (2026-05-04)

## Corpus Check
- 4 files · ~2,861 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 109 nodes · 128 edges · 18 communities detected
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `Transformer` - 21 edges
2. `NoamScheduler` - 7 edges
3. `LabelSmoothingLoss` - 5 edges
4. `Multi30kDataset` - 4 edges
5. `MultiHeadAttention` - 4 edges
6. `PositionalEncoding` - 4 edges
7. `PositionwiseFeedForward` - 4 edges
8. `EncoderLayer` - 4 edges
9. `DecoderLayer` - 4 edges
10. `Encoder` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Label smoothing as in "Attention Is All You Need"      Smoothed target distribut` --uses--> `Transformer`  [INFERRED]
  train.py → model.py
- `Args:             logits : shape [batch * tgt_len, vocab_size]  (raw model outpu` --uses--> `Transformer`  [INFERRED]
  train.py → model.py
- `Run one epoch of training or evaluation.      Args:         data_iter  : DataLoa` --uses--> `Transformer`  [INFERRED]
  train.py → model.py
- `Generate a translation token-by-token using greedy decoding.      Args:` --uses--> `Transformer`  [INFERRED]
  train.py → model.py
- `Evaluate translation quality with corpus-level BLEU score.      Args:         mo` --uses--> `Transformer`  [INFERRED]
  train.py → model.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (13): get_lr_history(), NoamScheduler, Noam Learning Rate Scheduler Reference: "Attention Is All You Need" (Vaswani et, # TODO: Implement the NoamScheduler class below, Noam learning rate scheduler as described in "Attention Is All You Need".      A, # TODO: Store d_model and warmup_steps as instance attributes, # TODO: Call the parent __init__, Compute the Noam scaling factor for the current step.          Returns: (+5 more)

### Community 1 - "Community 1"
Cohesion: 0.17
Nodes (11): make_src_mask(), make_tgt_mask(), model.py — Transformer Architecture Skeleton DA6401 Assignment 3: "Attention Is, # TODO: Task 2.3 — define:, # TODO:instantiate:, # TODO: instantiate:, # TODO: Instantiate, Compute Scaled Dot-Product Attention.          Attention(Q, K, V) = softmax( Q·K (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.18
Nodes (7): Multi30kDataset, Builds the vocabulary mapping for src (de) and tgt (en), including:         <unk, # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent, Convert English and German sentences into integer token lists using         spac, # TODO: Tokenize and convert words to indices, Loads the Multi30k dataset and prepares tokenizers., # TODO: Load dataset, load spacy tokenizers for de and en

### Community 3 - "Community 3"
Cohesion: 0.18
Nodes (6): EncoderLayer, MultiHeadAttention, Multi-Head Attention as in "Attention Is All You Need", §3.2.2.          MultiHe, Args:             query : shape [batch, seq_q, d_model]             key   : shap, Single Transformer encoder sub-layer:         x → [Self-Attention → Add & Norm], Args:             x        : shape [batch, src_len, d_model]             src_mas

### Community 4 - "Community 4"
Cohesion: 0.22
Nodes (6): Full Encoder-Decoder Transformer for sequence-to-sequence tasks.      Args:, Run the full encoder stack.          Args:             src      : Token indices,, Run the full decoder stack and project to vocabulary logits.          Args:, Full encoder-decoder forward pass.          Args:             src      : shape [, Transformer, train.py — Training Pipeline, Inference & Evaluation DA6401 Assignment 3: "Atten

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (6): # TODO: Task 3.3 — implement token-by-token greedy decoding, # TODO: Task 3 — loop test set, decode, compute and return BLEU, # TODO: implement using torch.save({...}, path), # TODO: implement restore logic, # TODO: implement full experiment, # TODO: Task 3.1

### Community 6 - "Community 6"
Cohesion: 0.4
Nodes (3): PositionalEncoding, Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.      Arg, Args:             x : Input embeddings, shape [batch, seq_len, d_model]

### Community 7 - "Community 7"
Cohesion: 0.4
Nodes (3): PositionwiseFeedForward, Position-wise Feed-Forward Network, §3.3:          FFN(x) = max(0, x·W₁ + b₁)·W₂, Args:             x : shape [batch, seq_len, d_model]         Returns:

### Community 8 - "Community 8"
Cohesion: 0.4
Nodes (3): DecoderLayer, Single Transformer decoder sub-layer:         x → [Masked Self-Attn → Add & Norm, Args:             x        : shape [batch, tgt_len, d_model]             memory

### Community 9 - "Community 9"
Cohesion: 0.4
Nodes (3): Encoder, Stack of N identical EncoderLayer modules with final LayerNorm., Args:             x    : shape [batch, src_len, d_model]             mask : shap

### Community 10 - "Community 10"
Cohesion: 0.4
Nodes (3): Decoder, Stack of N identical DecoderLayer modules with final LayerNorm., Args:             x        : shape [batch, tgt_len, d_model]             memory

### Community 11 - "Community 11"
Cohesion: 0.4
Nodes (3): LabelSmoothingLoss, Label smoothing as in "Attention Is All You Need"      Smoothed target distribut, Args:             logits : shape [batch * tgt_len, vocab_size]  (raw model outpu

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (2): load_checkpoint(), Restore model (and optionally optimizer/scheduler) state from disk.      Args:

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (2): Run one epoch of training or evaluation.      Args:         data_iter  : DataLoa, run_epoch()

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (2): greedy_decode(), Generate a translation token-by-token using greedy decoding.      Args:

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (2): evaluate_bleu(), Evaluate translation quality with corpus-level BLEU score.      Args:         mo

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (2): Save model + optimiser + scheduler state to disk.      The autograder will call, save_checkpoint()

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): Set up and run the full training experiment.      Steps:         1. Init W&B:, run_training_experiment()

## Knowledge Gaps
- **43 isolated node(s):** `Loads the Multi30k dataset and prepares tokenizers.`, `Builds the vocabulary mapping for src (de) and tgt (en), including:         <unk`, `Convert English and German sentences into integer token lists using         spac`, `# TODO: Load dataset, load spacy tokenizers for de and en`, `# TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (2 nodes): `load_checkpoint()`, `Restore model (and optionally optimizer/scheduler) state from disk.      Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (2 nodes): `Run one epoch of training or evaluation.      Args:         data_iter  : DataLoa`, `run_epoch()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (2 nodes): `greedy_decode()`, `Generate a translation token-by-token using greedy decoding.      Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (2 nodes): `evaluate_bleu()`, `Evaluate translation quality with corpus-level BLEU score.      Args:         mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (2 nodes): `Save model + optimiser + scheduler state to disk.      The autograder will call`, `save_checkpoint()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (2 nodes): `Set up and run the full training experiment.      Steps:         1. Init W&B:`, `run_training_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Transformer` connect `Community 4` to `Community 1`, `Community 3`, `Community 5`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 17`?**
  _High betweenness centrality (0.325) - this node is a cross-community bridge._
- **Why does `MultiHeadAttention` connect `Community 3` to `Community 1`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `PositionalEncoding` connect `Community 6` to `Community 1`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Transformer` (e.g. with `LabelSmoothingLoss` and `train.py — Training Pipeline, Inference & Evaluation DA6401 Assignment 3: "Atten`) actually correct?**
  _`Transformer` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Loads the Multi30k dataset and prepares tokenizers.`, `Builds the vocabulary mapping for src (de) and tgt (en), including:         <unk`, `Convert English and German sentences into integer token lists using         spac` to the rest of the system?**
  _43 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._