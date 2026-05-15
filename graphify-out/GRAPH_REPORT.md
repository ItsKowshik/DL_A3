# Graph Report - A3  (2026-05-15)

## Corpus Check
- 15 files · ~91,726 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 279 nodes · 612 edges · 61 communities detected
- Extraction: 45% EXTRACTED · 55% INFERRED · 0% AMBIGUOUS · INFERRED: 336 edges (avg confidence: 0.55)
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
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]

## God Nodes (most connected - your core abstractions)
1. `Transformer` - 79 edges
2. `NoamScheduler` - 54 edges
3. `LabelSmoothingLoss` - 34 edges
4. `PositionwiseFeedForward` - 28 edges
5. `Encoder` - 28 edges
6. `Decoder` - 28 edges
7. `MultiHeadAttention` - 26 edges
8. `PositionalEncoding` - 26 edges
9. `EncoderLayer` - 26 edges
10. `DecoderLayer` - 26 edges

## Surprising Connections (you probably didn't know these)
- `debug_infer.py — diagnose why infer() outputs only periods` --uses--> `Transformer`  [INFERRED]
  debug_infer.py → model.py
- `experiment_2_3.py — Attention Rollout & Head Specialization DA6401 Assignment 3` --uses--> `Transformer`  [INFERRED]
  experiment_2_3.py → model.py
- `Single head heatmap as matplotlib figure → wandb.Image.` --uses--> `Transformer`  [INFERRED]
  experiment_2_3.py → model.py
- `Log one heatmap per head as wandb.Image.` --uses--> `Transformer`  [INFERRED]
  experiment_2_3.py → model.py
- `Log comparison heatmaps for the 4 most interesting heads.` --uses--> `Transformer`  [INFERRED]
  experiment_2_3.py → model.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.15
Nodes (35): DecoderLayerAblation, EncoderLayerAblation, experiment_2_2.py — Ablation: Scaling Factor 1/√dk DA6401 Assignment 3 — W&B Rep, DecoderLayer using MHAWithScalingToggle., Build Transformer with scaling toggle in every attention layer., Build Transformer with scaling toggle baked into every attn layer., Mean grad norm of W_q and W_k across all MHA layers., Compute mean gradient norm of all W_q and W_k weight matrices     across all enc (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (38): experiment_2_1.py — Noam Scheduler vs Fixed Learning Rate DA6401 Assignment 3 —, One epoch of training or evaluation.     Returns (avg_loss, token_accuracy)., run_epoch_with_acc(), run_experiment(), Side-by-side comparison of specific heads for report.     interesting_heads: lis, Compute simple statistics to characterise each head.      Returns dict with per-, Run encoder forward pass and return attention weights from     every head in the, Plot one heatmap per attention head.      Args:         attn_weights : [num_head (+30 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (28): collate_fn(), get_dataloaders(), Multi30kDataset, dataset.py — Data Loading, Tokenization & Vocabulary DA6401 Assignment 3: "Atten, German text → list of lowercase tokens via spaCy., English text → list of lowercase tokens via spaCy., Builds the vocabulary mapping for src (de) and tgt (en), including:         <unk, Convert token lists → index tensors.         Wraps each sentence with <sos> and (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (19): Train transformer with either Noam or fixed LR.      Args:         use_noam : Tr, LabelSmoothingLossTracked, experiment_2_5.py — Decoder Sensitivity: Label Smoothing DA6401 Assignment 3 — W, Returns (avg_loss, perplexity, mean_confidence)., Run one epoch. Returns (avg_loss, avg_perplexity, mean_confidence).     Confiden, Label smoothing loss that also tracks prediction confidence.      Confidence = m, Label smoothing loss that also tracks prediction confidence.      Confidence = m, Args:             logits : [batch * tgt_len, vocab_size]             target : [b (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.27
Nodes (5): build_ablation_transformer(), get_qk_grad_norms(), MHAWithScalingToggle, run_experiment(), scaled_dot_product_attention_ablation()

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (10): analyze_heads(), extract_encoder_attn(), log_head_heatmaps(), log_specialization_heatmaps(), make_heatmap_fig(), experiment_2_3.py — Attention Rollout & Head Specialization DA6401 Assignment 3, Single head heatmap as matplotlib figure → wandb.Image., Log one heatmap per head as wandb.Image. (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.22
Nodes (4): Dataset, debug_infer.py — diagnose why infer() outputs only periods, TransformerWithPE, Transformer

### Community 7 - "Community 7"
Cohesion: 0.25
Nodes (5): [batch, seq, d_model] → [batch, num_heads, seq, d_k], [batch, num_heads, seq, d_k] → [batch, seq, d_model], Args:             query : shape [batch, seq_q, d_model]             key   : shap, Compute Scaled Dot-Product Attention.          Attention(Q, K, V) = softmax( Q·K, scaled_dot_product_attention()

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (3): _MinimalVocab, Lightweight vocabulary: only needs a pre-built stoi dict.     No dependency on d, Download checkpoint, load weights + vocab + spaCy tokenizer.         Architectur

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): Args:             x : Input embeddings, shape [batch, seq_len, d_model]

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Args:             x : shape [batch, seq_len, d_model]         Returns:

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Args:             x    : shape [batch, src_len, d_model]             mask : shap

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Args:             x        : shape [batch, src_len, d_model]             src_mas

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Args:             x        : shape [batch, tgt_len, d_model]             memory

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Args:             x        : shape [batch, tgt_len, d_model]             memory

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Build all model layers.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Download checkpoint, load weights + vocab + spaCy tokenizer.         Architectur

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): German string → list of lowercase tokens via spaCy.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): End-to-end NMT: German string → English string.         Greedy autoregressive de

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Run the full encoder stack.          Args:             src      : Token indices,

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Run the full decoder stack and project to vocabulary logits.          Args:

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Full encoder-decoder forward pass.          Args:             src      : shape [

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Loads the Multi30k dataset and prepares tokenizers.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Builds the vocabulary mapping for src (de) and tgt (en), including:         <unk

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Convert English and German sentences into integer token lists using         spac

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): # TODO: Load dataset, load spacy tokenizers for de and en

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): # TODO: Tokenize and convert words to indices

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Compute the Noam scaling factor for the current step.          Returns:

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Compute learning rates for every param group.          Called internally by PyTo

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): # TODO: Implement the NoamScheduler class below

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): # TODO: Store d_model and warmup_steps as instance attributes

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): # TODO: Call the parent __init__

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): # TODO: Implement and return the Noam scale factor

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): # TODO: Return a list of scaled LRs, one per param group

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Compute Scaled Dot-Product Attention.          Attention(Q, K, V) = softmax( Q·K

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Build a padding mask for the encoder (source sequence).      Args:         src

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Build a combined padding + causal (look-ahead) mask for the decoder.      Args:

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Multi-Head Attention as in "Attention Is All You Need", §3.2.2.          MultiHe

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Args:             query : shape [batch, seq_q, d_model]             key   : shap

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.      Arg

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Args:             x : Input embeddings, shape [batch, seq_len, d_model]

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Position-wise Feed-Forward Network, §3.3:          FFN(x) = max(0, x·W₁ + b₁)·W₂

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Args:             x : shape [batch, seq_len, d_model]         Returns:

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Single Transformer encoder sub-layer:         x → [Self-Attention → Add & Norm]

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Args:             x        : shape [batch, src_len, d_model]             src_mas

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Single Transformer decoder sub-layer:         x → [Masked Self-Attn → Add & Norm

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Args:             x        : shape [batch, tgt_len, d_model]             memory

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Stack of N identical EncoderLayer modules with final LayerNorm.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Args:             x    : shape [batch, src_len, d_model]             mask : shap

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Stack of N identical DecoderLayer modules with final LayerNorm.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Args:             x        : shape [batch, tgt_len, d_model]             memory

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Full Encoder-Decoder Transformer for sequence-to-sequence tasks.      Args:

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Run the full encoder stack.          Args:             src      : Token indices,

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Run the full decoder stack and project to vocabulary logits.          Args:

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Full encoder-decoder forward pass.          Args:             src      : shape [

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): # TODO: Task 2.3 — define:

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): # TODO:instantiate:

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): # TODO: instantiate:

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): # TODO: Instantiate

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): # TODO: implement using torch.save({...}, path)

## Knowledge Gaps
- **108 isolated node(s):** `dataset.py — Data Loading, Tokenization & Vocabulary DA6401 Assignment 3: "Atten`, `Token ↔ index mapping.      Special tokens always occupy fixed indices:`, `Build vocab from list-of-token-lists.         Only tokens appearing >= min_freq`, `List[str] → List[int]`, `Loads the Multi30k dataset and prepares tokenizers.          Args:             s` (+103 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 9`** (2 nodes): `.forward()`, `Args:             x : Input embeddings, shape [batch, seq_len, d_model]`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (2 nodes): `.forward()`, `Args:             x : shape [batch, seq_len, d_model]         Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (2 nodes): `.forward()`, `Args:             x    : shape [batch, src_len, d_model]             mask : shap`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (2 nodes): `.forward()`, `Args:             x        : shape [batch, src_len, d_model]             src_mas`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (2 nodes): `.forward()`, `Args:             x        : shape [batch, tgt_len, d_model]             memory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (2 nodes): `.forward()`, `Args:             x        : shape [batch, tgt_len, d_model]             memory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Build all model layers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Download checkpoint, load weights + vocab + spaCy tokenizer.         Architectur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `German string → list of lowercase tokens via spaCy.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `End-to-end NMT: German string → English string.         Greedy autoregressive de`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Run the full encoder stack.          Args:             src      : Token indices,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Run the full decoder stack and project to vocabulary logits.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Full encoder-decoder forward pass.          Args:             src      : shape [`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Loads the Multi30k dataset and prepares tokenizers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Builds the vocabulary mapping for src (de) and tgt (en), including:         <unk`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Convert English and German sentences into integer token lists using         spac`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `# TODO: Load dataset, load spacy tokenizers for de and en`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `# TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `# TODO: Tokenize and convert words to indices`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Compute the Noam scaling factor for the current step.          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Compute learning rates for every param group.          Called internally by PyTo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `# TODO: Implement the NoamScheduler class below`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `# TODO: Store d_model and warmup_steps as instance attributes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `# TODO: Call the parent __init__`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `# TODO: Implement and return the Noam scale factor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `# TODO: Return a list of scaled LRs, one per param group`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Compute Scaled Dot-Product Attention.          Attention(Q, K, V) = softmax( Q·K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Build a padding mask for the encoder (source sequence).      Args:         src`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Build a combined padding + causal (look-ahead) mask for the decoder.      Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Multi-Head Attention as in "Attention Is All You Need", §3.2.2.          MultiHe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Args:             query : shape [batch, seq_q, d_model]             key   : shap`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.      Arg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Args:             x : Input embeddings, shape [batch, seq_len, d_model]`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Position-wise Feed-Forward Network, §3.3:          FFN(x) = max(0, x·W₁ + b₁)·W₂`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Args:             x : shape [batch, seq_len, d_model]         Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Single Transformer encoder sub-layer:         x → [Self-Attention → Add & Norm]`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Args:             x        : shape [batch, src_len, d_model]             src_mas`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Single Transformer decoder sub-layer:         x → [Masked Self-Attn → Add & Norm`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Args:             x        : shape [batch, tgt_len, d_model]             memory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Stack of N identical EncoderLayer modules with final LayerNorm.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Args:             x    : shape [batch, src_len, d_model]             mask : shap`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Stack of N identical DecoderLayer modules with final LayerNorm.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Args:             x        : shape [batch, tgt_len, d_model]             memory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Full Encoder-Decoder Transformer for sequence-to-sequence tasks.      Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Run the full encoder stack.          Args:             src      : Token indices,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Run the full decoder stack and project to vocabulary logits.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Full encoder-decoder forward pass.          Args:             src      : shape [`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `# TODO: Task 2.3 — define:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `# TODO:instantiate:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `# TODO: instantiate:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `# TODO: Instantiate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `# TODO: implement using torch.save({...}, path)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Transformer` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 8`?**
  _High betweenness centrality (0.282) - this node is a cross-community bridge._
- **Why does `NoamScheduler` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `Vocabulary` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 68 inferred relationships involving `Transformer` (e.g. with `debug_infer.py — diagnose why infer() outputs only periods` and `experiment_2_1.py — Noam Scheduler vs Fixed Learning Rate DA6401 Assignment 3 —`) actually correct?**
  _`Transformer` has 68 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `NoamScheduler` (e.g. with `experiment_2_1.py — Noam Scheduler vs Fixed Learning Rate DA6401 Assignment 3 —` and `One epoch of training or evaluation.     Returns (avg_loss, token_accuracy).`) actually correct?**
  _`NoamScheduler` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `LabelSmoothingLoss` (e.g. with `experiment_2_1.py — Noam Scheduler vs Fixed Learning Rate DA6401 Assignment 3 —` and `One epoch of training or evaluation.     Returns (avg_loss, token_accuracy).`) actually correct?**
  _`LabelSmoothingLoss` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `PositionwiseFeedForward` (e.g. with `MHAWithScalingToggle` and `EncoderLayerAblation`) actually correct?**
  _`PositionwiseFeedForward` has 22 INFERRED edges - model-reasoned connections that need verification._