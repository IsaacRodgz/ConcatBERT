# ConcatBERT model for multimodal classification with Text and Images

General architecture:

* Text representation: Last BERT 786 dimensional hidden vectors (Taking average of all hidden vectors or taking hidden vector associated with CLS token)
* Image representation: VGG16 4096 dimensional vector feature

Both text and image features are concatenated and passed through:

* MLP which outputs prediction classes.
* Multimodal Gated Layer (based on https://arxiv.org/abs/1702.01992) which weights relevance of each modality and combines them to output prediction classes

Datasets used include:

* Hateful memes detection from Facebook Challenge
* Multimodal IMDb (used plot of movie as text and poster of movie as image)
