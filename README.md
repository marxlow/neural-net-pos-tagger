# neural-net-pos-tagger
A POS tagger of a word sequence that is trained with CNN and RNN using character and word level embeddings

# How to run:

Train the Model:
`python3.5 build_tagger.py sents.train model-file`

Run the Model against the Dev-set with the trained Model:
`python3.5 run_tagger.py sents.test model-file sents.out`

Get results of the Model against the Dev-set:
`python3.5 eval.py sents.out sents.answer`
