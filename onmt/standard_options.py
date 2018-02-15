from collections import namedtuple
import torch

USE_CUDA = torch.cuda.is_available()


'''
This file will store the standard options used by openNMT-py in a dictionary

'''

stdOptions = {

    # Model options
    'model_type':'text', # Type of encoder to use. Options are [text|img]

    # Embedding Options
    'word_vec_size':-1,       # Word embedding for both.
    'src_word_vec_size':500, # Src word embedding sizes
    'tgt_word_vec_size':500, # Tgt word embedding sizes
    'feat_vec_size':-1,      # If specified, feature embedding sizes will be set to this. Otherwise, feat_vec_exponent
                             # will be used.
    'feat_merge':'concat',   # Merge action for the features embeddings
    'feat_vec_exponent':0.7, # When using -feat_merge concat, feature embedding sizes will be set to
                             # N^feat_vec_exponent where N is the number of values the feature takes.
    'position_encoding':False, #Use a sin to mark relative words positions.
    'share_decoder_embeddings':False, #Share the word and out embeddings for decoder.

    # RNN Options
    'encoder_type':'rnn', # Type of encoder layer to use. Choices: ['rnn', 'brnn', 'mean', 'transformer']
    'decoder_type':'rnn', # Type of decoder layer to use. Choices: ['rnn', 'transformer']
    'layers':-1,           # Number of layers in enc/dec.
    'enc_layers':2,       # Number of layers in the encoder
    'dec_layers':2,       # Number of layers in the decoder
    'rnn_size':500,       # Size of LSTM hidden states
    'input_feed':1,       # Feed the context vector at each time step as additional input (via concatenation
                          # with the word embeddings) to the decoder.
    'rnn_type':'LSTM',    # The gate type to use in the RNNs. Choices: ['LSTM', 'GRU']
    'brnn':False,         # Deprecated, use `encoder_type`.
    'brnn_merge':'concat',# Merge action for the bidir hidden states
    'context_gate':None,  # Type of context gate to use. Do not select for no context gate.
                          # Choices: ['concat', 'sum']

    # Attention options

    'global_attention':'general', # The attention type to use: dotprot or general (Luong) or MLP (Bahdanau)
                                  # Choices: ['dot', 'general', 'mlp']

    # Genenerator and loss options.

    'copy_attn':False,            # Train copy attention layer
    'copy_attn_force':False,      # When available, train to copy.
    'coverage_attn':False,        # Train a coverage attention layer.
    'lambda_coverage':1,           # Lambda value for coverage.

    # Training Options

    'save_model':'model', # Model filename (the model will be saved as <save_model>_epochN_PPL.pt where PPL is the
                          # validation perplexity
    'train_from':'',      # If training from a checkpoint then this is the path to the pretrained model's state_dict.

    # GPU options

    'gpuid':[0],  # Use CUDA on the listed devices.
    'seed':-1,    # Random seed used for the experiments reproducibility.

    # Init options

    'start_epoch':1,  # The epoch from which to start
    'param_init':0.1, # Parameters are initialized over uniform distribution with support (-param_init, param_init).
                      # Use 0 to not use initialization

    # Pretrained word vectors

    'pre_word_vecs_enc':None, # If a valid path is specified, then this will load pretrained word embeddings
                              # on the encoder side. See README for specific formatting instructions.
    'pre_word_vecs_dec':None, # If a valid path is specified, then this will load pretrained word embeddings
                              # on the decoder side. See README for specific formatting instructions.

    # Fixed word vectors

    'fix_word_vecs_enc':False, # Fix word embeddings on the encoder side
    'fix_word_vecs_dec':False, # Fix word embeddings on the decoder side

    # Optimization options

    'batch_size':64,
    'max_generator_batches':32, # Maximum batches of words in a sequence to run the generator on in parallel.
                                # Higher is faster, but uses more memory.
    'epochs':13,                # Number of training epochs
    'optim':'sgd',              # Optimization method. Choices: ['sgd', 'adagrad', 'adadelta', 'adam']
    'max_grad_norm':5.0,        # If the norm of the gradient vector exceeds this, renormalize it to have the norm
                                # equal to max_grad_norm
    'dropout':0.3,              # Dropout probability; applied in LSTM stacks.
    'truncated_decoder':0,      # Truncated bptt

    #Learning rate

    'learning_rate':1.0,        # Starting learning rate. If adagrad/adadelta/adam is used, then this is the global
                                # learning rate. Recommended settings: sgd = 1, adagrad = 0.1,
    'learning_rate_decay':0.5, # If update_learning_rate, decay learning rate by this much if (i) perplexity does
                                # not decrease on the validation set or (ii) epoch has gone past start_decay_at
    'start_decay_at':8,         # Start decaying every epoch after and including this epoch
    'start_checkpoint_at':0,    # Start checkpointing every epoch after and including this epoch
    'decay_method': '',         # Use a custom decay rate. Choices: ['noam']
    'warmup_steps': 4000,       # Number of warmup steps for custom decay.
    'report_every': 50,         # Print stats at this interval.
    'exp_host':'',              # Send logs to this crayon server.
    'exp': ''                   # Name of the experiment for logging.

}

# the standard preprocessing options to use with openNMT-py dataset parser
standardPreProcessingOptions = {

    # Dictionary options
    'src_vocab_size': 50000,  # Size of the source vocabulary
    'tgt_vocab_size': 50000,  # Size of the target vocabulary
    'src_words_min_frequency': 0,
    'tgt_words_min_frequency': 0,

    # Truncation options
    'src_seq_length': 50,  # Maximum source sequence length
    'src_seq_length_trunc': 0,  # Truncate source sequence length.
    'tgt_seq_length': 50,  # Maximum target sequence length to keep
    'tgt_seq_length_trunc': 0,  # Truncate target sequence length.

    # Data processing options

    'shuffle': 1,  # Shuffle data
    'lower': True,  # lowercase data

    # Options most relevant to summarization

    'dynamic_dict': False,  # Create dynamic dictionaries
    'share_vocab': False  # Share source and target vocabulary
}

standardTranslationOptions = {

    'tgt':None,            # True target sequence (optional)
    'output':'pred.txt',   # Path to output the predictions (each line will be the decoded sequence)
    'beam_size':5,
    'batch_size':1,
    'max_sent_length':100,
    'replace_unk':True,    # Replace the generated UNK tokens with the source token that had highest attention weight.
                           # If phrase_table is provided, it will lookup the identified source token and
                           # give the corresponding target token. If it is not provided (or the identified source token
                           # does not exist in the table) then it will copy the source token
    'verbose':False,       # Print scores and predictions for each sentence'
    'attn_debug':False,    # Print best attn for each word
    'dump_beam':'',        # File to dump beam information to.
    'n_best':1,            # If verbose is set, will output the n_best decoded sentences
    'gpu':0,               # Device to run on
    'dynamic_dict':False,   # Create dynamic dictionaries
    'share_vocab':False,    # Share source and target vocabulary
    'cuda':USE_CUDA
}




