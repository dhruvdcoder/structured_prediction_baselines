local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'ontonotes';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = 130;
local transformer_model = 'bert-base-uncased';
local transformer_hidden_dim = 768;
//local max_length = 512;

local attention_dropout = std.parseJson(std.extVar('attention_dropout_10x'))/10.0;
local attention_dim = std.parseJson(std.extVar('attention_dim'));
local ff_activation = 'softplus';
local cross_entropy_loss_weight = 1;
local score_loss_weight = std.parseJson(std.extVar('score_loss_weight'));
local ff_weight_decay = std.parseJson(std.extVar('weight_decay'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local task_nn = {
  type: 'sequence-tagging',
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer',  // we don't use mismatched because that is what allennlp srl model does
        model_name: transformer_model,
        //max_length: max_length,
      },
    },
  },
};

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'srl',
    bert_model_name: transformer_model,
    //[if test == '1' then 'max_instances']: 100,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),
  vocabulary: {
    type: 'from_files',
    directory: data_dir + '/' + dataset_metadata.dir_name + '/' + 'bert_vocab',
  },
  // Model
  model: {
    type: 'seal-srl',
    label_encoding: 'BIO',
    using_bert_encoder: true,
    decode_on_wordpieces: true,
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    task_nn: task_nn,
    inference_module: {
      type: 'sequence-tagging-inference-net-normalized',
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'sequence-tagging-score-loss', // jy: fix here.. 
            log_key: 'neg.nce_score',
            normalize_y: true,
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'sequence-tagging-masked-cross-entropy',
            log_key: 'ce',
            reduction: 'none',
            normalize_y: false,
          },
        ],
        loss_weights: [score_loss_weight, cross_entropy_loss_weight],
        reduction: 'mean',
      },
    },
    oracle_value_function: { type: 'manhattan', differentiable: true},
    score_nn: {
      type: 'sequence-tagging',
      task_nn: task_nn,
      global_score: {
        type: 'self-attention-full-sequence',
        num_heads: 1,
        num_tags: num_labels,
        attention_dim: attention_dim,
        dropout: attention_dropout,
      },
    },
    loss_fn: {
      type: 'seqtag-nce-ranking-with-discrete-sampling',
      reduction: 'mean',
      log_key: 'seq_nce_loss',
      num_samples: 10,
    },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 32,  // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['tokens'],
    },
    //max_instances_in_memory: if test == '1' then 10 else 1000,
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 25,
    grad_norm: { task_nn: 1.0 },
    patience: 4,
    validation_metric: '+f1-measure-overall',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 2,
        verbose: true,
      },
    },
    optimizer: {
      optimizers: {
        task_nn:
          {
            lr: 0.00001,
            weight_decay: ff_weight_decay,
            type: 'huggingface_adamw',
          },
        score_nn: {
          lr: 0.00001,
          weight_decay: ff_weight_decay,
          type: 'adamw',
        },
      },
    },
    checkpointer: {
      keep_most_recent_by_count: 1,
    },
    callbacks: [
      'track_epoch_callback',
      'slurm',
    ] + (
      if use_wandb then [
        {
          type: 'wandb_allennlp',
          should_log_parameter_statistics: false,
          sub_callbacks: [{ type: 'log_best_validation_metrics', priority: 100 }],
          save_model_archive: false,
          watch_model: false,
        },
      ]
      else []
    ),
    inner_mode: 'score_nn',
    num_steps: { task_nn: 1, score_nn: 1 },
  },
}
