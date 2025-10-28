# pylint: disable=line-too-long
r"""Pre-training on SensorLM data.

Example usage:
big_vision.trainers.proj.image_text.siglip \
--config proj/image_text/siglip_sensorlm.py:batch_size=1024 \
--workdir

"""

import big_vision.configs.common as bvcc

sensorlm_data_dir = PATH_TO_DATA_DIR

TOKENS_CAPTION_BANK = {  # actual number of tokens 
    'low_level_caption': 512,
    'middle_level_caption': 512,
    'high_level_summary_caption': 256,
    'high_level_all_caption': 1024,
    'middle_low_level_caption': 1024,
    'high_low_level_caption': 1024,
    'high_middle_level_caption': 512,
    'high_middle_low_level_caption': 1024,
}


def get_config(arg=None):
  """The base configuration."""
  c = bvcc.parse_arg(
      arg,
      res=200,
      mode='xm',
      page_lang='en',
      num_slices=0,
      batch_size=1024,
      lr=5e-4,
      wd=1e-4,
      key_name='high_level_summary_caption',
  )
  c.max_len = TOKENS_CAPTION_BANK[c.key_name]

  c.input = {}
  c.input.data = dict(
      name='sensorlm', split='train', data_dir=sensorlm_data_dir
  )

  c.input.batch_size = c.batch_size
  c.input.shuffle_buffer_size = 50_000

  c.total_examples = 50_000_000
  n_steps = c.total_examples // c.input.batch_size

  tok = lambda inkey: (
      f'tokenize(max_len={c.max_len}, model="c4_en", eos="sticky",'
      f' inkey="{inkey}", pad_value=1)'
  )
  voc = 32_000

  c.input.pp = '|'.join([
      'copy(inkey="input_signal", outkey="image")',
      'decode_sensor',
      f'{tok(c.key_name)}',
      'keep("image", "labels")',
  ])

  c.log_training_steps = 50
  c.ckpt_steps = 500
  # c.keep_ckpt_steps = 20_000

  # Model section
  c.model_name = 'proj.image_text.two_towers'
  c.model = dict(
      image_model='vit', text_model='proj.image_text.text_transformer'
  )
  c.model.image = dict(
      variant='B/10/2',
      pool_type='map',
      head_zeroinit=False,
      scan=True,
  )
  c.model.text = dict(variant='B', vocab_size=voc, scan=True)
  c.model.bias_init = -10.0
  c.model.temperature_init = 10.0
  # Remember to update text_out_dim to match the image embedding size.
  c.model.out_dim = (None, 768)  # (image_out_dim, text_out_dim)

  # FSDP strategy.
  if c.num_slices:
    c.mesh = [('slice', c.num_slices), ('data', -1)]
    c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
    c.sharding_rules = [('act_batch', ('slice', 'data'))]
  else:
    c.mesh = [('data', -1)]
    c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
    c.sharding_rules = [('act_batch', ('data',))]

  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.grad_clip_norm = 1.0

  c.schedule = dict(
      decay_type='rsqrt',
      warmup_steps=0.2 * n_steps,
      cooldown_steps=0.2 * n_steps,
  )

  c.seed = 0

  return c
