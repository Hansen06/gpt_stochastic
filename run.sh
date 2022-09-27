
python train_encoder.py
  --config-name=brownian_bridge
  wandb_settings.exp_dir=tickettalk
  data_params.name=tickettalk
  model_params.latent_dim=32

python run_time_clm.py
  --model_name_or_path gpt2
  --dataset_name tickettalk
  --do_train
  --do_eval
  --per_device_eval_batch_size=1
  --per_device_train_batch_size=1
  --save_total_limit=1
  --load_best_model_at_end=True
  --overwrite_output_dir
  --num_train_epochs=10
  --seed=1
  --encoder_filepath=/epoch=99-step=21999.ckpt
  --latent_dim=32
  --output_dir tickettalk
  --evaluation_strategy=steps
  --eval_steps=1000
  --use_contrastive_embeddings