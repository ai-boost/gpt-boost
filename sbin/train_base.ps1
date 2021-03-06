# windows powershell
Start-Transcript

python -m train.train `
--model_path E:\\data\\models\\gpt-boost\\novel-gpt-base `
--tb_path .\\runs\\novel-gpt-base-finetune-2 `
--tokenizer_path E:\\data\\models\\tokenizer `
--model_config_path .\\config\\model_config\\config_base.json `
--data_path E:\\data\\corpus\\gpt-boost\\all.txt `
--training_mode 1 `
--gradient_accumulation_steps 16 `
--per_device_train_batch_size 12 `
--num_train_epochs 10 `
--warmup_steps 10000 `
--logging_steps 50 `
--save_steps 50 `
--learning_rate 1e-4

Stop-Transcript
