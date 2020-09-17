# windows powershell
python -m train.train `
--model_path E:\\data\\models\\gpt-boost\\novel-gpt-tiny-memory `
--tb_path .\\runs\\novel-gpt-tiny-memory `
--tokenizer_path E:\\data\\models\\tokenizer `
--model_config_path .\\config\\model_config\\config_tiny.json `
--data_path E:\\data\\corpus\\gpt-boost\\all.txt `
--training_mode 0 `
--gradient_accumulation_steps 1 `
--per_device_train_batch_size 12 `
--num_train_epochs 10 `
--warmup_steps 10000 `
--logging_steps 500 `
--save_steps 500 `
--learning_rate 1e-4
