# windows powershell
python -m train.train `
--model_path E:\\data\\models\\gpt-boost\\novel-gpt-tiny `
--tb_path .\\runs\\novel-gpt-tiny `
--tokenizer_path E:\\data\\models\\tokenizer `
--model_config_path .\\config\\model_config\\config_tiny.json `
--data_path E:\\data\\corpus\\gpt-boost\\all.txt `
--training_mode 0 `
--gradient_accumulation_steps 4 `
--per_device_train_batch_size 48 `
--num_train_epochs 10 `
--warmup_steps 10000 `
--learning_rate 1e-4
