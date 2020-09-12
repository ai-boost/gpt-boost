from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast


def run():
    tb_path = ".\\runs\\novel-gpt-tiny"
    data_path = "E:\\data\\corpus\\gpt-boost\\novel_1G.txt"
    tokenizer_path = "E:\\data\\models\\tokenizer"
    model_path = "E:\\data\\models\\gpt-boost\\novel-gpt-tiny"

    n_head = 8
    n_layer = 1
    n_embd = 512
    n_positions = 256
    vocab_size = 20000

    warmup_steps = 5000
    learning_rate = 1e-4
    num_train_epochs = 10
    finetuning_mode = False
    gradient_accumulation_steps = 4
    per_device_train_batch_size = 48

    print("initializing tokenizer and model...")

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, max_len=n_positions)
    if finetuning_mode:
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        config = GPT2Config(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            vocab_size=vocab_size,
            n_positions=n_positions,
        )
        model = GPT2LMHeadModel(config=config)

    print("model parameters: ", model.num_parameters(), "\nloading datasets...")

    dataset = TextDataset(
        block_size=256,
        tokenizer=tokenizer,
        file_path=data_path,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        fp16=True,
        do_train=True,
        save_total_limit=5,
        logging_dir=tb_path,
        output_dir=model_path,
        prediction_loss_only=True,
        overwrite_output_dir=True,
        dataloader_drop_last=True,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(model_path)


if __name__ == '__main__':
    run()
