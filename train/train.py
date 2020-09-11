from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import TextDataset


def run():
    model_path = "E:\\data\\models\\gpt-boost\\gpt-2-layer-memory"
    data_path = "E:\\data\\corpus\\gpt-boost\\novel.txt"
    tokenizer_path = ".\\tokenizer"
    tb_path = ".\\runs\\gpt-2-layer-memory"

    n_embd = 512
    n_head = 8
    n_layer = 2
    n_positions = 256
    vocab_size = 20000
    finetuning_mode = False

    learning_rate = 1e-4
    gradient_accumulation_steps = 1
    num_train_epochs = 10
    per_device_train_batch_size = 1

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, max_len=256)

    if finetuning_mode:
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        from transformers import GPT2Config

        config = GPT2Config(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_positions=n_positions,
            vocab_size=vocab_size
        )
        model = GPT2LMHeadModel(config=config)

    # %%

    model.num_parameters()

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=256,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        no_cuda=False,
        do_train=True,
        fp16=True,
        logging_dir=tb_path,
        logging_steps=100,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        warmup_steps=10000,
        save_steps=1000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model(model_path)


if __name__ == '__main__':
    run()
