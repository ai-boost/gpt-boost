import argparse

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast

from utils.data_loader import TextDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--tb_path", type=str, help="tensorboard log path")
    parser.add_argument("--tokenizer_path", type=str, help="tokenizer path")
    parser.add_argument("--model_config_path", type=str, help="config path")
    parser.add_argument("--data_path", type=str, help="training dataset path")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="warm up steps")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epochs")
    parser.add_argument("--save_total_limit", default=3, type=int, help="save total limit")
    parser.add_argument("--finetuning_mode", default=False, type=bool, help="fine tune or pretrain")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--per_device_train_batch_size", type=int, help="training batch size per device")
    args = parser.parse_args()

    return args


def run():
    args = get_args()

    print("initializing tokenizer and model...")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    if args.finetuning_mode:
        config = GPT2Config.from_pretrained(args.model_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    else:
        config = GPT2Config.from_pretrained(args.model_config_path)
        model = GPT2LMHeadModel(config=config)

    print("model parameters: ", model.num_parameters(), "\nloading datasets...")

    dataset = TextDataset(
        block_size=config.n_positions,
        tokenizer=tokenizer,
        file_path=args.data_path,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        fp16=True,
        do_train=True,
        logging_dir=args.tb_path,
        prediction_loss_only=True,
        overwrite_output_dir=True,
        dataloader_drop_last=True,
        output_dir=args.model_path,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.model_path)


if __name__ == '__main__':
    run()
