import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from datetime import datetime

MODEL_NAME = "gpt2-p10k-cossine"
DATABASE_NAME = "prefix10k.txt"
RUN_NAME = str(datetime.now().month) + "/" + str(datetime.now().day) + "/" + str(datetime.now().hour)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

device = torch.device("cuda:0")


model_checkpoint="gpt2-large"

dataset = load_dataset("augustocsc/prefix_expressions", data_files=DATABASE_NAME)['train'].train_test_split(train_size=.8, test_size=.2)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

#block_size = tokenizer.model_max_length
block_size = 128

cores = multiprocessing.cpu_count()

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=300,
    num_proc=cores,
)


model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
#this need bc we added tokens (bos_token, etc).
model.resize_token_embeddings(len(tokenizer))


torch.cuda.empty_cache()
training_args = TrainingArguments(
                            num_train_epochs=10,
                            #max_steps=1000,
                            output_dir=MODEL_NAME,
                            resume_from_checkpoint=True,
                            run_name=RUN_NAME,
                            seed=42,
                            save_steps=1000,
                            save_total_limit=2,
                            optim= "adamw_hf",
                            learning_rate=5e-5,
                            weight_decay=0.01,
                            lr_scheduler_type = "cosine",
                            evaluation_strategy="steps",
                            eval_steps=200,
                            per_device_train_batch_size=64,
                            per_device_eval_batch_size=64,
                            auto_find_batch_size=True,
                            push_to_hub=True,
                            hub_strategy = "checkpoint",
                            report_to="wandb",
                            overwrite_output_dir = True,
                            )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets['train'],
    eval_dataset=lm_datasets['test'],
)

trainer.train()
trainer.push_to_hub()
tokenizer.save_pretrained(MODEL_NAME, push_to_hub=True)