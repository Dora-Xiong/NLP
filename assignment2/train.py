import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import wandb
import numpy as np
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import adapters
from adapters import AdapterArguments, AdapterTrainer, setup_adapter_training
from dataHelper import get_dataset

# Initialize logging
logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"] = "classification"  
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the get_dataset function in dataHelper.py)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length; sequences longer than this will be truncated."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Pad all samples to max_seq_length if True; dynamically pad if False."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from Huggingface."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if different from model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if different from model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store pretrained models downloaded from Huggingface."}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Use fast tokenizer if True."}
    )
    model_revision: str = field(
        default="main", metadata={"help": "Specific model version to use, like a branch or tag name."}
    )

def main():
    # Initialize arguments and parse
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Initialize logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Detect last checkpoint if resuming
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue training."
            )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load dataset
    try:
        raw_datasets = get_dataset(data_args.dataset_name, sep_token="<sep>")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    label_list = raw_datasets["train"].unique("label")
    num_labels = len(label_list)

    # Load model configuration, tokenizer, and model
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    if adapter_args.train_adapter:
        adapters.init(model)

    # Prepare dataset and datacollator
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    processed_datasets = raw_datasets.map(
        lambda e: tokenizer(e["text"], padding=padding, max_length=max_seq_length, truncation=True), batched=True
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    data_collator = (
        default_data_collator if data_args.pad_to_max_length else DataCollatorWithPadding(tokenizer, padding="longest")
    )

    # Define metrics
    accuracy_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    micro_f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)
    macro_f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=p.label_ids),
            "micro_f1": micro_f1_metric.compute(predictions=preds, references=p.label_ids, average="micro"),
            "macro_f1": macro_f1_metric.compute(predictions=preds, references=p.label_ids, average="macro"),
        }
        
    # Initialize adapter
    setup_adapter_training(model, adapter_args, data_args.dataset_name)

    # Initialize trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()
        
    # Evaluation
    if training_args.do_eval:
        trainer.evaluate()
        
    wandb.finish()

if __name__ == "__main__":
    main()
