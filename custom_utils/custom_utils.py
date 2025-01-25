import pandas as pd
from PIL import Image
from datasets import Dataset, Image, DatasetDict
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor,
    BeitForImageClassification,
    BeitImageProcessor,
    MobileViTForImageClassification,
    MobileViTImageProcessor
)
from torchvision.transforms import (
    Normalize,
    ToTensor,
    Resize,
    CenterCrop,
    Compose,
)
import torch
from transformers import TrainingArguments, Trainer
import evaluate


def create_dataset(test_ids: list[int], df: pd.DataFrame) -> tuple[DatasetDict, Dataset]:
    test_ids_set = set(test_ids)
    all_indices = set(df.index)
    train_ids = list(all_indices - test_ids_set)
    
    train_dataset = Dataset.from_dict({
        "image": df.iloc[train_ids, 0].to_list(),
        "label": df.iloc[train_ids, 1].to_list()
    }).cast_column("image", Image())


    train_val_dataset = train_dataset.class_encode_column("label").train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    test_dataset = Dataset.from_dict({
        "image": df.iloc[test_ids, 0].to_list(),
        "label": df.iloc[test_ids, 1].to_list()
    }).cast_column("image", Image())

    return train_val_dataset, test_dataset

def apply_transforms(examples, dataset_transform):
    examples["pixel_values"] = [dataset_transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def prepare_dataset(processor, train_dataset, val_dataset, test_dataset):
    try:
        image_mean, image_std = processor.image_mean, processor.image_std
        size = processor.size["height"]
        
        normalize = Normalize(mean=image_mean, std=image_std)
        
        dataset_transform = Compose([
            Resize(size),
            CenterCrop((size, size)),
            ToTensor(),
            normalize,
        ])
    except:
        size = processor.crop_size["height"]
        
        dataset_transform = Compose([
            Resize(size),
            CenterCrop((size, size)),
            ToTensor(),
        ])
    
    train_dataset.set_transform(lambda examples: apply_transforms(examples, dataset_transform))
    val_dataset.set_transform(lambda examples: apply_transforms(examples, dataset_transform))
    test_dataset.set_transform(lambda examples: apply_transforms(examples, dataset_transform))
    return train_dataset, val_dataset, test_dataset


def train_model(model: ViTForImageClassification | BeitForImageClassification | MobileViTForImageClassification, 
                processor: ViTImageProcessor | BeitImageProcessor | MobileViTImageProcessor, 
                train_dataset: Dataset, val_dataset: Dataset, run_name: str) -> Trainer:
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}
        
    train_args = TrainingArguments(
        output_dir="models-info",
        report_to="wandb",
        save_strategy="no",
        logging_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        run_name=run_name,
        weight_decay=0.01,
        logging_dir="logs",
        remove_unused_columns=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
