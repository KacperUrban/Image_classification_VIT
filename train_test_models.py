from custom_utils.custom_utils import (
    create_dataset,
    prepare_dataset,
    train_model,
)
from dotenv import load_dotenv
import os
import wandb
import json
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    BeitForImageClassification,
    BeitImageProcessor,
    MobileViTForImageClassification,
    MobileViTImageProcessor,
)
from datasets import Dataset
import numpy as np
import pandas as pd


def train_test_google(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, tag: str
) -> None:
    model = ViTForImageClassification.from_pretrained("models/google/model")
    processor = ViTImageProcessor.from_pretrained("models/google/processor")

    train_dataset_google, val_dataset_google, test_dataset_google = prepare_dataset(
        processor, train_dataset, val_dataset, test_dataset
    )

    trained_model = train_model(
        model, processor, train_dataset_google, val_dataset_google, f"google-{tag}"
    )

    outputs = trained_model.predict(test_dataset_google)
    print(
        f"Accuracy: {np.round(outputs.metrics['test_accuracy'], 3)}, f1-score: {np.round(outputs.metrics['test_f1'], 3)}"
    )

    wandb.finish()


def train_test_microsoft(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, tag: str
) -> None:
    model = BeitForImageClassification.from_pretrained("models/microsoft/model")
    processor = BeitImageProcessor.from_pretrained("models/microsoft/processor")

    train_dataset_microsoft, val_dataset_microsoft, test_dataset_microsoft = (
        prepare_dataset(processor, train_dataset, val_dataset, test_dataset)
    )

    trained_model = train_model(
        model,
        processor,
        train_dataset_microsoft,
        val_dataset_microsoft,
        f"microsoft-{tag}",
    )

    outputs = trained_model.predict(test_dataset_microsoft)
    print(
        f"Accuracy: {np.round(outputs.metrics['test_accuracy'], 3)}, f1-score: {np.round(outputs.metrics['test_f1'], 3)}"
    )

    wandb.finish()


def train_test_apple(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, tag: str
) -> None:
    model = MobileViTForImageClassification.from_pretrained("models/apple/model")
    processor = MobileViTImageProcessor.from_pretrained("models/apple/processor")

    train_dataset_apple, val_dataset_apple, test_dataset_apple = (
        prepare_dataset(processor, train_dataset, val_dataset, test_dataset)
    )

    trained_model = train_model(
        model,
        processor,
        train_dataset_apple,
        val_dataset_apple,
        f"apple-{tag}",
    )

    outputs = trained_model.predict(test_dataset_apple)
    print(
        f"Accuracy: {np.round(outputs.metrics['test_accuracy'], 3)}, f1-score: {np.round(outputs.metrics['test_f1'], 3)}"
    )

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.environ["wandb_api_key"])
    os.environ["WANDB_PROJECT"] = "Classification-BR-VIT"
    os.environ["WANDB_WATCH"] = "false"

    with open("data/dictionary_ids.json") as file:
        data = json.load(file)

    df_information = pd.read_csv("data/data_informations.csv")

    tags = [
        "day",
        "night",
        "winter",
        "spring",
        "autumn",
        "day-winter",
        "night-winter",
        "day-spring",
        "night-spring",
        "day-autumn",
        "night-autumn",
    ]

    for tag in tags:
        ids = data[tag]

        train_val_dataset, test_dataset = create_dataset(ids, df_information)
        train_dataset, val_dataset = (
            train_val_dataset["train"],
            train_val_dataset["test"],
        )

        train_test_google(train_dataset, val_dataset, test_dataset, tag)
        train_test_microsoft(train_dataset, val_dataset, test_dataset, tag)
        train_test_apple(train_dataset, val_dataset, test_dataset, tag)
