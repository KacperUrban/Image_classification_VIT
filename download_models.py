from transformers import ViTForImageClassification, ViTImageProcessor, BeitForImageClassification, BeitImageProcessor, MobileViTForImageClassification, MobileViTImageProcessor
import torch

if __name__ == "__main__":
    print("Downloading models and processors...")
    googlevit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=50,force_download=True, attn_implementation="sdpa", torch_dtype=torch.float16, ignore_mismatched_sizes=True)
    googlevit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", force_download=True)

    microsoftvit_model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224", num_labels=50, force_download=True, attn_implementation="sdpa", torch_dtype=torch.float16, ignore_mismatched_sizes=True)
    microsfotvit_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224", force_download=True)

    applevit_model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small", num_labels=50, force_download=True, torch_dtype=torch.float16, ignore_mismatched_sizes=True)
    applevit_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small", force_download=True)

    print("Saving models and processors...")

    googlevit_model.save_pretrained("models/google/model/")
    googlevit_processor.save_pretrained("models/google/processor/")


    microsoftvit_model.save_pretrained("models/microsoft/model/")
    microsfotvit_processor.save_pretrained("models/microsoft/processor/")


    applevit_model.save_pretrained("models/apple/model/")
    applevit_processor.save_pretrained("models/apple/processor/")

    print("Succussfully saved!")
