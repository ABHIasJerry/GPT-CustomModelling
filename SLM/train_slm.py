# ================================================================
# 📂 Project   : GPT Custom Modelling
# 📜 File      : train_slm.py
# 🧑‍💻 Author    : Abhinaba
# 🕒 Created   : 2026-05-23
# 🔄 Revision  : v1.0.0
# ✨ Purpose   : Training Small Language model with custom dataset
# ================================================================
# 📝 Change Log:
#   v1.0.0 | 2026-05-23 | Initial version
# ================================================================

import os
import torch
import shutil
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    GPT2TokenizerFast, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
import warnings
warnings.filterwarnings("ignore") # Suppress all warnings
warnings.warn("This will not be shown", UserWarning)
print("[Warning Supressed] > Script continues without warning output")


def build_and_train_slm(data_path="private_data.txt", output_dir="./my_slm"):
    """
    Build and train a small language model (SLM) on custom text data.
    
    Fixes applied:
    1. Added padding="max_length" to ensure consistent tensor shapes
    2. Filter empty sequences to prevent 0-element tensor errors
    3. Validate dataset before training
    """
    
    # --- STEP 1: CLEAN DIRECTORY & TRAIN TOKENIZER ---
    print("=" * 50)
    print("STEP 1: Preparing directories and tokenizer...")
    print("=" * 50)
    
    # Manually handle directory overwriting to avoid TrainingArguments errors
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_path)
    
    # Train ByteLevelBPE tokenizer
    bw_tokenizer = ByteLevelBPETokenizer()
    bw_tokenizer.train(
        files=[data_path], 
        vocab_size=30000, 
        min_frequency=2, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    bw_tokenizer.save_model(tokenizer_path)
    print(f"✓ Tokenizer trained and saved to {tokenizer_path}")

    # Load tokenizer for transformers
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # --- STEP 2: PREPARE DATASET ---
    print("\n" + "=" * 50)
    print("STEP 2: Loading and tokenizing dataset...")
    print("=" * 50)
    
    raw_datasets = load_dataset("text", data_files={"train": data_path})
    print(f"✓ Raw dataset loaded: {len(raw_datasets['train'])} examples")

    def tokenize_function(examples):
        """Tokenize text with padding and truncation for consistent shapes."""
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=512,
            padding="max_length"  # FIX: Ensure all sequences have same length
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    print(f"✓ Dataset tokenized")

    # --- STEP 3: FILTER EMPTY SEQUENCES ---
    print("\n" + "=" * 50)
    print("STEP 3: Filtering empty sequences...")
    print("=" * 50)
    
    def filter_empty_sequences(examples):
        """Remove sequences that are entirely padding tokens."""
        valid = []
        for input_ids in examples["input_ids"]:
            # Count non-pad tokens
            non_pad_count = sum(1 for token_id in input_ids if token_id != tokenizer.pad_token_id)
            valid.append(non_pad_count > 0)
        return valid

    dataset_size_before = len(tokenized_datasets['train'])
    tokenized_datasets = tokenized_datasets.filter(
        filter_empty_sequences,
        batched=True,
        batch_size=1000,
        desc="Filtering empty sequences"
    )
    dataset_size_after = len(tokenized_datasets['train'])
    
    print(f"✓ Dataset size before filtering: {dataset_size_before}")
    print(f"✓ Dataset size after filtering: {dataset_size_after}")
    
    # Verify we have data
    if dataset_size_after == 0:
        raise ValueError(
            "❌ No valid sequences found in dataset after filtering. "
            "Check your input data for content."
        )

    # --- STEP 4: CONFIGURE MODEL ---
    print("\n" + "=" * 50)
    print("STEP 4: Initializing GPT-2 model...")
    print("=" * 50)
    
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  - Vocab size: {len(tokenizer)}")
    print(f"  - Max sequence length: 512")
    print(f"  - Embedding dimension: 512")
    print(f"  - Layers: 6")
    print(f"  - Attention heads: 8")
    print(f"  - Total parameters: {total_params:,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    print(f"✓ Data collator configured for causal language modeling")

    # --- STEP 5: SET TRAINING ARGUMENTS ---
    print("\n" + "=" * 50)
    print("STEP 5: Configuring training arguments...")
    print("=" * 50)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=1,
        remove_unused_columns=False
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Training arguments configured")
    print(f"  - Device: {device}")
    print(f"  - Batch size: 8")
    print(f"  - Learning rate: 1e-4")
    print(f"  - Epochs: 5")
    print(f"  - FP16: {torch.cuda.is_available()}")

    # --- STEP 6: START TRAINING ---
    print("\n" + "=" * 50)
    print("STEP 6: Starting training...")
    print("=" * 50)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()
    print(f"\n✓ Training completed successfully!")

    # --- STEP 7: SAVE MODEL ---
    print("\n" + "=" * 50)
    print("STEP 7: Saving model and tokenizer...")
    print("=" * 50)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model saved to {output_dir}")
    print(f"✓ Tokenizer saved to {output_dir}")
    print("\n" + "=" * 50)
    print("✅ Training pipeline completed successfully!")
    print("=" * 50)


def create_sample_data(data_path="private_data.txt", num_samples=1000):
    """Create sample training data if it doesn't exist."""
    if os.path.exists(data_path):
        print(f"Data file {data_path} already exists. Skipping creation.")
        return
    
    print(f"Creating sample training data: {data_path}")
    with open(data_path, "w", encoding="utf-8") as f:
        # Create diverse training data
        templates = [
            "This is a sample sentence number {} for training the SLM.",
            "Machine learning models require large amounts of training data to perform well.",
            "Natural language processing is a challenging and important field in AI.",
            "The quick brown fox jumps over the lazy dog.",
            "Language models can generate text that is coherent and contextually relevant.",
            "Training a small language model requires careful consideration of various hyperparameters.",
            "Data preprocessing is an essential step in machine learning pipelines.",
            "The transformer architecture has revolutionized the field of deep learning.",
        ]
        
        for i in range(num_samples):
            template = templates[i % len(templates)]
            f.write(template.format(i) + "\n")
    
    print(f"✓ Created {num_samples} training examples in {data_path}")


if __name__ == "__main__":
    # Create sample data if needed
    create_sample_data()
    
    # Train the model
    build_and_train_slm()