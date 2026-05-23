# ================================================================
# 📂 Project   : GPT Custom Modelling
# 📜 File      : retrain_slm.py
# 🧑‍💻 Author    : Abhinaba
# 🕒 Created   : 2026-05-23
# 🔄 Revision  : v1.0.0
# ✨ Purpose   : Retraining Small Language model with custom dataset
# ================================================================
# 📝 Change Log:
#   v1.0.0 | 2026-05-23 | Initial version
# ================================================================

import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TextGenerationPipeline,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore") # Suppress all warnings
warnings.warn("This will not be shown", UserWarning)
print("[Warning Supressed] > Script continues without warning output")

# ============================================================================
# PART 1: LOAD AND USE THE SAVED MODEL FOR INFERENCE
# ============================================================================

def load_model_and_tokenizer(model_dir="./my_slm"):
    """
    Load the trained model and tokenizer from saved directory.
    
    Args:
        model_dir (str): Path to the directory where model was saved
    
    Returns:
        model: The loaded GPT2 model
        tokenizer: The loaded tokenizer
    """
    print("=" * 60)
    print("LOADING SAVED MODEL AND TOKENIZER")
    print("=" * 60)
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    print(f"✓ Tokenizer loaded from {model_dir}")
    print(f"  - Vocab size: {len(tokenizer)}")
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print(f"✓ Model loaded from {model_dir}")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✓ Model moved to device: {device}")
    
    return model, tokenizer


def generate_text_simple(model, tokenizer, prompt, max_length=100, num_sequences=1):
    """
    Simple text generation without pipeline.
    
    Args:
        model: The loaded GPT2 model
        tokenizer: The tokenizer
        prompt (str): Input prompt for generation
        max_length (int): Maximum length of generated text
        num_sequences (int): Number of sequences to generate
    
    Returns:
        list: Generated text sequences
    """
    print("\n" + "=" * 60)
    print("TEXT GENERATION (SIMPLE METHOD)")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    
    device = next(model.parameters()).device
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=0.7,  # Lower = more deterministic
            top_p=0.9,        # Nucleus sampling
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print results
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    for i, text in enumerate(generated_texts, 1):
        print(f"Generated Text {i}:")
        print(f"{text}\n")
    
    return generated_texts


def generate_text_pipeline(model, tokenizer, prompt, max_length=100):
    """
    Text generation using transformers pipeline (easier to use).
    
    Args:
        model: The loaded GPT2 model
        tokenizer: The tokenizer
        prompt (str): Input prompt for generation
        max_length (int): Maximum length of generated text
    
    Returns:
        str: Generated text
    """
    print("\n" + "=" * 60)
    print("TEXT GENERATION (PIPELINE METHOD)")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    
    # Create pipeline
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    
    # Generate
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )
    
    generated_text = result[0]['generated_text']
    print(f"Generated Text:")
    print(f"{generated_text}\n")
    
    return generated_text


def calculate_perplexity(model, tokenizer, text):
    """
    Calculate perplexity of the model on given text.
    Lower perplexity = better performance
    
    Args:
        model: The loaded GPT2 model
        tokenizer: The tokenizer
        text (str): Text to evaluate
    
    Returns:
        float: Perplexity score
    """
    print("\n" + "=" * 60)
    print("CALCULATE MODEL PERPLEXITY")
    print("=" * 60)
    print(f"Text: {text}\n")
    
    device = next(model.parameters()).device
    
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}\n")
    
    return perplexity.item()


# ============================================================================
# PART 2: RETRAIN THE MODEL ON NEW DATA
# ============================================================================

def retrain_model(
    model_dir="./my_slm",
    new_data_path="new_data.txt",
    output_dir="./my_slm_retrained",
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5
):
    """
    Retrain the saved model on new data (fine-tuning).
    
    Args:
        model_dir (str): Path to the original saved model
        new_data_path (str): Path to new training data
        output_dir (str): Where to save the retrained model
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate (lower for fine-tuning)
    """
    print("\n" + "=" * 60)
    print("RETRAINING MODEL ON NEW DATA")
    print("=" * 60)
    
    # Check if new data exists
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"Data file not found: {new_data_path}")
    
    # Load saved model and tokenizer
    print(f"\n1. Loading pretrained model from {model_dir}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print(f"✓ Model and tokenizer loaded")
    
    # Load new dataset
    print(f"\n2. Loading new training data from {new_data_path}...")
    raw_datasets = load_dataset("text", data_files={"train": new_data_path})
    print(f"✓ Loaded {len(raw_datasets['train'])} examples")
    
    # Tokenize new data
    print(f"\n3. Tokenizing new data...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing new data"
    )
    print(f"✓ Data tokenized")
    
    # Filter empty sequences
    print(f"\n4. Filtering empty sequences...")
    def filter_empty_sequences(examples):
        valid = []
        for input_ids in examples["input_ids"]:
            non_pad_count = sum(1 for token_id in input_ids if token_id != tokenizer.pad_token_id)
            valid.append(non_pad_count > 0)
        return valid
    
    dataset_before = len(tokenized_datasets['train'])
    tokenized_datasets = tokenized_datasets.filter(
        filter_empty_sequences,
        batched=True,
        batch_size=1000
    )
    dataset_after = len(tokenized_datasets['train'])
    print(f"✓ Dataset size: {dataset_before} → {dataset_after}")
    
    if dataset_after == 0:
        raise ValueError("No valid sequences in new data!")
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Training arguments (note: lower learning rate for fine-tuning)
    print(f"\n5. Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=learning_rate,  # Lower learning rate for fine-tuning
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=1,
        remove_unused_columns=False
    )
    print(f"✓ Training configured")
    print(f"  - Learning rate: {learning_rate} (lower for fine-tuning)")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    
    # Start training
    print(f"\n6. Starting retraining...")
    trainer.train()
    print(f"✓ Retraining completed!")
    
    # Save retrained model
    print(f"\n7. Saving retrained model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Retrained model saved")
    
    return model, tokenizer


# ============================================================================
# PART 3: CONTINUOUS LEARNING / INCREMENTAL UPDATES
# ============================================================================

def update_model_incrementally(
    model_dir="./my_slm",
    new_data_path="incremental_data.txt",
    num_epochs=1,
    learning_rate=1e-5
):
    """
    Perform incremental updates (continuous learning) with very low learning rate.
    Use this for smaller updates without forgetting previous knowledge.
    
    Args:
        model_dir (str): Path to the model to update
        new_data_path (str): Path to new incremental data
        num_epochs (int): Should be 1 or 2 for incremental updates
        learning_rate (float): Very low learning rate to prevent catastrophic forgetting
    """
    print("\n" + "=" * 60)
    print("INCREMENTAL MODEL UPDATE (CONTINUOUS LEARNING)")
    print("=" * 60)
    
    # Load model
    print("\nLoading model for incremental update...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print("✓ Model loaded")
    
    # Load small amount of new data
    print(f"\nLoading incremental data from {new_data_path}...")
    raw_datasets = load_dataset("text", data_files={"train": new_data_path})
    print(f"✓ Loaded {len(raw_datasets['train'])} examples")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Filter empty sequences
    def filter_empty_sequences(examples):
        valid = []
        for input_ids in examples["input_ids"]:
            non_pad_count = sum(1 for token_id in input_ids if token_id != tokenizer.pad_token_id)
            valid.append(non_pad_count > 0)
        return valid
    
    tokenized_datasets = tokenized_datasets.filter(
        filter_empty_sequences,
        batched=True,
        batch_size=1000
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training with very low learning rate
    print("\nSetting up incremental learning...")
    training_args = TrainingArguments(
        output_dir=model_dir,  # Overwrite the same model
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,  # Smaller batch for stability
        save_steps=100,
        logging_steps=25,
        learning_rate=learning_rate,  # Very low learning rate
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    
    print("Starting incremental training...")
    trainer.train()
    
    # Save updated model
    print(f"\nSaving updated model to {model_dir}...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("✓ Model updated successfully")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("#" * 60)
    print("# COMPREHENSIVE GUIDE: LOAD, USE, AND RETRAIN SLM MODEL")
    print("#" * 60)
    
    # -----------------------------------------------------------------------
    # EXAMPLE 1: LOAD AND GENERATE TEXT
    # -----------------------------------------------------------------------
    print("\n\n[EXAMPLE 1] Loading saved model and generating text...")
    print("-" * 60)
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_dir="./my_slm")
        
        # Generate text (simple method)
        prompt = "Machine learning"
        generate_text_simple(model, tokenizer, prompt, max_length=80, num_sequences=2)
        
        # Generate text (pipeline method)
        # generate_text_pipeline(model, tokenizer, prompt, max_length=80)
        
        # Calculate perplexity
        # test_text = "This is a test sentence for evaluating the model."
        # calculate_perplexity(model, tokenizer, test_text)
        
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Make sure to train the model first using train_slm_fixed.py")
    
    
    # -----------------------------------------------------------------------
    # EXAMPLE 2: RETRAIN ON NEW DATA
    # -----------------------------------------------------------------------
    print("\n\n[EXAMPLE 2] Retraining on new data...")
    print("-" * 60)
    
    # First create some new training data
    new_data_file = "new_training_data.txt"
    if not os.path.exists(new_data_file):
        print(f"Creating new training data: {new_data_file}")
        with open(new_data_file, "w", encoding="utf-8") as f:
            for i in range(100):
                f.write(f"New sample sentence {i} about deep learning and neural networks.\n")
                f.write(f"Transformers are powerful models for natural language processing.\n")
        print(f"✓ Created {new_data_file}")
    
    # Uncomment to retrain (this will take time)
    # try:
    #     retrain_model(
    #         model_dir="./my_slm",
    #         new_data_path="new_training_data.txt",
    #         output_dir="./my_slm_retrained",
    #         num_epochs=2,
    #         batch_size=8,
    #         learning_rate=5e-5  # Lower than original training
    #     )
    # except FileNotFoundError as e:
    #     print(f"⚠️  {e}")
    
    
    # -----------------------------------------------------------------------
    # EXAMPLE 3: INCREMENTAL UPDATE
    # -----------------------------------------------------------------------
    print("\n\n[EXAMPLE 3] Incremental model update...")
    print("-" * 60)
    
    # Create incremental data
    incremental_data_file = "incremental_data.txt"
    if not os.path.exists(incremental_data_file):
        print(f"Creating incremental data: {incremental_data_file}")
        with open(incremental_data_file, "w", encoding="utf-8") as f:
            for i in range(30):
                f.write(f"Incremental update {i}: New knowledge about AI and ML.\n")
        print(f"✓ Created {incremental_data_file}")
    
    # Uncomment to perform incremental update
    # try:
    #     update_model_incrementally(
    #         model_dir="./my_slm",
    #         new_data_path="incremental_data.txt",
    #         num_epochs=1,
    #         learning_rate=1e-5  # Very low for continuous learning
    #     )
    # except FileNotFoundError as e:
    #     print(f"⚠️  {e}")
    
    
    print("\n\n" + "#" * 60)
    print("# COMPLETE! See comments above to enable Examples 2 & 3")
    print("#" * 60)