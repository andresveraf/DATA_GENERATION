"""
Transformer NER Model Training for Latin American PII Data
===========================================================

This module trains BERT-based multilingual NER models using Hugging Face Transformers.
Optimized for Spanish and Portuguese PII entity recognition.

Features:
- BERT multilingual base model fine-tuning
- BIO tagging scheme for entity labeling
- Token-level classification with proper alignment
- Comprehensive evaluation metrics
- Model checkpointing and best model saving
- Support for data augmentation and class balancing

Author: Andrés Vera Figueroa
Date: August 2025
Purpose: Production-ready multilingual NER training
"""

import json
import os
import torch
import numpy as np
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from datasets import Dataset
import torch.nn as nn

@dataclass
class TrainingConfig:
    model_name: str = "bert-base-multilingual-cased"
    output_dir: str = "models"
    train_file: str = "output/train_transformer_50000.json"
    dev_file: str = "output/dev_transformer_10000.json"
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    fp16: bool = True  # Enable for faster training if supported

class TransformerNERTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Define entity labels (BIO scheme)
        self.labels = [
            "O",  # Outside
            "B-CUSTOMER_NAME", "I-CUSTOMER_NAME",
            "B-ID_NUMBER", "I-ID_NUMBER", 
            "B-ADDRESS", "I-ADDRESS",
            "B-PHONE_NUMBER", "I-PHONE_NUMBER",
            "B-EMAIL", "I-EMAIL",
            "B-AMOUNT", "I-AMOUNT",
            "B-SEQ_NUMBER", "I-SEQ_NUMBER"
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        print(f"🏷️  Entity Labels ({len(self.labels)}): {self.labels}")
        
        # Initialize tokenizer and model
        print(f"🤖 Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        # Data collator
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=config.max_length
        )
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Load dataset from JSON file"""
        print(f"📂 Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data):,} examples")
        return data
    
    def align_labels_with_tokens(self, labels: List[str], word_ids: List[Optional[int]]) -> List[int]:
        """Align entity labels with tokenized words"""
        aligned_labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First token of a word gets the label
                if word_id < len(labels):
                    aligned_labels.append(self.label2id[labels[word_id]])
                else:
                    aligned_labels.append(self.label2id["O"])
            else:
                # Subsequent tokens of the same word
                if word_id < len(labels):
                    label = labels[word_id]
                    # Convert B- to I- for continuation tokens
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    aligned_labels.append(self.label2id[label])
                else:
                    aligned_labels.append(self.label2id["O"])
            
            previous_word_id = word_id
        
        return aligned_labels
    
    def create_bio_labels(self, text: str, entities: List[Dict]) -> List[str]:
        """Create BIO labels for the text"""
        # Split text into words (simple whitespace tokenization)
        words = text.split()
        labels = ["O"] * len(words)
        
        # Convert character-based spans to word-based labels
        char_to_word = {}
        char_pos = 0
        
        for word_idx, word in enumerate(words):
            for char_idx in range(len(word)):
                char_to_word[char_pos + char_idx] = word_idx
            char_pos += len(word) + 1  # +1 for space
        
        # Sort entities by start position to handle overlaps
        sorted_entities = sorted(entities, key=lambda x: x["start"])
        
        for entity in sorted_entities:
            start_char = entity["start"]
            end_char = entity["end"]
            label = entity["label"]
            
            # Find word indices that overlap with this entity
            start_word = None
            end_word = None
            
            for char_idx in range(start_char, min(end_char, max(char_to_word.keys()) + 1)):
                if char_idx in char_to_word:
                    word_idx = char_to_word[char_idx]
                    if start_word is None:
                        start_word = word_idx
                    end_word = word_idx
            
            # Apply BIO labeling
            if start_word is not None and end_word is not None:
                # Check for conflicts (already labeled)
                conflict = any(labels[i] != "O" for i in range(start_word, end_word + 1))
                if not conflict:
                    labels[start_word] = f"B-{label}"
                    for i in range(start_word + 1, end_word + 1):
                        labels[i] = f"I-{label}"
        
        return labels
    
    def tokenize_and_align_labels(self, examples: List[Dict]) -> Dict:
        """Tokenize text and align labels"""
        texts = []
        all_labels = []
        
        print("🔄 Processing examples...")
        failed_examples = 0
        
        for idx, example in enumerate(examples):
            try:
                text = example["text"]
                entities = example["entities"]
                
                # Create BIO labels
                bio_labels = self.create_bio_labels(text, entities)
                
                texts.append(text)
                all_labels.append(bio_labels)
                
            except Exception as e:
                failed_examples += 1
                if failed_examples <= 5:  # Show first 5 errors
                    print(f"⚠️  Error processing example {idx}: {e}")
                continue
        
        if failed_examples > 0:
            print(f"⚠️  Failed to process {failed_examples:,} examples")
        
        print(f"🔤 Tokenizing {len(texts):,} examples...")
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.config.max_length,
            is_split_into_words=False,
            return_offsets_mapping=True
        )
        
        # Align labels with tokens
        aligned_labels = []
        alignment_errors = 0
        
        for i in range(len(texts)):
            try:
                # Get word IDs for alignment
                words = texts[i].split()
                labels = all_labels[i]
                
                # Create mapping from character positions to word indices
                char_to_word = {}
                char_pos = 0
                for word_idx, word in enumerate(words):
                    for char_idx in range(len(word)):
                        char_to_word[char_pos + char_idx] = word_idx
                    char_pos += len(word) + 1  # +1 for space
                
                # Map tokens to words using offset mapping
                word_ids = []
                for start, end in tokenized["offset_mapping"][i]:
                    if start == end == 0:  # Special tokens
                        word_ids.append(None)
                    else:
                        # Find which word this token belongs to
                        word_id = None
                        for char_idx in range(start, end):
                            if char_idx in char_to_word:
                                word_id = char_to_word[char_idx]
                                break
                        word_ids.append(word_id)
                
                # Align labels
                aligned = self.align_labels_with_tokens(labels, word_ids)
                aligned_labels.append(aligned)
                
            except Exception as e:
                alignment_errors += 1
                if alignment_errors <= 3:
                    print(f"⚠️  Label alignment error {i}: {e}")
                # Fallback: all O labels
                aligned_labels.append([self.label2id["O"]] * len(tokenized["input_ids"][i]))
        
        if alignment_errors > 0:
            print(f"⚠️  {alignment_errors:,} label alignment errors (using O labels as fallback)")
        
        # Remove offset mapping (not needed for training)
        tokenized.pop("offset_mapping")
        tokenized["labels"] = aligned_labels
        
        return tokenized
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare training and validation datasets"""
        # Load data
        train_data = self.load_dataset(self.config.train_file)
        dev_data = self.load_dataset(self.config.dev_file)
        
        # Tokenize and align labels
        train_tokenized = self.tokenize_and_align_labels(train_data)
        dev_tokenized = self.tokenize_and_align_labels(dev_data)
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict(train_tokenized)
        dev_dataset = Dataset.from_dict(dev_tokenized)
        
        print(f"📊 Training set: {len(train_dataset):,} examples")
        print(f"📊 Dev set: {len(dev_dataset):,} examples")
        
        return train_dataset, dev_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for sklearn metrics
        flat_true_labels = [label for sublist in true_labels for label in sublist]
        flat_predictions = [pred for sublist in true_predictions for pred in sublist]
        
        # Calculate metrics
        f1 = f1_score(flat_true_labels, flat_predictions, average='weighted', zero_division=0)
        
        # Entity-level metrics
        entity_f1_scores = {}
        for label in set(flat_true_labels):
            if label != "O":
                entity_labels = [1 if l == label else 0 for l in flat_true_labels]
                entity_preds = [1 if p == label else 0 for p in flat_predictions]
                entity_f1 = f1_score(entity_labels, entity_preds, zero_division=0)
                entity_f1_scores[f"f1_{label}"] = entity_f1
        
        return {
            "f1": f1,
            **entity_f1_scores
        }
    
    def train(self) -> str:
        """Train the model"""
        print("🚀 Starting transformer NER training...")
        
        # Prepare datasets
        train_dataset, dev_dataset = self.prepare_datasets()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = os.path.join(self.config.output_dir, f"transformer_ner_{timestamp}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            warmup_ratio=self.config.warmup_ratio,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            dataloader_drop_last=False,
            run_name=f"transformer_ner_{timestamp}",
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Train
        print(f"📊 Training configuration:")
        print(f"   - Model: {self.config.model_name}")
        print(f"   - Train examples: {len(train_dataset):,}")
        print(f"   - Dev examples: {len(dev_dataset):,}")
        print(f"   - Batch size: {self.config.batch_size}")
        print(f"   - Learning rate: {self.config.learning_rate}")
        print(f"   - Epochs: {self.config.num_epochs}")
        print(f"   - Output: {model_output_dir}")
        
        trainer.train()
        
        # Save the final model
        final_model_dir = os.path.join(model_output_dir, "final_model")
        trainer.save_model(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        # Evaluate on dev set
        print("\n📊 Final evaluation on dev set:")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        eval_file = os.path.join(model_output_dir, "evaluation_results.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save training configuration
        config_dict = {
            "model_name": self.config.model_name,
            "training_args": training_args.to_dict(),
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "training_completed": datetime.now().isoformat(),
            "final_eval_results": eval_results
        }
        
        config_file = os.path.join(model_output_dir, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved: {final_model_dir}")
        print(f"📊 Dev F1 Score: {eval_results.get('eval_f1', 0):.4f}")
        print(f"📄 Config saved: {config_file}")
        
        return final_model_dir

def main():
    parser = argparse.ArgumentParser(description="Train Transformer NER model")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased", 
                       help="Pre-trained model name")
    parser.add_argument("--train-file", default="output/train_transformer_50000.json",
                       help="Training data file")
    parser.add_argument("--dev-file", default="output/dev_transformer_10000.json", 
                       help="Development data file")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    
    args = parser.parse_args()
    
    print("🤖 Transformer NER Model Training")
    print("==================================")
    
    # Check if files exist
    if not os.path.exists(args.train_file):
        print(f"❌ Training file not found: {args.train_file}")
        print("💡 Run transformer_data_generator.py first to create the dataset")
        return
    
    if not os.path.exists(args.dev_file):
        print(f"❌ Dev file not found: {args.dev_file}")
        print("💡 Run transformer_data_generator.py first to create the dataset") 
        return
    
    # Create training configuration
    config = TrainingConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        dev_file=args.dev_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_length=args.max_length
    )
    
    # Initialize trainer and start training
    trainer = TransformerNERTrainer(config)
    model_path = trainer.train()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"🎯 Use the trained model for inference:")
    print(f"   Model path: {model_path}")
    print(f"   Languages: Spanish + Portuguese")
    print(f"   Entities: 7 types (CUSTOMER_NAME, ID_NUMBER, ADDRESS, PHONE_NUMBER, EMAIL, AMOUNT, SEQ_NUMBER)")

if __name__ == "__main__":
    main()
