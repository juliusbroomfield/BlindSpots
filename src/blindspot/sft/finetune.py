#!/usr/bin/env python3
"""
loRA fine-tuning of Llama 3.1 8B on GPT-5 distillation data.

setup:
  - LoRA: rank 16, alpha 32, q/k/v/o_proj
  - Loss on response tokens only (prompt masked)
  - 3-5 epochs, lr 2e-4, cosine schedule, 10% warmup
  - Effective batch size 16 via gradient accumulation
  - Save checkpoint every epoch; best = lowest val loss
  - Logs to wandb

usage:
  python finetune_lora.py --train finetune_data/train.jsonl \
                          --val   finetune_data/val.jsonl   \
                          --output checkpoints/

  # Eval on held-out after training:
  python finetune_lora.py --eval-only \
                          --checkpoint checkpoints/best \
                          --held-out finetune_data/held_out.jsonl
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)

# config
# These are the settings reported in appendix D.4: LoRA rank 16, 3 epochs,
# cosine schedule, peak LR 2e-4, effective batch size 16.

SEED             = 42
MODEL_ID         = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LEN      = 4096

LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

NUM_EPOCHS       = 3
LR               = 2e-4
WARMUP_RATIO     = 0.10
WEIGHT_DECAY     = 0.01
GRAD_CLIP        = 1.0
PER_DEVICE_BS    = 2      # adjust to fit GPU memory
GRAD_ACCUM_STEPS = 8      # effective batch = PER_DEVICE_BS * GRAD_ACCUM_STEPS = 16


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# tokenization

def tokenize_example(example: dict, tokenizer) -> dict:
    """
    apply Llama chat template and mask prompt tokens.
    returns input_ids, attention_mask, labels (prompt positions = -100).
    """
    messages = example["messages"]

    # full sequence
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # prompt only (user turn) — to find where the response starts
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_enc   = tokenizer(full_text,   truncation=True, max_length=MAX_SEQ_LEN)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN)

    input_ids      = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    labels         = list(input_ids)  # copy

    # mask prompt tokens
    prompt_len = len(prompt_enc["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def load_dataset(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        raw = [json.loads(line) for line in f]
    dataset = Dataset.from_list(raw)
    dataset = dataset.map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=["messages"],
        desc=f"Tokenizing {os.path.basename(path)}",
    )
    return dataset


# model setup

def build_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# training

def compute_loss(model, batch, device):
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in dataloader:
        loss = compute_loss(model, batch, device)
        if not torch.isnan(loss):
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")


def train(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="standpoint-llm-finetune",
        name=args.run_name or "llama-3.1-8b-lora",
        config={
            "model":       MODEL_ID,
            "lora_rank":   LORA_RANK,
            "lora_alpha":  LORA_ALPHA,
            "lr":          LR,
            "epochs":      NUM_EPOCHS,
            "batch_size":  PER_DEVICE_BS * GRAD_ACCUM_STEPS,
            "max_seq_len": MAX_SEQ_LEN,
            "seed":        SEED,
        },
    )

    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = build_model_and_tokenizer(MODEL_ID)

    # verify chat template on a few examples before training
    print("\nverifying chat template on the first example")
    with open(args.train) as f:
        sample = json.loads(f.readline())
    enc = tokenize_example(sample, tokenizer)
    n_prompt = sum(1 for label in enc["labels"] if label == -100)
    n_resp   = sum(1 for label in enc["labels"] if label != -100)
    print(f"  Prompt tokens (masked): {n_prompt}")
    print(f"  Response tokens (loss): {n_resp}")
    assert n_resp > 0, "All tokens are masked — chat template may be wrong!"
    print("  ✓ Chat template looks correct\n")

    print("Loading datasets...")
    train_ds = load_dataset(args.train, tokenizer)
    val_ds   = load_dataset(args.val,   tokenizer)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )
    train_loader = DataLoader(train_ds, batch_size=PER_DEVICE_BS,
                              shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=PER_DEVICE_BS,
                              shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps   = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0

    print(f"Starting training: {NUM_EPOCHS} epochs, {total_steps} optimizer steps\n")
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            loss = compute_loss(model, batch, device) / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches  += 1

            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({
                    "train/loss":      epoch_loss / n_batches,
                    "train/lr":        scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "step":            global_step,
                })

                if global_step % 50 == 0:
                    print(f"  epoch {epoch} step {global_step} | "
                          f"loss={epoch_loss/n_batches:.4f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e} | "
                          f"grad_norm={grad_norm:.3f}")

        avg_train_loss = epoch_loss / n_batches
        val_loss       = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} complete")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")

        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss":   val_loss,
            "epoch":            epoch,
        })

        # save checkpoint every epoch
        ckpt_dir = os.path.join(args.output, f"epoch_{epoch}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  Saved → {ckpt_dir}")

        # track best checkpoint by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir      = os.path.join(args.output, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ✓ New best val loss ({val_loss:.4f}) → saved to {best_dir}")

        wandb.log({"best/val_loss": best_val_loss})
        print()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    wandb.finish()


# held-out evaluation

def eval_held_out(args):
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    held_out_ds = load_dataset(args.held_out, tokenizer)
    collator    = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )
    loader = DataLoader(held_out_ds, batch_size=PER_DEVICE_BS,
                        shuffle=False, collate_fn=collator)

    loss = evaluate(model, loader, device)
    print(f"Held-out loss: {loss:.4f}  (perplexity: {math.exp(loss):.2f})")
    return loss


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune Llama 3.1 8B")
    parser.add_argument("--train",      type=str, default="finetune_data/train.jsonl")
    parser.add_argument("--val",        type=str, default="finetune_data/val.jsonl")
    parser.add_argument("--held-out",   type=str, default="finetune_data/held_out.jsonl")
    parser.add_argument("--output",     type=str, default="checkpoints")
    parser.add_argument("--run-name",   type=str, default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.eval_only:
        assert args.checkpoint, "--checkpoint required for --eval-only"
        eval_held_out(args)
    else:
        train(args)
