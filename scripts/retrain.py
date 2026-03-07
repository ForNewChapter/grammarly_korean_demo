import torch
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*60)
print("🔄 모델 재훈련 (전체 데이터)")
print("="*60)

# 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# 데이터 로드
print("\n📂 데이터 로딩...")
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

# 전체 데이터 사용 (메모리 여유 있으면)
# 또는 더 많은 샘플 사용
train_df = train_df.sample(n=min(50000, len(train_df)), random_state=42)
val_df = val_df.sample(n=min(5000, len(val_df)), random_state=42)

print(f"Train: {len(train_df):,}개")
print(f"Validation: {len(val_df):,}개")

# 모델 로드 (처음부터 또는 이전 모델에서)
print("\n🤖 모델 로딩...")
model_name = "gogamza/kobart-base-v2"  # 처음부터
# model_name = "models/kobart-spell-final"  # 이전 모델에서 계속

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# 데이터셋 준비
train_dataset = Dataset.from_pandas(train_df[['original', 'corrected']])
val_dataset = Dataset.from_pandas(val_df[['original', 'corrected']])

def preprocess_function(examples):
    inputs = examples["original"]
    targets = examples["corrected"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 토크나이징
print("\n🔧 데이터 전처리...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 훈련 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("models") / f"kobart-spell-v2-{timestamp}"

training_args = Seq2SeqTrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,  # 더 많은 epochs
    per_device_train_batch_size=8,  # 배치 크기 증가
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    
    logging_steps=100,
    report_to="none",
    
    predict_with_generate=True,
    generation_max_length=128,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    use_mps_device=device.type == "mps",
    dataloader_num_workers=0,
    fp16=False,
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 훈련
print("\n🚀 훈련 시작!")
print(f"총 스텝: {len(train_dataset) // 8 * 3}")
trainer.train()

# 저장
final_dir = Path("models") / "kobart-spell-v2-final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

print(f"\n✅ 완료! 모델 저장: {final_dir}")
