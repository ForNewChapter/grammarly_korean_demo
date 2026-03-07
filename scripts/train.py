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
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpellCorrectionTrainer:
    def __init__(self):
        # 디바이스 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ Using Apple M3 Pro GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")
        
        # 경로 설정
        self.data_path = Path("data/processed")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 모델 설정
        self.model_name = "gogamza/kobart-base-v2"
        self.max_length = 128
        
        # 훈련 설정 (M3 Pro 최적화)
        self.batch_size = 4  # M3 Pro MPS를 위해 줄임
        self.learning_rate = 3e-5
        self.num_epochs = 1  # 테스트를 위해 1 epoch
        self.warmup_steps = 500
        
        print(f"📊 설정: batch_size={self.batch_size}, epochs={self.num_epochs}")
    
    def load_data(self):
        """데이터 로드"""
        print("\n📂 데이터 로딩 중...")
        
        train_df = pd.read_csv(self.data_path / "train.csv")
        val_df = pd.read_csv(self.data_path / "validation.csv")
        
        # 테스트를 위해 일부만 사용
        train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(1000, len(val_df)), random_state=42)
        
        print(f"  Train: {len(train_df):,}개")
        print(f"  Validation: {len(val_df):,}개")
        
        return train_df, val_df
    
    def prepare_dataset(self, train_df, val_df):
        """데이터셋 준비"""
        print("\n🔄 데이터셋 변환 중...")
        
        # NaN 값 처리
        train_df = train_df.dropna(subset=['original', 'corrected'])
        val_df = val_df.dropna(subset=['original', 'corrected'])
        
        # 문자열로 변환
        train_df['original'] = train_df['original'].astype(str)
        train_df['corrected'] = train_df['corrected'].astype(str)
        val_df['original'] = val_df['original'].astype(str)
        val_df['corrected'] = val_df['corrected'].astype(str)
        
        # HuggingFace Dataset으로 변환
        train_dataset = Dataset.from_pandas(train_df[['original', 'corrected']])
        val_dataset = Dataset.from_pandas(val_df[['original', 'corrected']])
        
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        return dataset
    
    def preprocess_function(self, examples):
        """토크나이징 함수"""
        inputs = examples["original"]
        targets = examples["corrected"]
        
        # 입력 토크나이징
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # 타겟 토크나이징
        labels = self.tokenizer(
            targets,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # token_type_ids 제거 (BART는 사용하지 않음)
        if 'token_type_ids' in model_inputs:
            del model_inputs['token_type_ids']
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train(self):
        """모델 훈련"""
        # 데이터 로드
        train_df, val_df = self.load_data()
        dataset = self.prepare_dataset(train_df, val_df)
        
        # 모델과 토크나이저 로드
        print("\n🤖 모델 로딩 중...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        print(f"  모델 파라미터: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
        # 데이터셋 전처리
        print("\n🔧 데이터 전처리 중...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=1  # MPS는 단일 프로세스 사용
        )
        
        # 훈련 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.model_dir / f"kobart-spell-{timestamp}"
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            
            # 평가 및 저장
            evaluation_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            
            # 로깅
            logging_dir=str(Path("logs") / timestamp),
            logging_steps=50,
            report_to="none",
            
            # 성능 설정
            predict_with_generate=True,
            generation_max_length=self.max_length,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # MPS 설정
            use_mps_device=self.device.type == "mps",
            dataloader_num_workers=0,
            
            # 기타
            push_to_hub=False,
            fp16=False,  # MPS는 fp16 미지원
            gradient_checkpointing=False,  # MPS에서 안정성을 위해 비활성화
        )
        
        # Data Collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8
        )
        
        # Trainer 생성
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
        )
        
        # 훈련 시작
        print("\n" + "="*60)
        print("🚀 훈련 시작!")
        print("="*60)
        print(f"  • Epochs: {self.num_epochs}")
        print(f"  • Train samples: {len(tokenized_dataset['train'])}")
        print(f"  • Train batches: {len(tokenized_dataset['train']) // self.batch_size}")
        print(f"  • Device: {self.device}")
        print(f"  • Output: {output_dir}")
        print("="*60)
        
        trainer.train()
        
        # 최종 모델 저장
        final_dir = self.model_dir / "kobart-spell-final"
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        print(f"\n✅ 훈련 완료!")
        print(f"  최종 모델: {final_dir}")
        
        return trainer, final_dir

if __name__ == "__main__":
    trainer = SpellCorrectionTrainer()
    trainer.train()
