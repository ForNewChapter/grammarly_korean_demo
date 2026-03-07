import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class SpellChecker:
    def __init__(self, model_path="models/kobart-spell-final"):
        """훈련된 모델 로드"""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")
        
        print(f"📚 Loading model from: {model_path}")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def correct(self, text, max_length=128):
        """단일 문장 교정"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # token_type_ids 제거
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.9
            )
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
    
    def test_examples(self):
        """예시 문장 테스트"""
        test_cases = [
            "안녕하세여 반갑습니다",
            "오늘 날씨가 너무 좋아서 기분이 조아요",
            "맞춤법을 틀리게 쓰는건 정말 어려워여",
            "이렇게 공부하니까 실력이 늘어나는것 같아여",
            "앱을 만들고 싶ㅍ어",
            "너무 재밌어서 시간 가는줄 몰랏네요",
            "공부를 열심히 햇더니 성적이 올랏어요",
            "내일 회의가 잇는데 준비를 해야되요",
            "이 문제를 어떻해 해결할지 모르겟어요",
            "오늘은날씨가좋아서기분이좋아요"
        ]
        
        print("\n" + "="*60)
        print("📝 맞춤법 교정 테스트")
        print("="*60)
        
        for i, text in enumerate(test_cases, 1):
            corrected = self.correct(text)
            
            print(f"\n[{i}]")
            print(f"❌ 원문: {text}")
            print(f"✅ 교정: {corrected}")
            
            if text != corrected:
                print(f"   → 변경됨")
            else:
                print(f"   → 변경 없음")
    
    def evaluate_on_test_data(self, n_samples=100):
        """테스트 데이터로 평가"""
        print("\n" + "="*60)
        print("📊 테스트 데이터 평가")
        print("="*60)
        
        # 테스트 데이터 로드
        test_df = pd.read_csv("data/processed/validation.csv").head(n_samples)
        
        correct_count = 0
        results = []
        
        print(f"평가 중... (총 {len(test_df)}개)")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            original = str(row['original'])
            expected = str(row['corrected'])
            
            predicted = self.correct(original)
            
            is_correct = (predicted.strip() == expected.strip())
            if is_correct:
                correct_count += 1
            
            results.append({
                'original': original,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct
            })
        
        accuracy = correct_count / len(test_df) * 100
        print(f"\n정확도: {accuracy:.2f}% ({correct_count}/{len(test_df)})")
        
        # 틀린 예시 몇 개 출력
        print("\n📋 예시 (틀린 것 위주):")
        wrong_examples = [r for r in results if not r['correct']][:5]
        
        for i, ex in enumerate(wrong_examples, 1):
            print(f"\n[{i}]")
            print(f"원문: {ex['original'][:50]}...")
            print(f"정답: {ex['expected'][:50]}...")
            print(f"예측: {ex['predicted'][:50]}...")
        
        return results

def main():
    # 모델 로드
    checker = SpellChecker()
    
    # 예시 테스트
    checker.test_examples()
    
    # 평가
    results = checker.evaluate_on_test_data(n_samples=50)
    
    # 인터랙티브 모드
    print("\n" + "="*60)
    print("💬 인터랙티브 모드 (종료: 'quit' 입력)")
    print("="*60)
    
    while True:
        text = input("\n입력> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if text:
            corrected = checker.correct(text)
            print(f"교정> {corrected}")

if __name__ == "__main__":
    main()
