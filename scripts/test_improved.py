import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from pathlib import Path
import pandas as pd

class ImprovedSpellChecker:
    def __init__(self, model_path="models/kobart-spell-final"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")
        
        print(f"📚 Loading model from: {model_path}")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 특수 토큰 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def correct(self, text, max_length=128):
        """개선된 교정 함수"""
        # 입력 토크나이징
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 개선된 생성 파라미터
            outputs = self.model.generate(
                **inputs,
                max_length=min(len(text) + 20, max_length),  # 입력 길이에 맞춤
                min_length=1,
                num_beams=4,  # 빔 서치 줄임
                early_stopping=True,
                no_repeat_ngram_size=3,  # 반복 방지
                length_penalty=1.0,  # 길이 패널티
                do_sample=False,  # 결정적 생성
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.tokenizer.convert_tokens_to_ids('鈐')]] if '鈐' in self.tokenizer.get_vocab() else None
            )
        
        # 디코딩
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 후처리: 이상한 문자 제거
        corrected = self.clean_output(corrected)
        
        return corrected
    
    def clean_output(self, text):
        """출력 정제"""
        # 이상한 문자 제거
        weird_chars = ['鈐', 'î', '�', '<unk>', '[UNK]']
        for char in weird_chars:
            text = text.replace(char, '')
        
        # 중복 제거 (연속된 같은 단어)
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
                prev_word = word
        
        return ' '.join(cleaned_words).strip()
    
    def test_examples(self):
        """테스트"""
        test_cases = [
            "안녕하세여 반갑습니다",
            "오늘 날씨가 너무 좋아서 기분이 조아요",
            "맞춤법을 틀리게 쓰는건 정말 어려워여",
            "이것도 고쳐줄수있나?",
            "공부를 열심히 햇더니 성적이 올랏어요",
        ]
        
        print("\n" + "="*60)
        print("📝 개선된 맞춤법 교정 테스트")
        print("="*60)
        
        for i, text in enumerate(test_cases, 1):
            corrected = self.correct(text)
            
            print(f"\n[{i}]")
            print(f"입력: {text}")
            print(f"교정: {corrected}")
            
            # 변경 사항 표시
            if text != corrected:
                changes = self.highlight_changes(text, corrected)
                if changes:
                    print(f"변경: {changes}")

    def highlight_changes(self, original, corrected):
        """변경 사항 찾기"""
        orig_words = original.split()
        corr_words = corrected.split()
        
        changes = []
        for i, (o, c) in enumerate(zip(orig_words, corr_words)):
            if o != c:
                changes.append(f"{o}→{c}")
        
        return ", ".join(changes[:3]) if changes else None

def main():
    checker = ImprovedSpellChecker()
    checker.test_examples()
    
    print("\n" + "="*60)
    print("💬 인터랙티브 모드 (종료: quit)")
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
