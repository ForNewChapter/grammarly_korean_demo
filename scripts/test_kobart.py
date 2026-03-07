import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import time

def test_kobart():
    print("="*60)
    print("🧪 KoBART 모델 테스트")
    print("="*60)
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple M3 Pro GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    print("\n📚 KoBART 모델 로딩 중...")
    start_time = time.time()
    
    # 모델과 토크나이저 로드
    model_name = "gogamza/kobart-base-v2"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # MPS로 이동
    model = model.to(device)
    model.eval()
    
    load_time = time.time() - start_time
    print(f"✅ 모델 로드 완료 ({load_time:.2f}초)")
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 모델 정보:")
    print(f"  • 총 파라미터: {total_params/1e6:.2f}M")
    print(f"  • 디바이스: {device}")
    
    # 간단한 추론 테스트
    print("\n🔤 추론 테스트:")
    test_texts = [
        "안녕하세여 반갑습니다",
        "오늘 날씨가 너무 좋아서 기분이 조아요",
        "이렇게 공부하니까 실력이 늘어나는것 같아여"
    ]
    
    for text in test_texts:
        # 토크나이징 (token_type_ids 제거)
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
        
        # token_type_ids가 있다면 제거
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # 디바이스로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        inference_time = (time.time() - start) * 1000
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  입력: {text}")
        print(f"  출력: {result}")
        print(f"  시간: {inference_time:.2f}ms")
    
    print("\n✅ 테스트 완료! 모델이 정상 작동합니다.")

if __name__ == "__main__":
    test_kobart()
