export const MODEL_ASSETS = [
  '/assets/models/profile_classifier.onnx',
  '/assets/models/spacing_boundary.onnx',
  '/assets/models/edit_tagger.onnx',
  '/assets/models/reranker.onnx',
  '/assets/models/guardrail.onnx',
] as const;

export const RULE_ASSETS = [
  '/assets/rules/common_misspellings.json',
  '/assets/rules/protected_patterns.json',
  '/assets/rules/thresholds.json',
  '/assets/dict/confusion_sets.json',
] as const;
