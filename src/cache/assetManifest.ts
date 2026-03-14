import { resolveAssetUrl } from './assetUrl';

export const MODEL_ASSETS = [
  resolveAssetUrl('/assets/models/profile_classifier.onnx'),
  resolveAssetUrl('/assets/models/spacing_boundary.onnx'),
  resolveAssetUrl('/assets/models/edit_tagger.onnx'),
  resolveAssetUrl('/assets/models/reranker.onnx'),
  resolveAssetUrl('/assets/models/guardrail.onnx'),
] as const;

export const RULE_ASSETS = [
  resolveAssetUrl('/assets/rules/common_misspellings.json'),
  resolveAssetUrl('/assets/rules/high_precision_surface_fixes.json'),
  resolveAssetUrl('/assets/rules/protected_patterns.json'),
  resolveAssetUrl('/assets/rules/thresholds.json'),
  resolveAssetUrl('/assets/dict/confusion_sets.json'),
  resolveAssetUrl('/assets/dict/typed_confusion_graph.json'),
  resolveAssetUrl('/assets/dict/inflection_recovery_rules.json'),
  resolveAssetUrl('/assets/dict/phrase_memory.json'),
  resolveAssetUrl('/assets/dict/predicate_family_seeds.json'),
] as const;
