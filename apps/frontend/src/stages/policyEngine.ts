import type { GuardrailDecision, TextEdit } from '../pipeline/types';

export interface PolicyOutput {
  autoApply: TextEdit[];
  suggestOnly: TextEdit[];
  reject: TextEdit[];
  finalGuardrail: GuardrailDecision;
}

export function runPolicyEngine(edits: Array<{ edit: TextEdit; decision: GuardrailDecision['decision'] }>): PolicyOutput {
  const autoApply: TextEdit[] = [];
  const suggestOnly: TextEdit[] = [];
  const reject: TextEdit[] = [];

  for (const item of edits) {
    if (item.decision === 'AUTO_APPLY') autoApply.push(item.edit);
    else if (item.decision === 'SUGGEST_ONLY') suggestOnly.push(item.edit);
    else reject.push(item.edit);
  }

  const finalGuardrail: GuardrailDecision = reject.length
    ? { decision: 'REJECT', reasonCodes: ['HAS_REJECT'], score: 0.2 }
    : suggestOnly.length
      ? { decision: 'SUGGEST_ONLY', reasonCodes: ['HAS_SUGGEST'], score: 0.7 }
      : { decision: 'AUTO_APPLY', reasonCodes: ['ALL_SAFE'], score: 0.95 };

  return { autoApply, suggestOnly, reject, finalGuardrail };
}
