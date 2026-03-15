import type { GuardrailDecision, Profile, ProtectedSpan, TextEdit } from '../pipeline/types';
import { isProtected } from './protectedSpan';

export function guardrailDecision(
  original: string,
  replacement: string,
  range: { start: number; end: number },
  profile: Profile,
  protectedSpans: ProtectedSpan[],
  score: number
): GuardrailDecision {
  const reasonCodes: string[] = [];

  if (isProtected(range, protectedSpans)) {
    return { decision: 'REJECT', reasonCodes: ['PROTECTED_SPAN'], score: 0.01 };
  }
  if (/[A-Za-z0-9]/.test(original) || /[A-Za-z0-9]/.test(replacement)) {
    return { decision: 'REJECT', reasonCodes: ['ALNUM_RISK'], score: 0.05 };
  }

  const ratio = Math.abs(replacement.length - original.length) / Math.max(1, original.length);
  if (ratio > 0.25) reasonCodes.push('EDIT_RATIO_HIGH');
  if (profile === 'CHAT') reasonCodes.push('CHAT_PROFILE');
  reasonCodes.push('LEXICAL_REPLACEMENT');

  return { decision: 'SUGGEST_ONLY', reasonCodes, score };
}

export function policyForEdit(edit: TextEdit, guardrail: GuardrailDecision): GuardrailDecision['decision'] {
  if (guardrail.decision === 'REJECT') return 'REJECT';
  if (edit.editType === 'SPACE_INSERT' || edit.editType === 'SPACE_DELETE') return 'AUTO_APPLY';
  if (edit.editType === 'SPELL' && edit.confidence >= 0.95) return 'AUTO_APPLY';
  return guardrail.decision;
}
