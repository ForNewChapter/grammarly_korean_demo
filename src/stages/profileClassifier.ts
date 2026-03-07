import type { Profile } from '../pipeline/types';

export interface ProfileOutput {
  profile: Profile;
  confidence: number;
}

export function classifyProfile(text: string): ProfileOutput {
  const len = Math.max(1, text.length);
  const hangul = (text.match(/[가-힣]/g) ?? []).length / len;
  const latin = (text.match(/[A-Za-z]/g) ?? []).length / len;
  const digits = (text.match(/\d/g) ?? []).length / len;
  const chatSig = /[ㅋㅎㅠ]{2,}|[!?]{2,}|\bㄹㅇ\b|\b개좋/.test(text);
  const spaces = (text.match(/\s/g) ?? []).length;

  let profile: Profile = 'NORMAL';
  if (latin > 0.3 || digits > 0.25) profile = 'MIXED';
  else if (chatSig) profile = 'CHAT';
  else if (hangul < 0.35) profile = 'QUERY';
  else if (spaces === 0 && text.length > 12) profile = 'NOISY';

  return { profile, confidence: 0.82 };
}
