export function normalizeText(text: string): string {
  return text.replace(/[\t\f\v]+/g, ' ').replace(/[ ]{2,}/g, ' ').trim();
}
