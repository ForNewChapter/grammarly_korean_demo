export function resolveAssetUrl(path: string): string {
  const trimmed = path.replace(/^\/+/, '');
  const origin =
    typeof globalThis.location !== 'undefined' && globalThis.location?.origin
      ? globalThis.location.origin
      : 'http://localhost';
  return new URL(trimmed, new URL(import.meta.env.BASE_URL, origin)).toString();
}
