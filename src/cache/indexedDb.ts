import { openDB } from 'idb';
import type { StageTrace } from '../pipeline/types';

interface TraceRecord {
  id?: number;
  createdAt: number;
  original: string;
  corrected: string;
  traces: StageTrace[];
}

interface SettingsRecord {
  key: string;
  value: unknown;
}

const DB_NAME = 'ko-proof-demo';
const DB_VERSION = 1;

const dbPromise = openDB(DB_NAME, DB_VERSION, {
  upgrade(db) {
    if (!db.objectStoreNames.contains('traces')) {
      db.createObjectStore('traces', { keyPath: 'id', autoIncrement: true });
    }
    if (!db.objectStoreNames.contains('settings')) {
      db.createObjectStore('settings', { keyPath: 'key' });
    }
  },
});

export async function saveTraceSession(original: string, corrected: string, traces: StageTrace[]): Promise<void> {
  const db = await dbPromise;
  await db.add('traces', { createdAt: Date.now(), original, corrected, traces } satisfies TraceRecord);
}

export async function getRecentTraceSessions(limit = 20): Promise<TraceRecord[]> {
  const db = await dbPromise;
  const all = (await db.getAll('traces')) as TraceRecord[];
  return all.sort((a, b) => b.createdAt - a.createdAt).slice(0, limit);
}

export async function setSetting(key: string, value: unknown): Promise<void> {
  const db = await dbPromise;
  await db.put('settings', { key, value } satisfies SettingsRecord);
}

export async function getSetting<T>(key: string, fallback: T): Promise<T> {
  const db = await dbPromise;
  const row = (await db.get('settings', key)) as SettingsRecord | undefined;
  return (row?.value as T) ?? fallback;
}
