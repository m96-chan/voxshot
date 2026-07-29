import { ZeroVoxError } from "../errors.js";
import type { VoiceEmbedding, VoiceStore, VoiceTensor } from "./types.js";
import { assertEmbedding, assertVoiceName, cloneEmbedding } from "./validate.js";

const DEFAULT_DATABASE_NAME = "zerovox";
const OBJECT_STORE_NAME = "voices";

export interface IndexedDbVoiceStoreOptions {
  /**
   * Database name, so applications can isolate voices per user or profile.
   *
   * @defaultValue "zerovox"
   */
  databaseName?: string;
}

/** Row shape written to IndexedDB. */
interface VoiceRecord {
  name: string;
  vector: Float32Array;
  sampleRate: number;
  createdAt: number;
  engine?: string;
  tensors?: Record<string, VoiceTensor>;
}

/**
 * Voice store backed by IndexedDB, so cloned voices survive a page reload.
 *
 * Embedding vectors are stored as `Float32Array` values and go through the
 * structured clone algorithm untouched — no base64 round trip.
 */
export class IndexedDbVoiceStore implements VoiceStore {
  readonly #databaseName: string;
  #database: IDBDatabase | undefined;

  constructor(options: IndexedDbVoiceStoreOptions = {}) {
    this.#databaseName = options.databaseName ?? DEFAULT_DATABASE_NAME;
  }

  /** Whether IndexedDB exists in this environment. */
  static isSupported(): boolean {
    return typeof globalThis.indexedDB !== "undefined" && globalThis.indexedDB !== null;
  }

  async save(name: string, embedding: VoiceEmbedding): Promise<void> {
    const key = assertVoiceName(name);
    const copy = cloneEmbedding(assertEmbedding(embedding));
    const record: VoiceRecord = {
      name: key,
      vector: copy.vector,
      sampleRate: copy.sampleRate,
      createdAt: copy.createdAt,
    };
    if (copy.engine !== undefined) {
      record.engine = copy.engine;
    }
    if (copy.tensors) {
      record.tensors = copy.tensors as Record<string, VoiceTensor>;
    }

    await this.#withStore("readwrite", (store) => store.put(record));
  }

  async load(name: string): Promise<VoiceEmbedding | undefined> {
    const key = assertVoiceName(name);
    const record = await this.#withStore<VoiceRecord | undefined>("readonly", (store) =>
      store.get(key),
    );
    if (!record) {
      return undefined;
    }
    return cloneEmbedding(record);
  }

  async list(): Promise<string[]> {
    const keys = await this.#withStore<IDBValidKey[]>("readonly", (store) => store.getAllKeys());
    return keys.map(String).sort();
  }

  async delete(name: string): Promise<boolean> {
    const key = assertVoiceName(name);
    const existing = await this.#withStore<number>("readonly", (store) => store.count(key));
    if (existing === 0) {
      return false;
    }
    await this.#withStore("readwrite", (store) => store.delete(key));
    return true;
  }

  async clear(): Promise<void> {
    await this.#withStore("readwrite", (store) => store.clear());
  }

  /** Close the underlying connection; the next call reopens it. */
  async close(): Promise<void> {
    this.#database?.close();
    this.#database = undefined;
  }

  async #withStore<T>(
    mode: IDBTransactionMode,
    run: (store: IDBObjectStore) => IDBRequest,
  ): Promise<T> {
    const database = await this.#open();
    const transaction = database.transaction(OBJECT_STORE_NAME, mode);
    const request = run(transaction.objectStore(OBJECT_STORE_NAME));

    return new Promise<T>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result as T);
      request.onerror = () => reject(wrap(request.error, "IndexedDB request failed"));
    });
  }

  async #open(): Promise<IDBDatabase> {
    if (this.#database) {
      return this.#database;
    }
    if (!IndexedDbVoiceStore.isSupported()) {
      throw new ZeroVoxError(
        "IndexedDB is not available in this environment. Use MemoryVoiceStore instead.",
      );
    }

    const request = globalThis.indexedDB.open(this.#databaseName, 1);
    this.#database = await new Promise<IDBDatabase>((resolve, reject) => {
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(OBJECT_STORE_NAME)) {
          database.createObjectStore(OBJECT_STORE_NAME, { keyPath: "name" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(wrap(request.error, "Failed to open the voice database"));
    });
    return this.#database;
  }
}

function wrap(cause: unknown, message: string): ZeroVoxError {
  const detail = cause instanceof Error ? `: ${cause.message}` : "";
  return new ZeroVoxError(`${message}${detail}`, "UNKNOWN", { cause });
}
