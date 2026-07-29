/**
 * `Array.from` cannot take a `Float32Array | BigInt64Array` union, so tests
 * that assert on tensor payloads go through this instead.
 */
export function toArray(
  data: Float32Array | BigInt64Array | undefined,
): (number | bigint)[] {
  if (!data) {
    return [];
  }
  const values: (number | bigint)[] = [];
  for (const value of data) {
    values.push(value);
  }
  return values;
}
