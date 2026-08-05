import { describe, expect, it } from 'vitest';
import { canonicalSha256, sha256Hex, stableStringify } from './provenance';

describe('provenance hashing', () => {
  it('matches standard SHA-256 vectors', () => {
    expect(sha256Hex('')).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(sha256Hex('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });

  it('canonicalizes object keys while retaining array order', () => {
    expect(stableStringify({ b: 2, a: 1 })).toBe('{"a":1,"b":2}');
    expect(canonicalSha256({ b: 2, a: [1, 2] })).toBe(canonicalSha256({ a: [1, 2], b: 2 }));
    expect(canonicalSha256({ a: [2, 1], b: 2 })).not.toBe(canonicalSha256({ a: [1, 2], b: 2 }));
  });
});
