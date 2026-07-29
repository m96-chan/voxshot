/**
 * Control characters that carry no phonetic meaning. Tab / LF / CR are kept
 * here on purpose: they are collapsed together with the rest of the
 * whitespace in {@link normalizeText}.
 */
const CONTROL_CHARACTERS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;

/**
 * Canonicalise text before synthesis.
 *
 * - Applies Unicode NFKC, so full width latin becomes half width and half
 *   width katakana becomes full width.
 * - Strips control characters.
 * - Collapses every whitespace run into a single space and trims the result.
 */
export function normalizeText(text: string): string {
  return text
    .normalize("NFKC")
    .replace(CONTROL_CHARACTERS, "")
    .replace(/\s+/g, " ")
    .trim();
}
