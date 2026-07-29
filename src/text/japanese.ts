/**
 * Kana readings for text-to-speech: numbers, dates, times, units, symbols and
 * Latin acronyms are rewritten into the way a Japanese speaker says them.
 *
 * This is intentionally a separate opt-in step and not part of
 * {@link normalizeText}: the rewrites are Japanese-specific and would corrupt
 * text meant for other languages.
 */

const DIGIT_READINGS = [
  "ぜろ",
  "いち",
  "に",
  "さん",
  "よん",
  "ご",
  "ろく",
  "なな",
  "はち",
  "きゅう",
] as const;

const HUNDREDS: Record<number, string> = { 3: "さんびゃく", 6: "ろっぴゃく", 8: "はっぴゃく" };
const THOUSANDS: Record<number, string> = { 3: "さんぜん", 8: "はっせん" };

/** Group magnitudes for four-digit grouping, smallest first. */
const GROUP_UNITS = ["", "まん", "おく", "ちょう"] as const;

/** Longest integer (in digits) readable with the 万/億/兆 groups. */
const MAX_GROUPED_DIGITS = 16;

const MONTH_READINGS: Record<number, string> = { 4: "しがつ", 7: "しちがつ", 9: "くがつ" };

const DAY_READINGS: Record<number, string> = {
  1: "ついたち",
  2: "ふつか",
  3: "みっか",
  4: "よっか",
  5: "いつか",
  6: "むいか",
  7: "なのか",
  8: "ようか",
  9: "ここのか",
  10: "とおか",
  14: "じゅうよっか",
  20: "はつか",
  24: "にじゅうよっか",
};

const MINUTE_ONES: Record<number, string> = {
  1: "いっぷん",
  2: "にふん",
  3: "さんぷん",
  4: "よんぷん",
  5: "ごふん",
  6: "ろっぷん",
  7: "ななふん",
  8: "はっぷん",
  9: "きゅうふん",
};

/** Units read by plain concatenation after the number reading. */
const UNIT_READINGS: Record<string, string> = {
  円: "えん",
  "%": "パーセント",
  "％": "パーセント",
  "℃": "ど",
  km: "キロメートル",
  kg: "キログラム",
  cm: "センチメートル",
  mm: "ミリメートル",
  m: "メートル",
  g: "グラム",
  秒: "びょう",
};

const LETTER_READINGS: Record<string, string> = {
  A: "エー",
  B: "ビー",
  C: "シー",
  D: "ディー",
  E: "イー",
  F: "エフ",
  G: "ジー",
  H: "エイチ",
  I: "アイ",
  J: "ジェー",
  K: "ケー",
  L: "エル",
  M: "エム",
  N: "エヌ",
  O: "オー",
  P: "ピー",
  Q: "キュー",
  R: "アール",
  S: "エス",
  T: "ティー",
  U: "ユー",
  V: "ブイ",
  W: "ダブリュー",
  X: "エックス",
  Y: "ワイ",
  Z: "ゼット",
};

const SIGN_READINGS: Record<string, string> = {
  "-": "マイナス",
  "−": "マイナス",
  "+": "プラス",
  "＋": "プラス",
  "±": "プラスマイナス",
};

/** `12:30` style clock times; `00` minutes / seconds are silent. */
const CLOCK_TIME = /(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)/g;

/** `~` between two numbers reads as a range. */
const NUMERIC_RANGE = /(?<=\d)\s*[~〜]\s*(?=[-−+＋±]?\d)/g;

/** `-` between two digit runs joins them, phone-number style. */
const NUMERIC_HYPHEN = /(?<=\d)[-−](?=\d)/g;

/** A number with optional sign, grouping commas, decimals and unit suffix. */
const NUMBER_TOKEN =
  /([-−+＋±])?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(年|月|日|時|分|円|km|kg|cm|mm|m|g|%|％|℃|秒)?/g;

const ACRONYM = /[A-Z]{2,}/g;

/** Read a string of digits one by one, e.g. `"03"` → `ぜろさん`. */
function readDigitByDigit(digits: string): string {
  let reading = "";
  for (const digit of digits) {
    reading += DIGIT_READINGS[Number(digit)];
  }
  return reading;
}

/** Read 1–9999 with the euphonic changes for hundreds and thousands. */
function readGroup(value: number): string {
  let reading = "";

  const thousands = Math.floor(value / 1000);
  if (thousands > 0) {
    reading +=
      THOUSANDS[thousands] ?? (thousands === 1 ? "せん" : `${DIGIT_READINGS[thousands]}せん`);
  }
  const hundreds = Math.floor((value % 1000) / 100);
  if (hundreds > 0) {
    reading +=
      HUNDREDS[hundreds] ?? (hundreds === 1 ? "ひゃく" : `${DIGIT_READINGS[hundreds]}ひゃく`);
  }
  const tens = Math.floor((value % 100) / 10);
  if (tens > 0) {
    reading += tens === 1 ? "じゅう" : `${DIGIT_READINGS[tens]}じゅう`;
  }
  const ones = value % 10;
  if (ones > 0) {
    reading += DIGIT_READINGS[ones];
  }
  return reading;
}

/** ちょう triggers a sokuon on いち / はち / じゅう (`1兆` → いっちょう). */
function attachGroupUnit(groupReading: string, unit: string): string {
  if (unit === "ちょう") {
    if (groupReading.endsWith("いち")) {
      return `${groupReading.slice(0, -2)}いっ${unit}`;
    }
    if (groupReading.endsWith("はち")) {
      return `${groupReading.slice(0, -2)}はっ${unit}`;
    }
    if (groupReading.endsWith("じゅう")) {
      return `${groupReading.slice(0, -3)}じゅっ${unit}`;
    }
  }
  return groupReading + unit;
}

/**
 * Read a non-negative integer given as a digit string.
 *
 * Falls back to digit-by-digit beyond the 兆 range, and for tokens with a
 * leading zero (`03`), which are labels rather than quantities.
 */
function readInteger(digits: string): string {
  if (digits.length > MAX_GROUPED_DIGITS || (digits.length > 1 && digits.startsWith("0"))) {
    return readDigitByDigit(digits);
  }
  const value = Number(digits);
  if (value === 0) {
    return "ぜろ";
  }

  let reading = "";
  for (let group = 0; group * 4 < digits.length; group += 1) {
    const start = Math.max(0, digits.length - (group + 1) * 4);
    const groupValue = Number(digits.slice(start, digits.length - group * 4));
    if (groupValue > 0) {
      reading = attachGroupUnit(readGroup(groupValue), GROUP_UNITS[group] as string) + reading;
    }
  }
  return reading;
}

/** `4年` → よねん; likewise for every year ending in 4. */
function readYear(value: number): string {
  if (value % 10 === 4) {
    return `${value > 4 ? readInteger(String(value - 4)) : ""}よねん`;
  }
  return `${readInteger(String(value))}ねん`;
}

function readMonth(value: number): string {
  return MONTH_READINGS[value] ?? `${readInteger(String(value))}がつ`;
}

function readDay(value: number): string {
  return DAY_READINGS[value] ?? `${readInteger(String(value))}にち`;
}

/** `4時` → よじ, `7時` → しちじ, `9時` → くじ, also in 14時 / 19時 / 24時. */
function readHour(value: number): string {
  const ones = value % 10;
  const irregular: Record<number, string> = { 4: "よじ", 7: "しちじ", 9: "くじ" };
  const suffix = irregular[ones];
  if (suffix !== undefined) {
    return `${value > ones ? readInteger(String(value - ones)) : ""}${suffix}`;
  }
  return value === 0 ? "れいじ" : `${readInteger(String(value))}じ`;
}

function readMinute(value: number): string {
  if (value === 0) {
    return "れいふん";
  }
  if (value % 10 === 0) {
    const tens = readInteger(String(value));
    return `${tens.slice(0, -3)}じゅっぷん`;
  }
  const ones = value % 10;
  return `${value > ones ? readInteger(String(value - ones)) : ""}${MINUTE_ONES[ones]}`;
}

/** Suffixes whose reading depends on the numeric value. */
const VALUE_SUFFIXES: Record<string, (value: number) => string> = {
  年: readYear,
  月: readMonth,
  日: readDay,
  時: readHour,
  分: readMinute,
};

function readNumberToken(
  sign: string | undefined,
  integer: string,
  fraction: string | undefined,
  suffix: string | undefined,
): string {
  const digits = integer.replace(/,/g, "");
  const signReading = sign ? (SIGN_READINGS[sign] as string) : "";

  const valueSuffix = suffix ? VALUE_SUFFIXES[suffix] : undefined;
  if (valueSuffix && fraction === undefined && digits.length <= MAX_GROUPED_DIGITS) {
    return signReading + valueSuffix(Number(digits));
  }

  let reading = signReading + readInteger(digits);
  if (fraction !== undefined) {
    reading += `てん${readDigitByDigit(fraction)}`;
  }
  if (suffix) {
    reading += UNIT_READINGS[suffix] ?? suffix;
  }
  return reading;
}

function readClockTime(hours: string, minutes: string, seconds: string | undefined): string {
  let reading = readHour(Number(hours));
  if (Number(minutes) > 0) {
    reading += readMinute(Number(minutes));
  }
  if (seconds !== undefined && Number(seconds) > 0) {
    reading += `${readInteger(String(Number(seconds)))}びょう`;
  }
  return reading;
}

/**
 * Rewrite numbers, dates, times, units, numeric symbols and upper-case
 * acronyms into kana readings, leaving everything else untouched.
 *
 * ```ts
 * toJapaneseReading("1,000円");            // "せんえん"
 * toJapaneseReading("会議は3月4日の14:00"); // "会議はさんがつよっかのじゅうよじ"
 * toJapaneseReading("AIが50%");            // "エーアイがごじゅうパーセント"
 * ```
 */
export function toJapaneseReading(text: string): string {
  return text
    .replace(NUMERIC_RANGE, "から")
    .replace(NUMERIC_HYPHEN, "の")
    .replace(CLOCK_TIME, (_, hours: string, minutes: string, seconds: string | undefined) =>
      readClockTime(hours, minutes, seconds),
    )
    .replace(
      NUMBER_TOKEN,
      (
        _,
        sign: string | undefined,
        integer: string,
        fraction: string | undefined,
        suffix: string | undefined,
      ) => readNumberToken(sign, integer, fraction, suffix),
    )
    .replace(ACRONYM, (letters) => {
      let reading = "";
      for (const letter of letters) {
        reading += LETTER_READINGS[letter];
      }
      return reading;
    });
}
