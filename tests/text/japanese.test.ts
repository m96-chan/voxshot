import { describe, expect, it } from "vitest";

import { toJapaneseReading } from "../../src/text/japanese.js";

describe("toJapaneseReading", () => {
  describe("integers", () => {
    it("reads single digits", () => {
      expect(toJapaneseReading("0")).toBe("ぜろ");
      expect(toJapaneseReading("1")).toBe("いち");
      expect(toJapaneseReading("7")).toBe("なな");
    });

    it("reads tens, hundreds and thousands without a leading いち", () => {
      expect(toJapaneseReading("10")).toBe("じゅう");
      expect(toJapaneseReading("100")).toBe("ひゃく");
      expect(toJapaneseReading("1000")).toBe("せん");
    });

    it("applies euphonic changes for hundreds and thousands", () => {
      expect(toJapaneseReading("300")).toBe("さんびゃく");
      expect(toJapaneseReading("600")).toBe("ろっぴゃく");
      expect(toJapaneseReading("800")).toBe("はっぴゃく");
      expect(toJapaneseReading("3000")).toBe("さんぜん");
      expect(toJapaneseReading("8000")).toBe("はっせん");
    });

    it("reads composite numbers", () => {
      expect(toJapaneseReading("123")).toBe("ひゃくにじゅうさん");
      expect(toJapaneseReading("2468")).toBe("にせんよんひゃくろくじゅうはち");
    });

    it("reads the 万/億/兆 groups with いち where required", () => {
      expect(toJapaneseReading("10000")).toBe("いちまん");
      expect(toJapaneseReading("100000000")).toBe("いちおく");
      expect(toJapaneseReading("1000000000000")).toBe("いっちょう");
      expect(toJapaneseReading("8000000000000")).toBe("はっちょう");
      expect(toJapaneseReading("10000000000000")).toBe("じゅっちょう");
      expect(toJapaneseReading("20030")).toBe("にまんさんじゅう");
    });

    it("strips digit-grouping commas before reading", () => {
      expect(toJapaneseReading("1,000")).toBe("せん");
      expect(toJapaneseReading("12,345")).toBe("いちまんにせんさんびゃくよんじゅうご");
    });

    it("falls back to digit-by-digit for absurdly long numbers", () => {
      expect(toJapaneseReading("12345678901234567890")).toBe(
        "いちにさんよんごろくななはちきゅうぜろいちにさんよんごろくななはちきゅうぜろ",
      );
    });
  });

  describe("decimals", () => {
    it("reads the fraction digit by digit after てん", () => {
      expect(toJapaneseReading("3.14")).toBe("さんてんいちよん");
      expect(toJapaneseReading("0.5")).toBe("ぜろてんご");
    });
  });

  describe("signs and ranges", () => {
    it("reads a leading minus as マイナス", () => {
      expect(toJapaneseReading("-5")).toBe("マイナスご");
      expect(toJapaneseReading("気温は-3度")).toBe("気温はマイナスさん度");
    });

    it("reads a leading plus as プラス", () => {
      expect(toJapaneseReading("+3")).toBe("プラスさん");
    });

    it("reads ~ between numbers as から", () => {
      expect(toJapaneseReading("1~3")).toBe("いちからさん");
      expect(toJapaneseReading("10〜20")).toBe("じゅうからにじゅう");
    });

    it("reads a hyphen between digit runs as の", () => {
      expect(toJapaneseReading("03-1234")).toBe("ぜろさんのせんにひゃくさんじゅうよん");
    });
  });

  describe("dates", () => {
    it("reads a full date with special month and day readings", () => {
      expect(toJapaneseReading("2024年1月5日")).toBe("にせんにじゅうよねんいちがついつか");
    });

    it("reads 4月 7月 9月 irregularly", () => {
      expect(toJapaneseReading("4月")).toBe("しがつ");
      expect(toJapaneseReading("7月")).toBe("しちがつ");
      expect(toJapaneseReading("9月")).toBe("くがつ");
    });

    it("reads the traditional day names", () => {
      expect(toJapaneseReading("1日")).toBe("ついたち");
      expect(toJapaneseReading("8日")).toBe("ようか");
      expect(toJapaneseReading("10日")).toBe("とおか");
      expect(toJapaneseReading("14日")).toBe("じゅうよっか");
      expect(toJapaneseReading("20日")).toBe("はつか");
      expect(toJapaneseReading("24日")).toBe("にじゅうよっか");
      expect(toJapaneseReading("11日")).toBe("じゅういちにち");
    });

    it("reads years ending in 4 with よねん", () => {
      expect(toJapaneseReading("4年")).toBe("よねん");
      expect(toJapaneseReading("2014年")).toBe("にせんじゅうよねん");
      expect(toJapaneseReading("3年")).toBe("さんねん");
    });
  });

  describe("times", () => {
    it("reads HH:MM clock times", () => {
      expect(toJapaneseReading("12:30")).toBe("じゅうにじさんじゅっぷん");
      expect(toJapaneseReading("9:05")).toBe("くじごふん");
    });

    it("reads HH:MM:SS clock times", () => {
      expect(toJapaneseReading("1:02:03")).toBe("いちじにふんさんびょう");
    });

    it("reads 時 with the irregular 4時 7時 9時", () => {
      expect(toJapaneseReading("4時")).toBe("よじ");
      expect(toJapaneseReading("7時")).toBe("しちじ");
      expect(toJapaneseReading("9時")).toBe("くじ");
      expect(toJapaneseReading("19時")).toBe("じゅうくじ");
    });

    it("reads 分 with ぷん/ふん euphonics", () => {
      expect(toJapaneseReading("1分")).toBe("いっぷん");
      expect(toJapaneseReading("2分")).toBe("にふん");
      expect(toJapaneseReading("3分")).toBe("さんぷん");
      expect(toJapaneseReading("6分")).toBe("ろっぷん");
      expect(toJapaneseReading("8分")).toBe("はっぷん");
      expect(toJapaneseReading("10分")).toBe("じゅっぷん");
      expect(toJapaneseReading("15分")).toBe("じゅうごふん");
      expect(toJapaneseReading("30分")).toBe("さんじゅっぷん");
      expect(toJapaneseReading("0分")).toBe("れいふん");
    });
  });

  describe("units", () => {
    it("reads currency", () => {
      expect(toJapaneseReading("1,000円")).toBe("せんえん");
      expect(toJapaneseReading("250円")).toBe("にひゃくごじゅうえん");
    });

    it("reads percent", () => {
      expect(toJapaneseReading("50%")).toBe("ごじゅうパーセント");
      expect(toJapaneseReading("3.5%")).toBe("さんてんごパーセント");
    });

    it("reads metric units", () => {
      expect(toJapaneseReading("5km")).toBe("ごキロメートル");
      expect(toJapaneseReading("3kg")).toBe("さんキログラム");
      expect(toJapaneseReading("20cm")).toBe("にじゅうセンチメートル");
      expect(toJapaneseReading("7mm")).toBe("ななミリメートル");
      expect(toJapaneseReading("100m")).toBe("ひゃくメートル");
      expect(toJapaneseReading("60g")).toBe("ろくじゅうグラム");
    });

    it("reads temperature degrees", () => {
      expect(toJapaneseReading("25℃")).toBe("にじゅうごど");
    });
  });

  describe("acronyms", () => {
    it("spells upper-case runs letter by letter in katakana", () => {
      expect(toJapaneseReading("AI")).toBe("エーアイ");
      expect(toJapaneseReading("URL")).toBe("ユーアールエル");
      expect(toJapaneseReading("NHK")).toBe("エヌエイチケー");
    });

    it("leaves mixed-case words and single capitals alone", () => {
      expect(toJapaneseReading("Hello")).toBe("Hello");
      expect(toJapaneseReading("X")).toBe("X");
    });
  });

  describe("mixed text", () => {
    it("converts inside a sentence and leaves the rest untouched", () => {
      expect(toJapaneseReading("会議は3月4日の14:00からです。")).toBe(
        "会議はさんがつよっかのじゅうよじからです。",
      );
    });

    it("handles the issue examples", () => {
      expect(toJapaneseReading("123")).toBe("ひゃくにじゅうさん");
      expect(toJapaneseReading("1,000円")).toBe("せんえん");
      expect(toJapaneseReading("AIが50%の確率で答える")).toBe(
        "エーアイがごじゅうパーセントの確率で答える",
      );
    });

    it("returns text without convertible tokens unchanged", () => {
      expect(toJapaneseReading("こんにちは、世界。")).toBe("こんにちは、世界。");
      expect(toJapaneseReading("")).toBe("");
    });
  });
});
