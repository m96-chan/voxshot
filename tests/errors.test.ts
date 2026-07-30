import { describe, expect, it } from "vitest";

import {
  AudioDecodeError,
  DeviceUnavailableError,
  DisposedError,
  InvalidInputError,
  NoVoiceError,
  VoiceNotFoundError,
  VoxShotError,
  isVoxShotError,
} from "../src/errors.js";

describe("VoxShotError", () => {
  it("carries a machine readable code and a message", () => {
    const error = new VoxShotError("something broke", "UNKNOWN");

    expect(error.message).toBe("something broke");
    expect(error.code).toBe("UNKNOWN");
    expect(error.name).toBe("VoxShotError");
    expect(error).toBeInstanceOf(Error);
  });

  it("keeps the original error as cause when provided", () => {
    const cause = new Error("root cause");
    const error = new VoxShotError("wrapped", "UNKNOWN", { cause });

    expect(error.cause).toBe(cause);
  });
});

describe("error subclasses", () => {
  it.each([
    [new DeviceUnavailableError("webgpu"), "DEVICE_UNAVAILABLE", "DeviceUnavailableError"],
    [new NoVoiceError(), "NO_VOICE", "NoVoiceError"],
    [new VoiceNotFoundError("alice"), "VOICE_NOT_FOUND", "VoiceNotFoundError"],
    [new InvalidInputError("text is empty"), "INVALID_INPUT", "InvalidInputError"],
    [new AudioDecodeError("bad header"), "AUDIO_DECODE_FAILED", "AudioDecodeError"],
    [new DisposedError(), "DISPOSED", "DisposedError"],
  ])("%o exposes its code and name", (error, code, name) => {
    expect(error).toBeInstanceOf(VoxShotError);
    expect(error.code).toBe(code);
    expect(error.name).toBe(name);
    expect(error.message.length).toBeGreaterThan(0);
  });

  it("mentions the requested device in DeviceUnavailableError", () => {
    expect(new DeviceUnavailableError("webgpu").message).toContain("webgpu");
  });

  it("mentions the missing voice name in VoiceNotFoundError", () => {
    const error = new VoiceNotFoundError("alice");

    expect(error.message).toContain("alice");
    expect(error.voiceName).toBe("alice");
  });
});

describe("isVoxShotError", () => {
  it("returns true for library errors", () => {
    expect(isVoxShotError(new NoVoiceError())).toBe(true);
  });

  it("returns false for anything else", () => {
    expect(isVoxShotError(new Error("plain"))).toBe(false);
    expect(isVoxShotError("NO_VOICE")).toBe(false);
    expect(isVoxShotError(null)).toBe(false);
  });
});
