/**
 * Verify what `npm publish` would actually ship.
 *
 * `npm test` cannot see any of this. It checks the repository; consumers get a
 * tarball, and the two differ by whatever `files` happens to say. Both defects
 * this guards against shipped in 0.2.0 with a fully green suite (#98).
 *
 * Packs for real rather than reading `files`, because reading the declaration
 * is how the maps came to point at a directory that was never included.
 */
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));

execFileSync("npm", ["run", "build"], { stdio: "inherit" });
const packed = JSON.parse(
  execFileSync("npm", ["pack", "--dry-run", "--json"], { encoding: "utf8", maxBuffer: 1 << 24 }),
)[0];
const shipped = new Set(packed.files.map((file) => file.path));

const problems = [];

// A license declaration in package.json is metadata, not a grant.
if (![...shipped].some((path) => /^LICEN[CS]E(\.\w+)?$/i.test(path))) {
  problems.push(`package.json declares "license": "${pkg.license}" but no license text is shipped.`);
}

// Every target of every export entry, not just the root's `types`.
//
// The narrow version of this check only looked at `exports["."].types`, which
// was the whole surface until `./miotts` existed. A subpath is exactly where
// this goes wrong quietly: the repository resolves it through `src`, the suite
// is green, `npm run build` succeeds, and the entry points into `dist` at a
// path `files` never carried. Naming the entry matters too — "types entry is
// not in the tarball" does not say which of them.
for (const [entry, target] of Object.entries(pkg.exports ?? { ".": pkg.types })) {
  const targets = typeof target === "string" ? { default: target } : target;
  for (const [condition, path] of Object.entries(targets ?? {})) {
    if (typeof path !== "string") continue;
    if (!shipped.has(path.replace(/^\.\//, ""))) {
      problems.push(`exports["${entry}"].${condition} points at ${path}, which is not in the tarball.`);
    }
  }
}

// A map whose sources are absent is worse than no map: same debugging, more bytes.
for (const path of shipped) {
  if (!path.endsWith(".map")) continue;
  const map = JSON.parse(readFileSync(new URL(`../${path}`, import.meta.url), "utf8"));
  if (map.sourcesContent?.length) continue;
  for (const source of map.sources ?? []) {
    // Map sources are relative to the map's own directory.
    const resolved = new URL(source, new URL(`../${path}`, import.meta.url)).pathname;
    const relative = resolved.slice(new URL("../", import.meta.url).pathname.length);
    if (!shipped.has(relative)) {
      problems.push(`${path} points at ${relative}, which is not shipped.`);
    }
  }
}

// Then the part reading `files` and `exports` cannot tell you: whether Node,
// standing in a consumer's directory, actually resolves each entry.
//
// The checks above read declarations. This one packs the real tarball, unpacks
// it into a throwaway `node_modules` and asks Node to import every subpath, so
// an `exports` map that is internally consistent but wrong — a condition order
// Node rejects, a directory `files` drops, an extension that does not match
// what was emitted — fails here instead of in someone else's install.
//
// No `npm install`: extracting into `node_modules/<name>` is the same thing
// for resolution purposes, and it neither reaches the network nor lets npm
// pull peer dependencies in and change what is being tested.
const tmp = mkdtempSync(join(tmpdir(), "voxshot-pack-"));
try {
  const tarball = execFileSync("npm", ["pack", "--pack-destination", tmp], {
    encoding: "utf8",
    // npm writes its file listing to stderr; the tarball name is the payload.
    stdio: ["ignore", "pipe", "ignore"],
  })
    .trim()
    .split("\n")
    .at(-1);
  const installed = join(tmp, "node_modules", pkg.name);
  mkdirSync(installed, { recursive: true });
  // `--strip-components=1` drops the "package/" prefix npm wraps a tarball in.
  execFileSync("tar", ["-xzf", join(tmp, tarball), "-C", installed, "--strip-components=1"]);
  writeFileSync(join(tmp, "package.json"), JSON.stringify({ type: "module" }));

  // Declared peers are linked in, because a consumer installs them too — an
  // optional peer that is absent is a documented requirement, not a packaging
  // defect. Only what `peerDependencies` actually names: an import of anything
  // that is neither bundled nor declared still fails here, which is the point.
  for (const name of Object.keys(pkg.peerDependencies ?? {})) {
    const source = new URL(`../node_modules/${name}`, import.meta.url).pathname;
    if (!existsSync(source)) continue;
    const target = join(tmp, "node_modules", name);
    mkdirSync(dirname(target), { recursive: true });
    symlinkSync(source, target, "dir");
  }

  for (const entry of Object.keys(pkg.exports ?? { ".": null })) {
    const specifier = entry === "." ? pkg.name : `${pkg.name}/${entry.replace(/^\.\//, "")}`;
    try {
      execFileSync(process.execPath, ["--input-type=module", "-e", `await import(${JSON.stringify(specifier)})`], {
        cwd: tmp,
        stdio: "pipe",
      });
    } catch (error) {
      // Node leads its stack with the frame, not the reason. Report the first
      // line that carries an error code or message, so the failure names what
      // is wrong instead of an offset into node:internal.
      const lines = String(error.stderr ?? error.message).trim().split("\n");
      const detail = lines.find((line) => /Error|\[ERR_/.test(line))?.trim() ?? lines[0];
      problems.push(`a consumer cannot import "${specifier}" from the tarball: ${detail}`);
    }
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

if (problems.length > 0) {
  console.error(`\nWhat would ship as ${pkg.name}@${pkg.version} is not publishable:\n`);
  for (const problem of problems) console.error(`  - ${problem}`);
  console.error(`\n${packed.files.length} files, ${Math.round(packed.unpackedSize / 1024)} kB.\n`);
  process.exit(1);
}

console.log(
  `${pkg.name}@${pkg.version}: ${packed.files.length} files, ` +
    `${Math.round(packed.unpackedSize / 1024)} kB; license, maps and all ` +
    `${Object.keys(pkg.exports ?? {}).length} export entries resolve from the tarball.`,
);
