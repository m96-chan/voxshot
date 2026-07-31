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
import { readFileSync } from "node:fs";

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

// A types entry nobody can resolve makes this an untyped package.
const types = pkg.exports?.["."]?.types ?? pkg.types;
if (types && !shipped.has(types.replace(/^\.\//, ""))) {
  problems.push(`types entry ${types} is not in the tarball.`);
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

if (problems.length > 0) {
  console.error(`\nWhat would ship as ${pkg.name}@${pkg.version} is not publishable:\n`);
  for (const problem of problems) console.error(`  - ${problem}`);
  console.error(`\n${packed.files.length} files, ${Math.round(packed.unpackedSize / 1024)} kB.\n`);
  process.exit(1);
}

console.log(
  `${pkg.name}@${pkg.version}: ${packed.files.length} files, ` +
    `${Math.round(packed.unpackedSize / 1024)} kB, license and maps check out.`,
);
