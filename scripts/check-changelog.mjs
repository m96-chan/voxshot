/**
 * Verify that a released section says only what that release could have said.
 *
 * `CHANGELOG.md` documented two fixes under `0.3.0` that landed after it was
 * cut (#113). The insertion looked for a `### Fixed` heading after
 * `## [Unreleased]` without bounding the search to that section, found the one
 * belonging to `0.3.0`, and wrote there. Every step looked right — the file
 * changed, the diff showed an addition under a `### Fixed` — just four sections
 * lower than intended.
 *
 * Asking git *when each line was written* is what catches that. An earlier
 * version matched issue numbers in commit messages instead and produced a false
 * positive on the first run: squash merges name the pull request, not the issue.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const git = (...args) => execFileSync("git", args, { encoding: "utf8" }).trim();
const lines = readFileSync(new URL("../CHANGELOG.md", import.meta.url), "utf8").split("\n");

const tags = new Set(git("tag", "--list").split("\n").filter(Boolean));

// `git blame` once, so the check costs one process rather than one per line.
const blame = new Map();
for (const entry of git("blame", "--line-porcelain", "CHANGELOG.md").split("\n")) {
  const header = entry.match(/^([0-9a-f]{40}) \d+ (\d+)/);
  if (header) blame.set(Number(header[2]), header[1]);
}

const problems = [];
let checked = 0;
let section = null;

lines.forEach((line, index) => {
  const heading = line.match(/^## \[([^\]]+)\]/);
  if (heading) {
    const tag = `v${heading[1]}`;
    section =
      heading[1] === "Unreleased" || !tags.has(tag)
        ? null
        : { name: heading[1], tag, writtenIn: blame.get(index + 1) };
    return;
  }
  if (!section || !line.startsWith("- ")) return;

  const sha = blame.get(index + 1);
  if (!sha) return;

  // Written in the same commit as its heading: the section was backfilled,
  // which is how 0.1.0 and 0.2.0 came to be documented at all (#58). Legitimate,
  // and not what this is looking for.
  if (sha === section.writtenIn) return;

  checked += 1;

  // `--is-ancestor` exits non-zero rather than printing, so the throw is the
  // answer: this line was written after the tag, and the release it sits under
  // cannot have contained it.
  try {
    execFileSync("git", ["merge-base", "--is-ancestor", sha, section.tag]);
  } catch {
    problems.push(
      `${section.name} (line ${index + 1}): written in ${sha.slice(0, 7)}, after ${section.tag}.\n      ${line.slice(0, 76)}`,
    );
  }
});

// A shallow, tagless checkout leaves nothing to compare against, and this
// would then report success having inspected nothing — the failure mode a
// check like this is most likely to die of. Say so instead.
if (checked === 0) {
  console.error(
    "\nCHANGELOG: no released section could be checked." +
      "\n  Tags found: " + (tags.size || "none") +
      "\n  This needs full history and tags — `fetch-depth: 0` in CI.\n",
  );
  process.exit(1);
}

if (problems.length > 0) {
  console.error("\nCHANGELOG credits a release with work that postdates it:\n");
  for (const problem of problems) console.error(`  - ${problem}`);
  console.error("");
  process.exit(1);
}

console.log(`CHANGELOG: ${checked} released entr${checked === 1 ? "y" : "ies"} predate their tags.`);
