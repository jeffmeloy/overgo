# Plan — eliminate non-stdlib Go dependencies

Goal: overgo depends only on the Go standard library (no third-party modules).
Applies to overgo + the overgo_gui worktree (same `go.mod`).

## Current dependency surface (measured)

Five **direct** external modules; every `cmd/sbom/main.go` hit is the SBOM
*lister*, not a consumer — the real usage sites are narrow:

| Module | Real usage site(s) | What it does | Pulls indirect? |
|---|---|---|---|
| `gopkg.in/yaml.v3` | `internal/dataroot/dataroot.go`, `internal/server/strict_yaml.go`, `cmd/compatibility` | parse compatibility.yaml + media/resource/dataroot policy YAML | kr/pretty |
| `golang.org/x/sys` | `internal/repodb/lock_{unix,windows}.go` | store file lock (Flock / LockFileEx) | — |
| `golang.org/x/text` | `internal/tokenizer/wpm.go` | Unicode **NFD** normalization (`norm.NFD.String`) | — |
| `github.com/dlclark/regexp2/v2` | `internal/sampling/trigger_regex.go` | ECMAScript-regex for **user-supplied** GBNF trigger patterns (backtracking) | — |
| `github.com/nikolalohinski/gonja/v2` | `internal/inference/chat.go` | Jinja2 chat-template rendering | logrus, json-iterator, go-humanize, modern-go/* |

**Leverage:** all 8 indirect deps come from **gonja** (logrus, json-iterator,
humanize, modern-go/concurrent, modern-go/reflect2, pkg/errors, x/exp) and
**yaml** (kr/pretty). Removing those two clears the entire transitive tree;
regexp2, x/sys, x/text are leaf deps with no transitive pull.

## Per-dependency replacement assessment

### 1. `golang.org/x/sys` → `syscall` + LazyDLL — LOW, doctrine-aligned
- **Unix:** `unix.Flock(fd, LOCK_EX|LOCK_NB)` → stdlib `syscall.Flock(fd, syscall.LOCK_EX|syscall.LOCK_NB)` (in stdlib on all unix GOOS). Direct swap.
- **Windows:** `windows.LockFileEx/UnlockFileEx` → call them via
  `syscall.NewLazyDLL("kernel32.dll").NewProc("LockFileEx")` — the **exact
  no-cgo DLL pattern overgo already uses for CUDA**. Reuse the `Overlapped`
  struct from `syscall` (present) or a local mirror.
- Risk: none functional; keep the `_unix`/`_windows` build-tag split.
- Effort: ~1 slice.

### 2. `golang.org/x/text` (NFD) → generated decomposition table — MEDIUM
- Only `norm.NFD.String` is used (rest of wpm.go is stdlib `unicode`). The
  stdlib has **no** normalization package.
- Replace with a **`go:generate` step that emits a canonical-decomposition
  table** (from the Unicode UCD `UnicodeData.txt`) into a committed `.go` file —
  same pattern the stdlib itself uses for `unicode` tables. Runtime code is a
  pure-stdlib table lookup + recursive canonical decomposition + canonical
  ordering (combining-class sort). No runtime dep; the generator runs offline.
- Scope note: WordPiece/BERT normalization needs NFD only (not full NFC/NFKC),
  so the table is bounded. Pin it with a fixture test (a set of strings whose
  NFD output is snapshotted against the current x/text result before removal).
- Effort: ~1–2 slices (generator + table + parity fixture).

### 3. `gopkg.in/yaml.v3` → migrate config to JSON (`encoding/json`) — MEDIUM
- The stdlib has no YAML. overgo already parses JSON everywhere, so the clean
  path is a **data-format migration**, not a YAML re-implementation:
  - Convert `compatibility.yaml`, `media_policy.yaml`, `resource_policy.yaml`,
    and the dataroot config to `.json` (a one-time offline `yaml→json` pass).
  - Swap the three call sites to `encoding/json`. `strict_yaml.go`'s
    reject-unknown-keys behavior maps to `json.Decoder.DisallowUnknownFields()`.
- Trade-off: JSON loses YAML comments in those files. If comments are
  load-bearing (compatibility.yaml is large + hand-annotated), keep them in a
  sibling `.md` or as `"_comment"` fields; decide per file.
- Kills the kr/pretty indirect dep.
- Effort: ~1–2 slices (format migration + call-site swap + gate on the parsed
  structs round-tripping identically).

### 4. `dlclark/regexp2` → RESOLVED: retained as the single justified exception (option b)
- regexp2 backs **user-supplied** GBNF trigger patterns compiled as ECMAScript
  with backtracking (lookahead/lookbehind/backreferences possible). Go's stdlib
  `regexp` is RE2: **linear-time, no backtracking, no lookaround/backrefs.**
- **Audit result (2026-08):** the ECMAScript feature set is a *tested, contract*
  *capability*, not incidental. `internal/sampling/gbnf_test.go`
  `TestLazyGBNFECMAScriptTriggerFeatures` explicitly exercises lookbehind
  `(?<=prefix)`, lookahead `(?=[0-9])`, and backreference `\2`;
  `TestLazyGBNFTriggerResourceLimits` asserts the bounded-backtracking guard
  errors on `(?:^){40000}`. These are the trigger semantics of llama.cpp's
  `std::regex` ECMAScript grammar, which overgo mirrors for parity. overgo's
  *own* generated pattern (`(<tool_call>)` in `chat_tool_grammar.go`) is
  RE2-expressible, but the trigger engine is a compat surface that accepts
  arbitrary caller/model-supplied patterns.
- **Decision — keep regexp2** (option b). Swapping to RE2 would delete a tested
  capability and break parity with the reference `std::regex` engine (a
  capability-retention violation, not a subtraction). regexp2 is **pure Go (no
  cgo, no transitive deps)**, so it does not violate the runtime-surface
  doctrine; it is a single leaf module. The ReDoS surface it would otherwise
  raise is already bounded structurally: `MaxBacktrackingStackSize(32768)` +
  `MatchTimeout(50ms)` + `maxGBNFTriggerPatternBytes(4096)` +
  `maxGBNFTriggerPatterns(128)`.
- Effort: zero code change — documented exception with rationale recorded in
  `trigger_regex.go` and here.

### 5. `gonja` (Jinja2) → minimal Jinja subset interpreter — HIGH (highest leverage)
- Chat-template rendering uses gonja's full environment (filters, tests, control
  structures, methods, whitespace control). stdlib `text/template` has different
  syntax and cannot run Jinja templates as-is; real chat templates
  (Qwen/Gemma/Llama) use `{%- for/if -%}`, filters (`tojson`, `trim`,
  `default`), and method calls.
- Path: **write a bounded Jinja2-subset interpreter** in stdlib only, scoped to
  exactly the constructs the shipped chat templates use — measured, not guessed:
  1. Inventory every `chat_template.jinja` across the served models; extract the
     set of tags/filters/tests/methods actually referenced.
  2. Implement a lexer + parser + evaluator covering only that set (loops,
     conditionals, whitespace-control `-`, the observed filters, attribute/method
     access, string ops). This is a real interpreter but bounded by the inventory.
  3. **Parity-gate hard:** render every model's template through both gonja and
     the new engine on real message fixtures; require byte-identical output
     before removing gonja. Any template using an unimplemented construct fails
     loudly (no silent divergence).
- Removing gonja clears logrus + json-iterator + humanize + modern-go/* +
  x/exp — the bulk of the tree — so it is the **highest-leverage** item despite
  being the hardest.
- Effort: several slices (inventory → interpreter → per-model parity fixtures).

## Staged stack (lowest-risk / highest-certainty first)

1. **x/sys → syscall/LazyDLL** — cleanest, doctrine-aligned, zero capability change.
2. **yaml → JSON** — kills kr/pretty; format migration, well-understood.
3. **x/text → generated NFD table** — self-contained, parity-fixture gated.
4. **gonja → Jinja subset** — biggest lift but clears 7 of 8 indirect deps;
   inventory-bounded + byte-parity gated.
5. **regexp2 → RESOLVED: retained** — the audit found lookaround/backreference
   are tested contract capabilities required for `std::regex` parity; kept as the
   single justified third-party (pure Go, leaf, no transitive deps).

Each step is independently landable and leaves the build green; after (1)–(4)
the only remaining external module is regexp2 (leaf, no transitive deps), which
(5) resolved as the one justified entry.

## Outcome — DONE
- `go.mod` `require` block: **one** third-party module, regexp2, the documented
  justified exception (rationale above + in `trigger_regex.go`). All other
  third-party (yaml.v3, x/sys, x/text, gonja) and every indirect dep removed.
- `go mod tidy` clean; SBOM (`cmd/sbom`) reflects the single-exception surface.
- Every replaced surface parity-gated against the dep it replaced: NFD strings
  (generated-table fixtures), parsed config structs (JSON round-trip), rendered
  chat templates (135-case byte-parity vs gonja), file locks (syscall/LazyDLL).
