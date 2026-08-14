// gate: the commit gate. One command owns scope refusal, hygiene, derived
// test scope, claim/manifest/SBOM verification, the scoped commit, and the
// store record. Never raw `git commit` during a campaign — the guard enforces
// that; this binary is the sanctioned path and marks its own commit
// subprocess with guard.GateEnv=1 (the guard owns that env-var name).
//
// Incident lineage honored here: -message-file only (shell-parsed prose loses
// backticked text to command substitution); a staged path outside -paths
// refuses rather than sweeps (a review commit once shipped another slice's
// staged deletions); the status mirror is advisory and the store record is
// authoritative (a killed gate once left a stale "running" status file);
// green output ends with the honesty line naming what did NOT run.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/guard"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testscope"
)

const (
	gateRecipeSeed    = "overgo-gate/v1"
	gateWorkloadSeed  = "overgo-gate-workload/v1"
	gateDebtFile      = "bin/gate_debt.json"
	gateHeartbeatFile = "bin/gate_lifecycle.json"
	gateRetryFile     = "bin/gate_cache.json"
)

type gateContext struct {
	repo        string
	paths       []string
	messageFile string
	storePath   string
	steps       []runrecord.GateStep
	honesty     []string
	start       time.Time
	environment runrecord.Environment
	preparation runrecord.GateLifecycle
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "gate: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file (required; never -m: the shell eats backticks)")
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships (required unless -merge)")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (relative to repo root)")
	merge := flag.Bool("merge", false, "finalize an in-progress merge: derive the shipped paths from the staged merge set and let the commit record both parents (stage it first with `git merge --no-ff --no-commit <branch>`)")
	planRef := flag.String("plan", "", "item/step this commit serves; MUST equal the plan's current open step (see `go run ./cmd/plan -next`). Required unless -merge. Off-plan commits are refused.")
	reconcile := flag.Bool("reconcile", false, "finalize the deterministic RepoDB batch in bin/gate_debt.json")
	watchdog := flag.Bool("watchdog", false, "print typed JSON liveness from bin/gate_lifecycle.json")
	staleAfter := flag.Duration("stale-after", 30*time.Second, "heartbeat age classified stale by -watchdog")
	flag.Parse()
	repo, err := os.Getwd()
	if err != nil {
		return err
	}
	cleanStore := filepath.Clean(*storePath)
	if cleanStore == "." || filepath.IsAbs(cleanStore) || cleanStore == ".." || strings.HasPrefix(cleanStore, ".."+string(filepath.Separator)) {
		return errors.New("gate: store path must stay below the repository root")
	}
	if *reconcile {
		preparation, err := reconcileGateDebt(repo, cleanStore)
		if err == nil {
			fmt.Printf("gate: reconciled RepoDB record debt for %s\n", preparation)
		}
		return err
	}
	if *watchdog {
		return printGateWatchdog(repo, *staleAfter)
	}
	if *messageFile == "" || (*pathsCSV == "" && !*merge) {
		return fmt.Errorf("usage: gate -message-file <path> (-paths <csv> | -merge) -plan <item>/<step> [-store <dir>]")
	}
	// Every commit -- including a merge finalize -- is bound to the plan's current
	// open step. Merges are no longer exempt: a sync/merge is a first-class plan
	// task (inject it with `plan -add`, then finalize with -plan <item>/do).
	if err := checkPlanBinding(repo, *planRef); err != nil {
		return err
	}
	g := &gateContext{
		repo: repo, messageFile: *messageFile, storePath: cleanStore, start: time.Now(),
	}
	if *merge {
		// Merge mode: the staged merge IS the plan. Deriving -paths from the
		// staged set makes the scope step trivially pass, and the existing
		// commit step (git commit -F, no pathspec) finalizes the two-parent
		// commit that git merge --no-commit left pending. Merges otherwise
		// bypass the gate entirely; this runs full hygiene + impacted tests on
		// the merged tree before the commit lands.
		if _, err := command(repo, "git", "rev-parse", "--verify", "-q", "MERGE_HEAD"); err != nil {
			return fmt.Errorf("-merge needs an in-progress merge (no MERGE_HEAD): run `git merge --no-ff --no-commit <branch>` first")
		}
		staged, err := gitLines(repo, "diff", "--cached", "--name-only")
		if err != nil {
			return err
		}
		if len(staged) == 0 {
			return fmt.Errorf("-merge: the staged merge set is empty")
		}
		for _, p := range staged {
			g.paths = append(g.paths, filepath.ToSlash(p))
		}
	} else {
		for _, p := range strings.Split(*pathsCSV, ",") {
			if p = strings.TrimSpace(p); p != "" {
				g.paths = append(g.paths, filepath.ToSlash(p))
			}
		}
	}
	if err := g.expandDirectoryPaths(); err != nil {
		return err
	}
	g.environment, err = discoverEnvironment(repo)
	if err != nil {
		return err
	}
	if err := g.prepare(); err != nil {
		return err
	}
	stopHeartbeat, err := g.startHeartbeat()
	if err != nil {
		return err
	}
	defer stopHeartbeat()

	outcome := runrecord.OutcomeSucceeded
	failure := ""
	if err := g.pipeline(); err != nil {
		outcome = runrecord.OutcomeFailed
		failure = err.Error()
	}
	recordErr := g.record(outcome, failure)
	stopHeartbeat()
	g.printSummary(outcome, failure)
	if recordErr != nil {
		_ = g.writeHeartbeat(runrecord.HeartbeatRecordDebt)
		fmt.Fprintf(os.Stderr, "gate: store record failed (result stands, record owed): %v\n", recordErr)
	} else {
		_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
	}
	if outcome != runrecord.OutcomeSucceeded {
		return fmt.Errorf("%s", failure)
	}
	if recordErr != nil {
		return fmt.Errorf("commit landed but RepoDB record debt remains: %w", recordErr)
	}
	return nil
}

// checkPlanBinding refuses any commit whose -plan is not the plan's current open
// step. This is the enforcement that makes off-plan work impossible to commit:
// the shared internal/plan.Current is the same "current step" cmd/plan dispatches
// and verifies, so the gate and the dispatcher can never disagree.
func checkPlanBinding(repo, ref string) error {
	if ref == "" {
		return fmt.Errorf("gate: -plan <item>/<step> is required (the plan's current open step; run `go run ./cmd/plan -next`)")
	}
	item, stepID, ok := strings.Cut(ref, "/")
	if !ok || item == "" || stepID == "" {
		return fmt.Errorf("gate: -plan must be <item>/<step>, got %q", ref)
	}
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return err
	}
	if err := plan.ValidateOpenWork(document); err != nil {
		return err
	}
	it, st, open := plan.Current(document)
	if !open {
		return fmt.Errorf("gate: -plan %s given but the plan is COMPLETE (no open step) -- nothing to commit against", ref)
	}
	if item != it.ID || stepID != st.ID {
		return fmt.Errorf("gate: -plan %s does NOT match the plan's current open step %s/%s -- commit only the dispatched step (off-plan commit REFUSED). If the plan is wrong, fix the plan first; do not commit around it", ref, it.ID, st.ID)
	}
	return nil
}

func (g *gateContext) pipeline() error {
	type step struct {
		name  string
		phase runrecord.Phase
		fn    func() (skipped bool, err error)
	}
	steps := []step{
		{"scope", runrecord.PhaseValidate, g.stepScope},
		{"fmt", runrecord.PhaseValidate, g.stepFmt},
		{"vet", runrecord.PhaseVet, g.stepVet},
		{"build", runrecord.PhaseBuild, g.stepBuild},
		{"test", runrecord.PhaseTest, g.stepTest},
		{"manifest", runrecord.PhaseValidate, g.stepManifest},
		{"sbom", runrecord.PhaseValidate, g.stepSBOM},
		{"claims", runrecord.PhaseValidate, g.stepClaims},
		{"magics", runrecord.PhaseValidate, g.stepMagics},
		{"device", runrecord.PhaseTest, g.stepDevice},
		{"commit", runrecord.PhasePackage, g.stepCommit},
	}
	cache := g.loadRetryCache()
	// Verification steps whose result depends only on tree state may reuse a
	// prior identical-tree success (the retry-loop tax: a failed commit step
	// re-paid full hygiene on every attempt). scope/fmt/magics are cheap and
	// always run; commit is never cached.
	cacheable := map[string]bool{"vet": true, "build": true, "test": true, "manifest": true, "sbom": true, "claims": true}
	for _, s := range steps {
		began := time.Now()
		var skipped bool
		var err error
		if cacheable[s.name] && cache.Steps[s.name] == string(runrecord.StepSucceeded) {
			skipped = true
			g.honesty = append(g.honesty, s.name+" reused: identical tree already passed this step")
		} else {
			skipped, err = s.fn()
		}
		record := runrecord.GateStep{
			Name: s.name, Phase: s.phase, Outcome: runrecord.StepSucceeded,
			DurationNS: uint64(time.Since(began).Nanoseconds()),
		}
		switch {
		case err != nil:
			record.Outcome = runrecord.StepFailed
		case skipped:
			record.Outcome = runrecord.StepSkipped
		}
		g.steps = append(g.steps, record)
		fmt.Printf("[gate] %-8s %-9s %6.2fs\n", s.name, record.Outcome, time.Since(began).Seconds())
		if err == nil && cacheable[s.name] && !skipped {
			cache.Steps[s.name] = string(runrecord.StepSucceeded)
			g.saveRetryCache(cache)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
	}
	return nil
}

// treeStateKey hashes HEAD plus every pending difference (staged, unstaged,
// and the content of untracked planned paths): identical key means the
// verification inputs are byte-identical.
func (g *gateContext) treeStateKey() (string, error) {
	head, err := command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	staged, err := command(g.repo, "git", "diff", "--cached")
	if err != nil {
		return "", err
	}
	unstaged, err := command(g.repo, "git", "diff")
	if err != nil {
		return "", err
	}
	hasher := sha256.New()
	hasher.Write([]byte(head))
	hasher.Write([]byte(staged))
	hasher.Write([]byte(unstaged))
	for _, p := range g.paths {
		raw, err := os.ReadFile(filepath.Join(g.repo, filepath.FromSlash(p)))
		if err != nil {
			continue // deletions contribute through the diffs
		}
		hasher.Write([]byte(p))
		hasher.Write(raw)
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

type retryCache struct {
	TreeKey     string            `json:"tree_key"`
	Environment string            `json:"environment"`
	Steps       map[string]string `json:"steps"`
}

func (g *gateContext) loadRetryCache() retryCache {
	key, _ := g.treeStateKey()
	empty := retryCache{TreeKey: key, Environment: g.environment.ID.String(), Steps: map[string]string{}}
	var cache retryCache
	if key == "" || readJSON(g.repo, gateRetryFile, &cache) != nil || !retryReusable(cache, key, empty.Environment) {
		return empty
	}
	return cache
}

func (g *gateContext) saveRetryCache(cache retryCache) {
	if cache.TreeKey != "" {
		_ = writeJSON(g.repo, gateRetryFile, cache, 0o644)
	}
}

func retryReusable(cache retryCache, treeKey, environment string) bool {
	return cache.TreeKey == treeKey && cache.Environment == environment && cache.Steps != nil
}

func discoverEnvironment(repo string) (runrecord.Environment, error) {
	out, err := command(repo, "go", "env", "CGO_ENABLED", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN")
	if err != nil {
		return runrecord.Environment{}, err
	}
	values := strings.Split(strings.ReplaceAll(out, "\r\n", "\n"), "\n")
	for len(values) < 4 {
		values = append(values, "")
	}
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH, Device: "host", Backend: "go",
		Driver: "cgo=" + strings.TrimSpace(values[0]),
		Runtime: fmt.Sprintf("%s;goflags=%s;goexperiment=%s;gotoolchain=%s", runtime.Version(),
			strings.TrimSpace(values[1]), strings.TrimSpace(values[2]), strings.TrimSpace(values[3])),
	})
}

// expandDirectoryPaths rewrites a -paths entry naming a directory into that
// directory's files (tracked + untracked, gitignore honored). Incident: a
// directory entry never matched porcelain's trailing-slash "?? dir/" form, so
// the add silently no-opped and a commit shipped a message claiming files it
// did not carry. Empty expansion refuses rather than dropping the entry.
func (g *gateContext) expandDirectoryPaths() error {
	var expanded []string
	seen := map[string]bool{}
	for _, p := range g.paths {
		info, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p)))
		if err != nil || !info.IsDir() {
			if !seen[p] {
				seen[p] = true
				expanded = append(expanded, p)
			}
			continue
		}
		files, err := gitLines(g.repo, "ls-files", "-co", "--exclude-standard", "--", p)
		if err != nil {
			return err
		}
		if len(files) == 0 {
			return fmt.Errorf("-paths entry %s is a directory with no eligible files", p)
		}
		for _, f := range files {
			f = filepath.ToSlash(f)
			if !seen[f] {
				seen[f] = true
				expanded = append(expanded, f)
			}
		}
	}
	g.paths = expanded
	return nil
}

// stepScope refuses staged paths outside the plan (the commit would ship
// them) and reports unstaged co-implementer dirt without blocking on it.
func (g *gateContext) stepScope() (bool, error) {
	staged, err := gitLines(g.repo, "diff", "--cached", "--name-only")
	if err != nil {
		return false, err
	}
	var rogue []string
	for _, p := range staged {
		if !slices.Contains(g.paths, filepath.ToSlash(p)) {
			rogue = append(rogue, p)
		}
	}
	if len(rogue) > 0 {
		return false, fmt.Errorf("staged outside -paths (would ship): %s", strings.Join(rogue, ", "))
	}
	dirty, err := gitLines(g.repo, "status", "--porcelain")
	if err != nil {
		return false, err
	}
	for _, line := range dirty {
		if len(line) < 4 {
			continue
		}
		p := filepath.ToSlash(strings.TrimSpace(line[3:]))
		if !slices.Contains(g.paths, p) {
			g.honesty = append(g.honesty, "unplanned dirty (not shipped): "+p)
		}
	}
	for _, p := range g.paths {
		if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p))); err != nil {
			if deleted, _ := gitLines(g.repo, "status", "--porcelain", "--", p); len(deleted) == 0 {
				return false, fmt.Errorf("planned path %s neither exists nor is a tracked deletion", p)
			}
		}
	}
	return false, nil
}

func (g *gateContext) changedGoFiles() []string {
	var out []string
	for _, p := range g.paths {
		if !strings.HasSuffix(p, ".go") {
			continue
		}
		// Skip planned .go paths no longer on disk (a rename/delete staged in
		// this commit): gofmt cannot stat a removed file, and the scope step
		// already validated the deletion is tracked.
		if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p))); err != nil {
			continue
		}
		out = append(out, p)
	}
	return out
}

func (g *gateContext) stepFmt() (bool, error) {
	files := g.changedGoFiles()
	if len(files) == 0 {
		return true, nil
	}
	args := append([]string{"-l"}, files...)
	out, err := command(g.repo, "gofmt", args...)
	if err != nil {
		return false, err
	}
	if s := strings.TrimSpace(out); s != "" {
		return false, fmt.Errorf("unformatted: %s", s)
	}
	return false, nil
}

func (g *gateContext) stepVet() (bool, error) {
	if len(g.changedGoFiles()) == 0 {
		return true, nil
	}
	_, err := command(g.repo, "go", "vet", "./...")
	return false, err
}

func (g *gateContext) pathsTouchGo() bool {
	for _, p := range g.paths {
		if strings.HasSuffix(p, ".go") || p == "go.mod" || p == "go.sum" {
			return true
		}
	}
	return false
}

func (g *gateContext) pathsTouchAny(prefixes ...string) bool {
	for _, p := range g.paths {
		for _, prefix := range prefixes {
			if p == prefix || strings.HasPrefix(p, prefix) {
				return true
			}
		}
	}
	return false
}

func (g *gateContext) stepBuild() (bool, error) {
	if !g.pathsTouchGo() && !g.pathsTouchAny("cmd/", "internal/") {
		g.honesty = append(g.honesty, "build skipped: no Go-owned source or asset paths in -paths")
		return true, nil
	}
	_, err := command(g.repo, "go", "build", "./...")
	return false, err
}

// stepTest derives scope from the import graph: the packages owning changed
// files plus every package whose transitive deps include one. A hand-listed
// impact table is a process magic; the graph is the derivation.
func (g *gateContext) stepTest() (bool, error) {
	changed, err := g.directChangedPackages()
	if err != nil {
		return false, err
	}
	if len(changed) == 0 {
		g.honesty = append(g.honesty, "tests skipped: no Go package owns a source or embedded asset in -paths")
		return true, nil
	}
	out, err := command(g.repo, "go", "list", "-f", "{{.ImportPath}} {{join .Deps \",\"}}", "./...")
	if err != nil {
		return false, err
	}
	var direct, dependent []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		importPath, deps, _ := strings.Cut(line, " ")
		if changed[importPath] {
			direct = append(direct, importPath)
			continue
		}
		for _, dep := range strings.Split(deps, ",") {
			if changed[dep] {
				dependent = append(dependent, importPath)
				break
			}
		}
	}
	if len(direct)+len(dependent) == 0 {
		g.honesty = append(g.honesty, "tests skipped: changed packages have no importers and no tests resolved")
		return true, nil
	}
	g.honesty = append(g.honesty, fmt.Sprintf("test scope: %d direct + %d dependent packages (derived from import graph)", len(direct), len(dependent)))
	if len(direct) > 0 {
		if _, err := runGoTests(g.repo, direct); err != nil {
			return false, err
		}
	}
	if len(dependent) == 0 {
		return false, nil
	}
	report, err := runGoTestsAdvisory(g.repo, dependent)
	if err != nil {
		return false, err
	}
	if len(report.Skipped)+len(report.Unavailable) > 0 {
		g.honesty = append(g.honesty, fmt.Sprintf(
			"dependent fixture evidence not credited: %d skipped, %d unavailable",
			len(report.Skipped), len(report.Unavailable),
		))
	}
	return false, nil
}

func (g *gateContext) directChangedPackages() (map[string]bool, error) {
	out, err := command(g.repo, "go", "list", "-json", "./...")
	if err != nil {
		return nil, fmt.Errorf("derive Go ownership: %w", err)
	}
	packages, err := testscope.DecodePackages(strings.NewReader(out))
	if err != nil {
		return nil, err
	}
	changed := map[string]bool{}
	for _, importPath := range testscope.DirectPackages(g.repo, g.paths, packages) {
		changed[importPath] = true
	}
	return changed, nil
}

func runGoTests(repo string, packages []string) (string, error) {
	out, err := command(repo, "go", append([]string{"test", "-short", "-json", "-count=1"}, packages...)...)
	if err != nil {
		return out, err
	}
	if err := testevidence.GoTestJSONShort(out); err != nil {
		return out, fmt.Errorf("impacted tests vacuous: %w", err)
	}
	return out, nil
}

func runGoTestsAdvisory(repo string, packages []string) (testevidence.GoTestReport, error) {
	out, err := command(repo, "go", append([]string{"test", "-json", "-count=1"}, packages...)...)
	if err != nil {
		return testevidence.GoTestReport{}, err
	}
	report, err := testevidence.GoTestJSONReport(out)
	if err != nil {
		return testevidence.GoTestReport{}, fmt.Errorf("dependent test evidence: %w", err)
	}
	return report, nil
}

func (g *gateContext) stepManifest() (bool, error) {
	if _, err := os.Stat(filepath.Join(g.repo, "kernels", "manifest.json")); err != nil {
		return true, nil
	}
	if !g.pathsTouchAny("kernels/", "internal/cuda/kernel/", "cmd/kernel-manifest/", "cmd/build-kernels/", "cmd/kernel-bindings/") {
		g.honesty = append(g.honesty, "manifest skipped: no kernel-owning paths in -paths")
		return true, nil
	}
	if _, err := command(g.repo, "go", "run", "./cmd/kernel-manifest"); err != nil {
		return false, err
	}
	// Bindings derive from the same manifest, but build-kernels updates the
	// manifest + PTX WITHOUT regenerating the executor bindings (that needs
	// `go generate ./internal/cuda/executor`). Verify freshness here so a stale
	// kernel_bindings_generated.go cannot ship on a kernel change.
	_, err := command(g.repo, "go", "test", "-run", "TestGeneratedBindingsMatchManifest", "-count=1", "./cmd/kernel-bindings")
	return false, err
}

func (g *gateContext) stepSBOM() (bool, error) {
	if !g.pathsTouchAny("go.mod", "go.sum", "LICENSES.md", "SBOM.cdx.json", "cmd/sbom/") {
		g.honesty = append(g.honesty, "sbom skipped: no dependency-owning paths in -paths")
		return true, nil
	}
	_, err := command(g.repo, "go", "run", "./cmd/sbom", "-check")
	return false, err
}

// stepClaims runs when the manifest itself, its checker, or any changed path
// mentioned in the manifest's raw bytes is in scope. Substring matching is
// deliberately safe-over-skip: a false positive runs the check, never the
// reverse.
func (g *gateContext) stepClaims() (bool, error) {
	run := g.pathsTouchAny("compatibility.json", "cmd/compatibility/", "internal/model/")
	if !run {
		raw, err := os.ReadFile(filepath.Join(g.repo, "compatibility.json"))
		if err != nil {
			return false, err
		}
		manifestText := string(raw)
		for _, p := range g.paths {
			if strings.Contains(manifestText, p) {
				run = true
				break
			}
		}
	}
	if !run {
		g.honesty = append(g.honesty, "claims skipped: no changed path appears in compatibility.json")
		return true, nil
	}
	_, err := command(g.repo, "go", "run", "./cmd/compatibility", "-check")
	return false, err
}

// stepMagics is Automation Doctrine Layer 6, scoped to this commit's files:
// numeric constants in changed production Go must have closure-ledger rows.
// Matching is by row NAME (owner-surface file IDs are version-pinned content
// hashes, so a name match is the honest path-stable heuristic). Advisory
// first — uncatalogued constants land in the honesty line, not a refusal —
// enforcement hardens once the 530-constant backlog is triaged
// (first-run-calibrates applied to enforcement itself).
func (g *gateContext) stepMagics() (bool, error) {
	candidates, err := closurescan.ScanFiles(g.repo, g.paths)
	if err != nil {
		return false, err
	}
	if len(candidates) == 0 {
		return true, nil
	}
	catalogued, err := ledgerNames(g.repo, g.storePath)
	if err != nil {
		g.honesty = append(g.honesty, "magic scan: ledger unreadable ("+err.Error()+"); constants unchecked")
		return false, nil
	}
	uncatalogued := 0
	for _, candidate := range candidates {
		if catalogued[candidate.Name] {
			continue
		}
		uncatalogued++
		g.honesty = append(g.honesty, fmt.Sprintf(
			"uncatalogued constant %s=%s (%s) — triage via closure-scan", candidate.Name, candidate.Value, candidate.File))
	}
	if uncatalogued == 0 {
		g.honesty = append(g.honesty, fmt.Sprintf("magic scan: %d constant(s) in scope, all catalogued", len(candidates)))
	}
	return false, nil
}

func ledgerNames(repo, storePath string) (map[string]bool, error) {
	store, err := repodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return nil, err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: 100_000})
	if err != nil {
		return nil, err
	}
	names := map[string]bool{}
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != closureledger.MediaType {
			continue
		}
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		document, err := closureledger.Parse(content.Data)
		if err != nil {
			continue
		}
		names[document.Name] = true
	}
	return names, nil
}

// stepDevice is the manifest-scoped device lane routing (Automation Doctrine
// Layer 3): the CUDA lane fires only when kernel-owning or CUDA-cone paths
// change; a failure INCLUDING device unavailability fails the commit —
// UNAVAILABLE never passes for a change that needs device evidence.
func (g *gateContext) stepDevice() (bool, error) {
	if !g.pathsTouchAny("kernels/", "internal/cuda/") && !g.pathsTouchDeviceSource() {
		g.honesty = append(g.honesty, "device lane skipped: no kernel or CUDA-cone paths in -paths")
		return true, nil
	}
	_, err := command(g.repo, "go", "run", "./cmd/device-lane")
	return false, err
}

// pathsTouchDeviceSource: device-lane code lives outside internal/cuda too (e.g.
// the device optimizer in internal/optimizer). Any _cuda_windows source/test in
// -paths fires the lane so its device evidence is not silently skipped.
func (g *gateContext) pathsTouchDeviceSource() bool {
	for _, p := range g.paths {
		if strings.Contains(p, "_cuda_windows") {
			return true
		}
	}
	return false
}

func (g *gateContext) stepCommit() (bool, error) {
	// Add only paths with UNSTAGED changes: git refuses an add pathspec for a
	// file that is gone with its deletion already fully staged (observed on
	// the .ps1 retirement commit, under both plain and -A forms). Fully
	// staged entries need no add; the scope step already proved staged
	// content stays inside -paths.
	dirty, err := gitLines(g.repo, append([]string{"status", "--porcelain", "--"}, g.paths...)...)
	if err != nil {
		return false, err
	}
	needAdd := map[string]bool{}
	for _, line := range dirty {
		if len(line) < 4 {
			continue
		}
		if line[1] != ' ' || strings.HasPrefix(line, "??") {
			needAdd[filepath.ToSlash(strings.TrimSpace(line[3:]))] = true
		}
	}
	var addList []string
	for _, p := range g.paths {
		if needAdd[p] {
			addList = append(addList, p)
		}
	}
	if len(addList) > 0 {
		addArgs := append([]string{"add", "-A", "--"}, addList...)
		if _, err := command(g.repo, "git", addArgs...); err != nil {
			return false, err
		}
	}
	cmd := exec.Command("git", "commit", "-F", g.messageFile)
	cmd.Dir = g.repo
	cmd.Env = append(os.Environ(), guard.GateEnv+"=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return false, fmt.Errorf("%v: %s", err, strings.TrimSpace(string(out)))
	}
	return false, nil
}

func (g *gateContext) prepare() error {
	if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(gateDebtFile))); err == nil {
		return errors.New("gate: unresolved bin/gate_debt.json; run `go run ./cmd/gate -reconcile` before another gate")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	treeKey, err := g.treeStateKey()
	if err != nil {
		return err
	}
	g.preparation, err = runrecord.NewGatePreparation(treeKey, g.environment.ID, g.start)
	if err != nil {
		return err
	}
	preparationContent, err := g.preparation.Content()
	if err != nil {
		return err
	}
	environmentContent, err := g.environment.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"gate/prepared/"+g.preparation.ID.String(),
		[]artifact.Content{environmentContent, preparationContent},
		g.preparation.Lineage(), nil,
	)
	if err != nil {
		return err
	}
	store, err := repodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle before Git commit: %w", err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return fmt.Errorf("prepare gate lifecycle before Git commit: %w", err)
	}
	return nil
}

func (g *gateContext) heartbeat(state runrecord.GateHeartbeatState) runrecord.GateHeartbeat {
	return runrecord.GateHeartbeat{
		Version: runrecord.GateHeartbeatVersion, State: state, Preparation: g.preparation.ID,
		TreeKey: g.preparation.TreeKey, Environment: g.environment.ID,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
}

func (g *gateContext) writeHeartbeat(state runrecord.GateHeartbeatState) error {
	heartbeat := g.heartbeat(state)
	if err := heartbeat.Validate(); err != nil {
		return err
	}
	return writeJSON(g.repo, gateHeartbeatFile, heartbeat, 0o644)
}

func (g *gateContext) startHeartbeat() (func(), error) {
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		return nil, err
	}
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_ = g.writeHeartbeat(runrecord.HeartbeatRunning)
			case <-stop:
				return
			}
		}
	}()
	return sync.OnceFunc(func() { close(stop); <-done }), nil
}

type gateDebtEnvelope struct {
	Version     uint16         `json:"version"`
	Preparation artifact.ID    `json:"preparation"`
	Batch       artifact.Batch `json:"batch"`
}

func (g *gateContext) oweRecord(batch artifact.Batch, cause error) error {
	debt := gateDebtEnvelope{Version: runrecord.GateLifecycleVersion, Preparation: g.preparation.ID, Batch: batch}
	if err := validateGateDebt(debt); err != nil {
		return fmt.Errorf("%w; invalid record debt: %v", cause, err)
	}
	if err := writeJSON(g.repo, gateDebtFile, debt, 0o600); err != nil {
		return fmt.Errorf("%w; persist record debt: %v", cause, err)
	}
	return cause
}

func reconcileGateDebt(repo, storePath string) (artifact.ID, error) {
	var debt gateDebtEnvelope
	if err := readJSON(repo, gateDebtFile, &debt); err != nil {
		return artifact.ID{}, fmt.Errorf("read record debt: %w", err)
	}
	if err := validateGateDebt(debt); err != nil {
		return artifact.ID{}, err
	}
	store, err := repodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	if _, ok, err := store.Content(context.Background(), debt.Preparation); err != nil {
		return artifact.ID{}, err
	} else if !ok {
		return artifact.ID{}, errors.New("gate: debt preparation is absent from RepoDB")
	}
	if _, err := store.Commit(context.Background(), debt.Batch); err != nil {
		return artifact.ID{}, err
	}
	if err := os.Remove(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err != nil {
		return artifact.ID{}, err
	}
	return debt.Preparation, nil
}

func validateGateDebt(debt gateDebtEnvelope) error {
	if debt.Version != runrecord.GateLifecycleVersion || debt.Preparation.Kind() != artifact.KindEvidence {
		return errors.New("gate: invalid record debt envelope")
	}
	if err := debt.Batch.Validate(); err != nil {
		return err
	}
	if debt.Batch.Key != "gate/final/"+debt.Preparation.String() {
		return errors.New("gate: record debt is not bound to its preparation")
	}
	matching := 0
	for _, content := range debt.Batch.Contents {
		if content.Descriptor.MediaType != runrecord.GateLifecycleMediaType || content.Descriptor.Schema != runrecord.GateLifecycleSchema {
			continue
		}
		lifecycle, err := runrecord.ParseGateLifecycle(content.Data)
		if err != nil {
			return err
		}
		if lifecycle.State == runrecord.GateFinalized && lifecycle.Preparation != nil && *lifecycle.Preparation == debt.Preparation {
			matching++
		}
	}
	if matching != 1 {
		return fmt.Errorf("gate: record debt has %d matching finalizations, want 1", matching)
	}
	return nil
}

type watchdogStatus struct {
	Version   uint16                       `json:"version"`
	State     runrecord.GateHeartbeatState `json:"state"`
	Heartbeat *runrecord.GateHeartbeat     `json:"heartbeat,omitempty"`
}

func printGateWatchdog(repo string, staleAfter time.Duration) error {
	var heartbeat runrecord.GateHeartbeat
	err := readJSON(repo, gateHeartbeatFile, &heartbeat)
	if errors.Is(err, os.ErrNotExist) {
		return printJSON(watchdogStatus{Version: runrecord.GateHeartbeatVersion, State: runrecord.HeartbeatAbsent})
	}
	if err != nil {
		return err
	}
	if err := heartbeat.Validate(); err != nil {
		return err
	}
	return printJSON(watchdogStatus{
		Version: heartbeat.Version, State: heartbeat.Watchdog(time.Now().UTC(), staleAfter), Heartbeat: &heartbeat,
	})
}

func (g *gateContext) record(outcome runrecord.Outcome, failure string) error {
	codeCommit, err := command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return err
	}
	codeCommit = strings.TrimSpace(codeCommit)

	recipeID, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte(gateRecipeSeed))
	if err != nil {
		return err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, g.environment.ID, codeCommit, outcome, failure,
		uint64(time.Since(g.start).Nanoseconds()), g.steps,
	)
	if err != nil {
		return err
	}
	batch, err := record.Batch("gate/final/" + g.preparation.ID.String())
	if err != nil {
		return err
	}
	environmentContent, err := g.environment.Content()
	if err != nil {
		return err
	}
	finalized, err := runrecord.NewGateFinalization(g.preparation, codeCommit, record.Result.ID, outcome)
	if err != nil {
		return err
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		return err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipeID})
	batch.Contents = append(batch.Contents, environmentContent, finalizedContent)
	batch.Lineage = append(batch.Lineage, finalized.Lineage()...)
	// A wall-time evaluation rides every successful run: evaluations are the
	// advisory layer's observation unit, so the gate's own history becomes
	// the calibration corpus (first run calibrates, second enforces).
	if outcome == runrecord.OutcomeSucceeded {
		workloadID, err := artifact.IdentifyBytes(artifact.KindDataset, []byte(gateWorkloadSeed))
		if err != nil {
			return err
		}
		evaluation, err := runrecord.NewEvaluation(recipeID, record.Run.ID, workloadID, []runrecord.Metric{{
			Name: "gate_wall_ns", Value: float64(time.Since(g.start).Nanoseconds()),
			Unit: "ns", Direction: runrecord.DirectionMinimize,
		}})
		if err != nil {
			return err
		}
		evaluationContent, err := evaluation.Content()
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: workloadID})
		batch.Contents = append(batch.Contents, evaluationContent)
		batch.Lineage = append(batch.Lineage, evaluation.Lineage()...)
	}

	store, err := repodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return g.oweRecord(batch, err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return g.oweRecord(batch, err)
	}
	_ = os.Remove(filepath.Join(g.repo, filepath.FromSlash(gateDebtFile)))
	if err := g.writeStatus(record, codeCommit, outcome, failure); err != nil {
		g.honesty = append(g.honesty, "advisory gate status mirror write failed: "+err.Error())
	}
	return nil
}

// writeStatus mirrors the store record for cheap shell consumption; the store
// is authoritative and this file is advisory by construction.
func (g *gateContext) writeStatus(record runrecord.GateRecord, codeCommit string, outcome runrecord.Outcome, failure string) error {
	status := map[string]any{
		"result_id": record.Result.ID.String(), "code_commit": codeCommit,
		"preparation_id": g.preparation.ID.String(), "environment_id": g.environment.ID.String(),
		"outcome": outcome, "failure": failure, "steps": record.Result.Steps,
		"honesty": g.honesty, "written": time.Now().UTC().Format(time.RFC3339),
	}
	return writeJSON(g.repo, "bin/gate_status.json", status, 0o644)
}

func (g *gateContext) printSummary(outcome runrecord.Outcome, failure string) {
	fmt.Printf("=== GATE %s in %.1fs ===\n", strings.ToUpper(string(outcome)), time.Since(g.start).Seconds())
	if failure != "" {
		fmt.Printf("failure: %s\n", failure)
	}
	run, skipped := 0, 0
	for _, s := range g.steps {
		switch s.Outcome {
		case runrecord.StepSkipped:
			skipped++
		default:
			run++
		}
	}
	fmt.Printf("honesty: steps run=%d skipped=%d\n", run, skipped)
	for _, line := range g.honesty {
		fmt.Printf("honesty: %s\n", line)
	}
}

func command(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%s %s: %v: %s", name, strings.Join(args, " "), err, tail(string(out), 2000))
	}
	return string(out), nil
}

func gitLines(dir string, args ...string) ([]string, error) {
	out, err := command(dir, "git", args...)
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, line := range strings.Split(out, "\n") {
		if line = strings.TrimRight(line, "\r"); strings.TrimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	return lines, nil
}

func readJSON(repo, name string, value any) error {
	return jsonfile.DecodeStrict(filepath.Join(repo, filepath.FromSlash(name)), value)
}

func writeJSON(repo, name string, value any, mode os.FileMode) error {
	return jsonfile.Write(filepath.Join(repo, filepath.FromSlash(name)), value, mode)
}

func printJSON(value any) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

func tail(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return "..." + s[len(s)-limit:]
}
