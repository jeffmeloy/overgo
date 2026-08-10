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
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/guard"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

const (
	gateRecipeSeed   = "overgo-gate/v1"
	gateWorkloadSeed = "overgo-gate-workload/v1"
)

type gateContext struct {
	repo        string
	paths       []string
	messageFile string
	storePath   string
	steps       []runrecord.GateStep
	honesty     []string
	start       time.Time
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "gate: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file (required; never -m: the shell eats backticks)")
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships (required)")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (relative to repo root)")
	flag.Parse()
	if *messageFile == "" || *pathsCSV == "" {
		return fmt.Errorf("usage: gate -message-file <path> -paths <csv> [-store <dir>]")
	}
	repo, err := os.Getwd()
	if err != nil {
		return err
	}
	g := &gateContext{
		repo: repo, messageFile: *messageFile, storePath: *storePath, start: time.Now(),
	}
	for _, p := range strings.Split(*pathsCSV, ",") {
		if p = strings.TrimSpace(p); p != "" {
			g.paths = append(g.paths, filepath.ToSlash(p))
		}
	}
	if err := g.expandDirectoryPaths(); err != nil {
		return err
	}

	outcome := runrecord.OutcomeSucceeded
	failure := ""
	if err := g.pipeline(); err != nil {
		outcome = runrecord.OutcomeFailed
		failure = err.Error()
	}
	recordErr := g.record(outcome, failure)
	g.printSummary(outcome, failure)
	if recordErr != nil {
		fmt.Fprintf(os.Stderr, "gate: store record failed (result stands, record owed): %v\n", recordErr)
	}
	if outcome != runrecord.OutcomeSucceeded {
		return fmt.Errorf("%s", failure)
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
	treeKey, cache := g.loadRetryCache()
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
			g.saveRetryCache(treeKey, cache)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
	}
	return nil
}

type retryCache struct {
	TreeKey string            `json:"tree_key"`
	Steps   map[string]string `json:"steps"`
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

func (g *gateContext) loadRetryCache() (string, retryCache) {
	empty := retryCache{Steps: map[string]string{}}
	key, err := g.treeStateKey()
	if err != nil {
		return "", empty
	}
	empty.TreeKey = key
	raw, err := os.ReadFile(filepath.Join(g.repo, "bin", "gate_cache.json"))
	if err != nil {
		return key, empty
	}
	var cache retryCache
	if json.Unmarshal(raw, &cache) != nil || cache.TreeKey != key || cache.Steps == nil {
		return key, empty
	}
	return key, cache
}

func (g *gateContext) saveRetryCache(key string, cache retryCache) {
	if key == "" {
		return
	}
	cache.TreeKey = key
	raw, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return
	}
	dir := filepath.Join(g.repo, "bin")
	if os.MkdirAll(dir, 0o755) != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(dir, "gate_cache.json"), append(raw, '\n'), 0o644)
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
		if strings.HasSuffix(p, ".go") {
			out = append(out, p)
		}
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
	if !g.pathsTouchGo() {
		g.honesty = append(g.honesty, "build skipped: no Go source or module files in -paths")
		return true, nil
	}
	_, err := command(g.repo, "go", "build", "./...")
	return false, err
}

// stepTest derives scope from the import graph: the packages owning changed
// files plus every package whose transitive deps include one. A hand-listed
// impact table is a process magic; the graph is the derivation.
func (g *gateContext) stepTest() (bool, error) {
	changed := map[string]bool{}
	for _, f := range g.changedGoFiles() {
		dir := filepath.ToSlash(filepath.Dir(f))
		out, err := command(g.repo, "go", "list", "./"+dir)
		if err != nil {
			continue // deleted package: its importers still test below
		}
		changed[strings.TrimSpace(out)] = true
	}
	if len(changed) == 0 {
		g.honesty = append(g.honesty, "tests skipped: no .go in -paths")
		return true, nil
	}
	out, err := command(g.repo, "go", "list", "-f", "{{.ImportPath}} {{join .Deps \",\"}}", "./...")
	if err != nil {
		return false, err
	}
	var impacted []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		importPath, deps, _ := strings.Cut(line, " ")
		if changed[importPath] {
			impacted = append(impacted, importPath)
			continue
		}
		for _, dep := range strings.Split(deps, ",") {
			if changed[dep] {
				impacted = append(impacted, importPath)
				break
			}
		}
	}
	if len(impacted) == 0 {
		g.honesty = append(g.honesty, "tests skipped: changed packages have no importers and no tests resolved")
		return true, nil
	}
	g.honesty = append(g.honesty, fmt.Sprintf("test scope: %d packages (derived from import graph)", len(impacted)))
	_, err = command(g.repo, "go", append([]string{"test", "-count=1"}, impacted...)...)
	return false, err
}

func (g *gateContext) stepManifest() (bool, error) {
	if _, err := os.Stat(filepath.Join(g.repo, "kernels", "manifest.json")); err != nil {
		return true, nil
	}
	if !g.pathsTouchAny("kernels/", "internal/cuda/kernel/", "cmd/kernel-manifest/", "cmd/build-kernels/") {
		g.honesty = append(g.honesty, "manifest skipped: no kernel-owning paths in -paths")
		return true, nil
	}
	_, err := command(g.repo, "go", "run", "./cmd/kernel-manifest")
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
	run := g.pathsTouchAny("compatibility.yaml", "cmd/compatibility/", "internal/model/")
	if !run {
		raw, err := os.ReadFile(filepath.Join(g.repo, "compatibility.yaml"))
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
		g.honesty = append(g.honesty, "claims skipped: no changed path appears in compatibility.yaml")
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
	if !g.pathsTouchAny("kernels/", "internal/cuda/") {
		g.honesty = append(g.honesty, "device lane skipped: no kernel or CUDA-cone paths in -paths")
		return true, nil
	}
	_, err := command(g.repo, "go", "run", "./cmd/device-lane")
	return false, err
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
	envDoc, err := json.Marshal(map[string]string{
		"os": runtime.GOOS, "arch": runtime.GOARCH, "go": runtime.Version(), "host": hostname(),
	})
	if err != nil {
		return err
	}
	environmentID, err := artifact.IdentifyBytes(artifact.KindEvidence, envDoc)
	if err != nil {
		return err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, environmentID, codeCommit, outcome, failure,
		uint64(time.Since(g.start).Nanoseconds()), g.steps,
	)
	if err != nil {
		return err
	}
	batch, err := record.Batch("gate/" + codeCommit)
	if err != nil {
		return err
	}
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: recipeID}, artifact.Descriptor{ID: environmentID})
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
		return err
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	return g.writeStatus(record, codeCommit, outcome, failure)
}

// writeStatus mirrors the store record for cheap shell consumption; the store
// is authoritative and this file is advisory by construction.
func (g *gateContext) writeStatus(record runrecord.GateRecord, codeCommit string, outcome runrecord.Outcome, failure string) error {
	status := map[string]any{
		"result_id": record.Result.ID.String(), "code_commit": codeCommit,
		"outcome": outcome, "failure": failure, "steps": record.Result.Steps,
		"honesty": g.honesty, "written": time.Now().UTC().Format(time.RFC3339),
	}
	raw, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Join(g.repo, "bin")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "gate_status.json"), append(raw, '\n'), 0o644)
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

func hostname() string {
	name, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	return name
}

func tail(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return "..." + s[len(s)-limit:]
}
