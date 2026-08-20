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
	"io"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/codeprofile"
	"overgo/internal/guard"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/protection"
	"overgo/internal/repoanalysis"
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
	repo         string
	paths        []string
	planRef      string
	messageFile  string
	storePath    string
	steps        []runrecord.GateStep
	honesty      []string
	start        time.Time
	environment  runrecord.Environment
	preparation  runrecord.GateLifecycle
	source       *repoanalysis.SourceSnapshot
	profile      *codeprofile.Profile
	profileDirty bool
	stepEvidence map[string]string
	phaseKeys    map[string]string
	cachePaths   []string
}

func main() {
	clioptions.MainNamed("gate", run)
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file (required; never -m: the shell eats backticks)")
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships (required unless -merge)")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (relative to repo root)")
	merge := flag.Bool("merge", false, "finalize an in-progress merge: derive the shipped paths from the staged merge set and let the commit record both parents (stage it first with `git merge --no-ff --no-commit <branch>`)")
	planRef := flag.String("plan", "", "item/step this commit serves; MUST equal the plan's current open step (see `go run ./cmd/plan -next`). Required unless -merge. Off-plan commits are refused.")
	reconcile := flag.Bool("reconcile", false, "finalize the deterministic RepoDB batch in bin/gate_debt.json")
	recordFailure := flag.Bool("record-failure", false, "recover an unbatchable post-commit record as a typed failed finalization")
	admitReview := flag.String("admit-review", "", "read-only: admit a RepoDB review-verdict ID against the current HEAD")
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
	if *recordFailure {
		preparation, err := recordUnbatchableFailure(repo, cleanStore)
		if err == nil {
			fmt.Printf("gate: recorded failed finalization for %s\n", preparation)
		}
		return err
	}
	if *admitReview != "" {
		head, err := command(repo, "git", "rev-parse", "HEAD")
		if err != nil {
			return err
		}
		if err := admitStoredReview(repo, cleanStore, *admitReview, strings.TrimSpace(head)); err != nil {
			return err
		}
		fmt.Printf("gate: review %s admitted at %s\n", *admitReview, strings.TrimSpace(head))
		return nil
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
		repo: repo, planRef: *planRef, messageFile: *messageFile, storePath: cleanStore, start: time.Now(),
		stepEvidence: map[string]string{}, phaseKeys: map[string]string{},
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
	if !slices.Contains(g.paths, plan.Path) {
		g.paths = append(g.paths, plan.Path)
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
	failureCode := ""
	var pipelineErr error
	if pipelineErr = g.pipeline(); pipelineErr != nil {
		outcome = runrecord.OutcomeFailed
		failureCode = terminalFailureCode(g.steps)
	}
	recordErr := g.record(outcome, failureCode)
	stopHeartbeat()
	failureDetail := ""
	if pipelineErr != nil {
		failureDetail = pipelineErr.Error()
	}
	g.printSummary(os.Stdout, outcome, failureDetail)
	if recordErr != nil {
		_ = g.writeHeartbeat(runrecord.HeartbeatRecordDebt)
		fmt.Fprintf(os.Stderr, "gate: store record failed (result stands, record owed): %v\n", recordErr)
	} else {
		_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
	}
	if pipelineErr != nil {
		return pipelineErr
	}
	if recordErr != nil {
		return fmt.Errorf("commit landed but RepoDB record debt remains: %w", recordErr)
	}
	return nil
}

func terminalFailureCode(steps []runrecord.GateStep) string {
	for index := len(steps) - 1; index >= 0; index-- {
		if steps[index].Outcome == runrecord.StepFailed {
			return steps[index].Name
		}
	}
	return "gate"
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

type gateStep struct {
	name  string
	phase runrecord.Phase
	fn    func() (skipped bool, err error)
}

func (g *gateContext) pipelineSteps() []gateStep {
	steps := []gateStep{
		{"protection", runrecord.PhaseValidate, g.stepProtection},
		{"scope", runrecord.PhaseValidate, g.stepScope},
		{"profile", runrecord.PhaseValidate, g.stepProfile},
	}
	return append(steps, []gateStep{
		{"fmt", runrecord.PhaseValidate, g.stepFmt},
		{"vet", runrecord.PhaseVet, g.stepVet},
		{"build", runrecord.PhaseBuild, g.stepBuild},
		{"test", runrecord.PhaseTest, g.stepTest},
		{"manifest", runrecord.PhaseValidate, g.stepManifest},
		{"sbom", runrecord.PhaseValidate, g.stepSBOM},
		{"claims", runrecord.PhaseValidate, g.stepClaims},
		{"docs", runrecord.PhaseValidate, g.stepDocumentation},
		{"magics", runrecord.PhaseValidate, g.stepMagics},
		{"device", runrecord.PhaseTest, g.stepDevice},
		{"acceptance", runrecord.PhaseTest, g.stepAcceptance},
		{"commit", runrecord.PhasePackage, g.stepCommit},
	}...)
}

func (g *gateContext) pipeline() error {
	steps := g.pipelineSteps()
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
		input := ""
		if cacheable[s.name] {
			input, err = g.phaseInputFingerprint(s.name)
		}
		cached := cache.Steps[s.name]
		if err == nil && cached.Input == input && cached.Outcome == string(runrecord.StepSucceeded) {
			skipped = true
			g.honesty = append(g.honesty, s.name+" reused: derived inputs already passed this step")
		} else if err == nil {
			skipped, err = s.fn()
		}
		duration := uint64(time.Since(began).Nanoseconds())
		if duration == 0 && !skipped {
			duration = 1
		}
		record := runrecord.GateStep{
			Name: s.name, Phase: s.phase, Outcome: runrecord.StepSucceeded,
			DurationNS: duration, Evidence: g.stepEvidence[s.name],
		}
		switch {
		case err != nil:
			record.Outcome = runrecord.StepFailed
		case skipped:
			record.Outcome = runrecord.StepSkipped
		}
		g.steps = append(g.steps, record)
		if err == nil && cacheable[s.name] && !skipped {
			cache.Steps[s.name] = phaseCache{Input: input, Outcome: string(runrecord.StepSucceeded)}
			g.saveRetryCache(cache)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
	}
	return nil
}

const (
	historicalRankingMarker = "<!-- overgo-document: historical-ranking -->"
	currentWorkMarker       = "<!-- overgo-current-work: docs/plan.json -->"
)

// stepDocumentation keeps prose assessments from masquerading as current
// work authority. Ranked current work belongs to the failable plan; prose may
// retain its historical ordering only with an explicit warning and redirect.
func (g *gateContext) stepDocumentation() (bool, error) {
	return false, documentationFreshness(g.repo)
}

func documentationFreshness(root string) error {
	return filepath.WalkDir(filepath.Join(root, "docs"), func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		if !strings.Contains(text, "## Execution Program") {
			return nil
		}
		header := text
		if len(header) > 1024 {
			header = header[:1024]
		}
		if !strings.Contains(header, historicalRankingMarker) || !strings.Contains(header, currentWorkMarker) {
			return fmt.Errorf("documentation ranking %s lacks a prominent historical marker and docs/plan.json redirect", filepath.ToSlash(path))
		}
		return nil
	})
}

func (g *gateContext) stepProtection() (bool, error) {
	configured, activated, err := protection.Verify(g.repo)
	if err != nil {
		return false, err
	}
	g.stepEvidence["protection"] = configured + ";activation=" + activated
	g.honesty = append(g.honesty, "protection: "+g.stepEvidence["protection"])
	return false, nil
}

func (g *gateContext) sourceSnapshot() (repoanalysis.SourceSnapshot, error) {
	if g.source != nil {
		return *g.source, nil
	}
	snapshot, err := repoanalysis.DiscoverGo(g.repo, "internal", "cmd")
	if err == nil {
		g.source = &snapshot
	}
	return snapshot, err
}

func (g *gateContext) stepProfile() (bool, error) {
	changed := g.plannedGoFiles()
	if len(changed) == 0 {
		return true, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return false, err
	}
	baseSource, err := sourceAtHEAD(g.repo, snapshot)
	if err != nil {
		return false, err
	}
	base, err := codeprofile.Build(baseSource)
	if err != nil {
		return false, err
	}
	signals := profileSignals(profile)
	g.honesty = append(g.honesty, fmt.Sprintf(
		"code profile: production=%d files/%d nodes test=%d/%d validator_subset=%d functions/%d nodes duplicate_excess=%d (production=%d validator=%d test=%d) clones=%d functions=%d exported=%d imports=%d",
		profile.Production.Files, profile.Production.Nodes, profile.Test.Files, profile.Test.Nodes,
		signals.validator.functions, signals.validator.nodes, profile.DuplicateExcessNodes,
		signals.production.duplicateExcess, signals.validator.duplicateExcess, signals.test.duplicateExcess,
		len(profile.Clones), len(profile.Functions), profile.ExportedDeclarations, profile.PackageImportEdges,
	))
	g.honesty = append(g.honesty, surfaceDeltaHonesty(base, profile))
	if g.automationPlan() {
		movement, err := codeprofile.MeasureProductionMovement(baseSource, snapshot)
		if err != nil {
			return false, err
		}
		summary, err := automationROIAdmission("commit", movement)
		g.honesty = append(g.honesty, summary)
		if err != nil {
			return false, err
		}
	}
	g.honesty = append(g.honesty, profileReviewFocus(profile, g.changedGoFiles()))
	if err := g.appendConsumerCensus(snapshot, baseSource, changed, &profile); err != nil {
		return false, err
	}
	g.profile = &profile
	return false, nil
}

func (g *gateContext) appendConsumerCensus(candidate, head repoanalysis.SourceSnapshot, changed []string, profile *codeprofile.Profile) error {
	selection, err := repoanalysis.HostBuildSelection(g.repo, "./cmd/...", "./internal/...")
	if err != nil {
		return err
	}
	if _, profile.Consumers, err = codeprofile.ProductionConsumerCensus(candidate, selection, nil); err != nil {
		return err
	}
	paths := pathSet(changed)
	declarations, current, err := codeprofile.ProductionConsumerCensus(candidate, selection, paths)
	if err != nil {
		return err
	}
	baseDeclarations, base, err := codeprofile.ProductionConsumerCensus(head, selection, paths)
	if err != nil {
		return err
	}
	g.honesty = append(g.honesty, consumerCensusHonesty("commit", selection.Context, declarations, base, current))
	if unconsumed := codeprofile.NewUnconsumedSurface(baseDeclarations, declarations); len(unconsumed) > 0 {
		return fmt.Errorf("new unconsumed production surface: %s", consumerCandidates(unconsumed))
	}

	mergeBase, err := command(g.repo, "git", "merge-base", "master", "HEAD")
	if err != nil {
		g.honesty = append(g.honesty, "consumer census plan-slice unavailable: "+err.Error())
		return nil
	}
	mergeBase = strings.TrimSpace(mergeBase)
	slicePaths, err := changedGoPathsAtRevision(g.repo, mergeBase, changed)
	if err != nil {
		return err
	}
	sliceBase, err := sourceAtRevision(g.repo, candidate, mergeBase, slicePaths)
	if err != nil {
		return err
	}
	if g.automationPlan() {
		movement, err := codeprofile.MeasureProductionMovement(sliceBase, candidate)
		if err != nil {
			return err
		}
		summary, err := automationROIAdmission("plan-slice@"+mergeBase[:12], movement)
		g.honesty = append(g.honesty, summary)
		if err != nil {
			return err
		}
	}
	paths = pathSet(slicePaths)
	declarations, current, err = codeprofile.ProductionConsumerCensus(candidate, selection, paths)
	if err != nil {
		return err
	}
	_, base, err = codeprofile.ProductionConsumerCensus(sliceBase, selection, paths)
	if err != nil {
		return err
	}
	g.honesty = append(g.honesty, consumerCensusHonesty("plan-slice@"+mergeBase[:12], selection.Context, declarations, base, current))
	return nil
}

func (g *gateContext) automationPlan() bool {
	item, _, _ := strings.Cut(g.planRef, "/")
	return strings.HasPrefix(item, "automation-")
}

func automationROIAdmission(scope string, movement codeprofile.ProductionMovement) (string, error) {
	summary := fmt.Sprintf(
		"automation ROI %s: production_ast=%d-%d net=%+d go_lines=%d-%d net=%+d; admission=net-negative",
		scope, movement.Added, movement.Deleted, movement.Added-movement.Deleted,
		movement.GoLinesAdded, movement.GoLinesDeleted, movement.GoLinesAdded-movement.GoLinesDeleted,
	)
	if movement.Added > 0 && movement.Added >= movement.Deleted ||
		movement.GoLinesAdded > 0 && movement.GoLinesAdded >= movement.GoLinesDeleted {
		return summary, fmt.Errorf("automation surface is not net-negative: %s", summary)
	}
	return summary, nil
}

func consumerCensusHonesty(scope, context string, declarations []codeprofile.ConsumerDeclaration, base, current codeprofile.ConsumerSummary) string {
	var candidates []string
	for _, declaration := range declarations {
		class := ""
		switch {
		case declaration.ProductionReferences == 0 && declaration.TestReferences > 0:
			class = "test-only"
		case declaration.ProductionReferences == 0 && declaration.TestReferences == 0 && declaration.Boundary == "":
			class = "zero"
		}
		if class != "" {
			candidates = append(candidates, consumerCandidate(declaration, class))
		}
	}
	sort.Strings(candidates)
	if len(candidates) > 5 {
		candidates = candidates[:5]
	}
	return fmt.Sprintf(
		"consumer census %s context=%s delta: production=%+d test_only=%+d boundary=%+d zero=%+d; current=%d/%d/%d/%d; candidates=%s; advisory_only=ambiguous dispatch is a boundary, tests are not production consumers",
		scope, context, current.Production-base.Production, current.TestOnly-base.TestOnly,
		current.Boundary-base.Boundary, current.Zero-base.Zero,
		current.Production, current.TestOnly, current.Boundary, current.Zero, strings.Join(candidates, ","),
	)
}

func consumerCandidates(declarations []codeprofile.ConsumerDeclaration) string {
	candidates := make([]string, len(declarations))
	for index, declaration := range declarations {
		class := "zero"
		if declaration.TestReferences > 0 {
			class = "test-only"
		}
		candidates[index] = consumerCandidate(declaration, class)
	}
	sort.Strings(candidates)
	return strings.Join(candidates, ",")
}

func consumerCandidate(declaration codeprofile.ConsumerDeclaration, class string) string {
	return declaration.File + ":" + declaration.Name + "=" + class
}

func changedGoPathsAtRevision(repo, revision string, pending []string) ([]string, error) {
	raw, err := command(repo, "git", "diff", "--no-renames", "--name-only", "-z", revision, "--", "cmd", "internal")
	if err != nil {
		return nil, err
	}
	paths := pathSet(pending)
	for _, name := range strings.Split(raw, "\x00") {
		if strings.HasSuffix(name, ".go") {
			paths[filepath.ToSlash(name)] = true
		}
	}
	return slices.Sorted(maps.Keys(paths)), nil
}

func sourceAtRevision(repo string, candidate repoanalysis.SourceSnapshot, revision string, paths []string) (repoanalysis.SourceSnapshot, error) {
	overlay := make(map[string][]byte, len(paths))
	for _, path := range paths {
		cmd := exec.Command("git", "show", revision+":"+path)
		cmd.Dir = repo
		data, err := cmd.Output()
		if err != nil {
			if _, missing := err.(*exec.ExitError); missing {
				overlay[path] = nil
				continue
			}
			return repoanalysis.SourceSnapshot{}, err
		}
		overlay[path] = data
	}
	return candidate.Overlay(overlay)
}

func pathSet(paths []string) map[string]bool {
	set := make(map[string]bool, len(paths))
	for _, path := range paths {
		set[filepath.ToSlash(path)] = true
	}
	return set
}

type profileSignal struct {
	functions, nodes, duplicateExcess int
}

type profileSignalSet struct {
	production, validator, test profileSignal
}

func profileSignals(profile codeprofile.Profile) profileSignalSet {
	var signals profileSignalSet
	for _, function := range profile.Functions {
		signal := signalForClass(&signals, function.AdvisoryClass)
		signal.functions++
		signal.nodes += function.Nodes
	}
	for _, clone := range profile.Clones {
		signal := signalForClass(&signals, clone.AdvisoryClass)
		signal.duplicateExcess += clone.Nodes * (len(clone.Functions) - 1)
	}
	return signals
}

func signalForClass(signals *profileSignalSet, class string) *profileSignal {
	switch class {
	case "validator":
		return &signals.validator
	case "test":
		return &signals.test
	default:
		return &signals.production
	}
}

func surfaceDeltaHonesty(base, candidate codeprofile.Profile) string {
	baseSignals, candidateSignals := profileSignals(base), profileSignals(candidate)
	productionFiles := candidate.Production.Files - base.Production.Files
	productionNodes := candidate.Production.Nodes - base.Production.Nodes
	duplicateExcess := candidate.DuplicateExcessNodes - base.DuplicateExcessNodes
	adverse := "none"
	if duplicateExcess < 0 && (productionFiles > 0 || productionNodes > 0) {
		adverse = "duplication fell while production grew; reduction does not offset surface growth"
	}
	return fmt.Sprintf(
		"code profile delta vs HEAD: production=%+d files/%+d nodes test=%+d/%+d validator_subset=%+d functions/%+d nodes duplicate_excess=%+d (production=%+d validator=%+d test=%+d) clones=%+d function_count=%+d exported=%+d imports=%+d; adverse_pattern=%s",
		productionFiles, productionNodes, candidate.Test.Files-base.Test.Files, candidate.Test.Nodes-base.Test.Nodes,
		candidateSignals.validator.functions-baseSignals.validator.functions, candidateSignals.validator.nodes-baseSignals.validator.nodes,
		duplicateExcess,
		candidateSignals.production.duplicateExcess-baseSignals.production.duplicateExcess,
		candidateSignals.validator.duplicateExcess-baseSignals.validator.duplicateExcess,
		candidateSignals.test.duplicateExcess-baseSignals.test.duplicateExcess,
		len(candidate.Clones)-len(base.Clones), len(candidate.Functions)-len(base.Functions),
		candidate.ExportedDeclarations-base.ExportedDeclarations, candidate.PackageImportEdges-base.PackageImportEdges, adverse,
	)
}

func profileReviewFocus(profile codeprofile.Profile, changed []string) string {
	paths := make(map[string]bool, len(changed))
	for _, path := range changed {
		paths[path] = true
	}
	return fmt.Sprintf(
		"code review candidates: production=%s; validator=%s; test=%s; exact_clone_production=%s; exact_clone_validator=%s; exact_clone_test=%s; advisory_only=inspect semantic ownership and numerical contracts, migrate callers and delete displaced paths, require parity evidence",
		largestChangedFunction(profile, paths, ""), largestChangedFunction(profile, paths, "validator"), largestChangedFunction(profile, paths, "test"),
		largestChangedClone(profile, paths, ""), largestChangedClone(profile, paths, "validator"), largestChangedClone(profile, paths, "test"),
	)
}

func largestChangedFunction(profile codeprofile.Profile, paths map[string]bool, class string) string {
	for _, function := range profile.Functions {
		if paths[function.File] && function.AdvisoryClass == class {
			return fmt.Sprintf("%s:%s nodes=%d branches=%d", function.File, function.Name, function.Nodes, function.Branches)
		}
	}
	return "none"
}

func largestChangedClone(profile codeprofile.Profile, paths map[string]bool, class string) string {
	for _, clone := range profile.Clones {
		if clone.AdvisoryClass == class && !cliMainClone(clone) && slices.ContainsFunc(clone.Functions, func(function string) bool {
			path, _, ok := strings.Cut(function, ":")
			return ok && paths[path]
		}) {
			return fmt.Sprintf("nodes=%d functions=%s", clone.Nodes, strings.Join(clone.Functions, ","))
		}
	}
	return "none"
}

func cliMainClone(clone codeprofile.Clone) bool {
	return len(clone.Functions) > 1 && !slices.ContainsFunc(clone.Functions, func(function string) bool {
		return !strings.HasSuffix(function, ":main")
	})
}

func sourceAtHEAD(repo string, candidate repoanalysis.SourceSnapshot) (repoanalysis.SourceSnapshot, error) {
	raw, err := command(repo, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return repoanalysis.SourceSnapshot{}, err
	}
	dirty, err := repoanalysis.ParseDirtyStatus([]byte(raw))
	if err != nil {
		return repoanalysis.SourceSnapshot{}, err
	}
	paths := map[string]bool{}
	for _, entry := range dirty {
		for _, path := range []string{entry.Path, entry.OriginalPath} {
			if !strings.HasSuffix(path, ".go") || !strings.HasPrefix(path, "internal/") && !strings.HasPrefix(path, "cmd/") {
				continue
			}
			paths[path] = true
		}
	}
	return sourceAtRevision(repo, candidate, "HEAD", slices.Sorted(maps.Keys(paths)))
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

type phaseCache struct {
	Input   string `json:"input"`
	Outcome string `json:"outcome"`
}

type retryCache struct {
	Environment string                `json:"environment"`
	Steps       map[string]phaseCache `json:"steps"`
}

func (g *gateContext) loadRetryCache() retryCache {
	empty := retryCache{Environment: g.environment.ID.String(), Steps: map[string]phaseCache{}}
	var cache retryCache
	if readJSON(g.repo, gateRetryFile, &cache) != nil || !retryReusable(cache, empty.Environment) {
		return empty
	}
	return cache
}

func (g *gateContext) saveRetryCache(cache retryCache) {
	if cache.Environment != "" {
		_ = writeJSON(g.repo, gateRetryFile, cache, 0o644)
	}
}

func retryReusable(cache retryCache, environment string) bool {
	return cache.Environment == environment && cache.Steps != nil
}

func (g *gateContext) phaseInputFingerprint(phase string) (string, error) {
	if key := g.phaseKeys[phase]; key != "" {
		return key, nil
	}
	paths := g.cachePaths
	var err error
	if paths == nil {
		paths, err = gitLines(g.repo, "ls-files", "-co", "--exclude-standard")
		if err != nil {
			return "", err
		}
		g.cachePaths = paths
	}
	var evidence map[string]bool
	if phase == "claims" {
		evidence, err = compatibilityEvidencePaths(g.repo)
		if err != nil {
			return "", err
		}
	}
	key, err := fingerprintPhaseInputs(g.repo, phase, paths, evidence)
	if err == nil {
		if g.phaseKeys == nil {
			g.phaseKeys = map[string]string{}
		}
		g.phaseKeys[phase] = key
	}
	return key, err
}

func fingerprintPhaseInputs(root, phase string, paths []string, claimEvidence map[string]bool) (string, error) {
	var selected []string
	for _, path := range paths {
		path = filepath.ToSlash(path)
		if phaseOwnsPath(phase, path, claimEvidence) {
			selected = append(selected, path)
		}
	}
	sort.Strings(selected)
	hasher := sha256.New()
	hasher.Write([]byte(phase + "\x00"))
	for _, path := range selected {
		hasher.Write([]byte(path + "\x00"))
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if errors.Is(err, os.ErrNotExist) {
			hasher.Write([]byte("<deleted>\x00"))
			continue
		}
		if err != nil {
			return "", err
		}
		hasher.Write(data)
		hasher.Write([]byte{0})
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func phaseOwnsPath(phase, path string, claimEvidence map[string]bool) bool {
	goSource := path == "go.mod" || path == "go.sum" || strings.HasSuffix(path, ".go")
	goInput := goSource ||
		(strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "cmd/")) && !strings.HasSuffix(path, ".md")
	switch phase {
	case "vet", "build":
		return goInput
	case "test":
		return goInput || path == "README.md" || strings.HasPrefix(path, "docs/")
	case "manifest":
		return goSource || strings.HasPrefix(path, "kernels/") || strings.HasPrefix(path, "cmd/kernel-") ||
			strings.HasPrefix(path, "internal/cuda/executor/")
	case "sbom":
		return goSource || path == "SBOM.cdx.json" || strings.HasPrefix(path, "cmd/sbom/")
	case "claims":
		return goSource || path == "compatibility.json" || path == "docs/COMPATIBILITY.md" ||
			strings.HasPrefix(path, "cmd/compatibility/") || claimEvidence[path]
	default:
		return false
	}
}

func compatibilityEvidencePaths(root string) (map[string]bool, error) {
	raw, err := os.ReadFile(filepath.Join(root, "compatibility.json"))
	if errors.Is(err, os.ErrNotExist) {
		return map[string]bool{}, nil
	}
	if err != nil {
		return nil, err
	}
	var document struct {
		Claims []struct {
			Evidence []struct {
				Path string `json:"path"`
			} `json:"evidence"`
		} `json:"claims"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, err
	}
	paths := map[string]bool{}
	for _, claim := range document.Claims {
		for _, evidence := range claim.Evidence {
			paths[filepath.ToSlash(evidence.Path)] = true
		}
	}
	return paths, nil
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
	status, err := command(g.repo, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return false, err
	}
	dirty, err := repoanalysis.ParseDirtyStatus([]byte(status))
	if err != nil {
		return false, err
	}
	dirtyPaths, unplanned := scopeDirty(g.paths, dirty)
	for _, path := range unplanned {
		g.honesty = append(g.honesty, "unplanned dirty (not shipped): "+path)
		g.profileDirty = g.profileDirty || strings.HasSuffix(path, ".go")
	}
	for _, p := range g.paths {
		if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p))); err != nil {
			if !dirtyPaths[p] {
				return false, fmt.Errorf("planned path %s neither exists nor is a tracked deletion", p)
			}
		}
	}
	return false, nil
}

func scopeDirty(planned []string, dirty []repoanalysis.DirtyPath) (map[string]bool, []string) {
	visible := map[string]bool{}
	var unplanned []string
	for _, entry := range dirty {
		for _, path := range []string{entry.Path, entry.OriginalPath} {
			if path == "" || visible[path] {
				continue
			}
			visible[path] = true
			if !slices.Contains(planned, path) {
				unplanned = append(unplanned, path)
			}
		}
	}
	return visible, unplanned
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

func (g *gateContext) plannedGoFiles() []string {
	var out []string
	for _, path := range g.paths {
		if strings.HasSuffix(path, ".go") && (strings.HasPrefix(path, "cmd/") || strings.HasPrefix(path, "internal/")) {
			out = append(out, path)
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
	if !g.pathsTouchAny("go.mod", "go.sum", "SBOM.cdx.json", "cmd/sbom/") {
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

// stepMagics: scoped constants against exact active closure evidence.
func (g *gateContext) stepMagics() (bool, error) {
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	candidates, err := closurescan.ScanSnapshot(snapshot, g.paths)
	if err != nil {
		return false, err
	}
	if len(candidates) == 0 {
		return true, nil
	}
	baseline, err := magicCandidatesAtHEAD(g.repo, snapshot, g.paths)
	if err != nil {
		return false, err
	}
	catalogued, err := activeMagicBindings(g.repo, g.storePath, snapshot, candidates)
	if err != nil {
		return false, err
	}
	diagnostic, err := admitMagicDelta(candidates, baseline, catalogued)
	if diagnostic != "" {
		g.honesty = append(g.honesty, diagnostic)
	}
	if err != nil {
		return false, err
	}
	return false, nil
}

func magicCandidatesAtHEAD(repo string, snapshot repoanalysis.SourceSnapshot, paths []string) ([]closurescan.Candidate, error) {
	overlay := map[string][]byte{}
	for _, path := range paths {
		if !strings.HasSuffix(path, ".go") {
			continue
		}
		cmd := exec.Command("git", "show", "HEAD:"+path)
		cmd.Dir = repo
		data, err := cmd.Output()
		if err != nil {
			if _, missing := err.(*exec.ExitError); missing {
				overlay[path] = nil
				continue
			}
			return nil, err
		}
		overlay[path] = data
	}
	baseline, err := snapshot.Overlay(overlay)
	if err != nil {
		return nil, err
	}
	return closurescan.ScanSnapshot(baseline, paths)
}

func admitMagicDelta(current, baseline []closurescan.Candidate, catalogued map[string]bool) (string, error) {
	previous := map[string]bool{}
	for _, candidate := range baseline {
		previous[candidate.DecisionKey()] = true
	}
	inherited := 0
	for _, candidate := range current {
		if catalogued[candidate.ExactKey()] {
			continue
		}
		if previous[candidate.DecisionKey()] {
			inherited++
			continue
		}
		return "", fmt.Errorf(
			"magic scan: new or changed uncatalogued constant %s=%s (%s)",
			candidate.Name, candidate.Value, candidate.File,
		)
	}
	if inherited > 0 {
		return fmt.Sprintf(
			"magic backlog: %d inherited uncatalogued constant(s) in touched files; run closure-scan for ranked detail",
			inherited,
		), nil
	}
	return fmt.Sprintf("magic scan: %d constant(s) in scope, all catalogued", len(current)), nil
}

func activeMagicBindings(
	repo, storePath string,
	snapshot repoanalysis.SourceSnapshot,
	candidates []closurescan.Candidate,
) (map[string]bool, error) {
	store, err := repodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return nil, err
	}
	defer store.Close()
	result, err := store.Query(context.Background(), repodb.Query{
		Kind: artifact.KindEvidence, MediaType: closureledger.MediaType,
		Schema: closureledger.Schema, MaxResults: repodb.MaxQueryResults,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, errors.New("magic scan: active-ledger query truncated")
	}
	seen := map[artifact.ID]bool{}
	var documents []closureledger.Document
	for _, alias := range result.Aliases {
		if !closureledger.IsActiveAlias(alias.Name) || seen[alias.Target] {
			continue
		}
		seen[alias.Target] = true
		content, found, err := store.Content(context.Background(), alias.Target)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, errors.New("magic scan: active document unavailable")
		}
		if err := content.Validate(); err != nil {
			return nil, err
		}
		document, err := closureledger.Parse(content.Data)
		if err != nil || document.ID != alias.Target {
			return nil, errors.New("magic scan: active document identity mismatch")
		}
		documents = append(documents, document)
	}
	issues, err := closurescan.ValidateBindings(snapshot, documents)
	if err != nil {
		return nil, err
	}
	if len(issues) != 0 {
		return nil, fmt.Errorf("magic scan: %d stale active binding(s), first=%s/%s", len(issues), issues[0].Kind, issues[0].Name)
	}
	bindings := map[string]bool{}
	for _, candidate := range candidates {
		binding, err := candidate.Binding()
		if err != nil {
			return nil, err
		}
		if _, active, err := closureledger.ResolveActiveBinding(
			context.Background(), store, binding, candidate.ValueJSON(),
		); err != nil {
			return nil, err
		} else if active {
			bindings[candidate.ExactKey()] = true
		}
	}
	return bindings, nil
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
	_, err := command(g.repo, "go", "run", "./cmd/device-lane", "-paths", strings.Join(g.paths, ","))
	return false, err
}

func (g *gateContext) stepAcceptance() (bool, error) {
	// cmd/plan -verify owns the verdict contract: it classifies the claim
	// (bitwise-deterministic / tolerance-bounded / stochastic-multi-seed) and
	// repeats deterministic go-test claims requiring per-test agreement. The
	// gate records the class alongside the plan reference.
	g.stepEvidence["acceptance"] = g.planRef + " verdict=" + string(acceptanceVerdictClass(g.repo))
	_, err := command(g.repo, "go", "run", "./cmd/plan", "-verify")
	return false, err
}

// acceptanceVerdictClass classifies the current step's verify command; an
// unreadable plan defaults to the strictest class.
func acceptanceVerdictClass(repo string) testevidence.VerdictClass {
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return testevidence.VerdictBitwiseDeterministic
	}
	_, step, open := plan.Current(document)
	if !open {
		return testevidence.VerdictBitwiseDeterministic
	}
	return testevidence.ClassifyVerifyCommand(step.Verify)
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
	// Validate the immutable result shape before Git advances. A schema error
	// discovered after commit cannot be represented by the normal debt batch.
	recipeID, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte(gateRecipeSeed))
	if err != nil {
		return false, err
	}
	steps := append(slices.Clone(g.steps), runrecord.GateStep{
		Name: "commit", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 1,
	})
	if _, err := runrecord.NewGateRecord(
		recipeID, g.environment.ID, strings.Repeat("0", 40), runrecord.OutcomeSucceeded, "", 1, steps,
	); err != nil {
		return false, fmt.Errorf("pre-commit record validation: %w", err)
	}
	rollbackPlan, err := advancePlanFile(g.repo, g.planRef)
	if err != nil {
		return false, err
	}
	committed := false
	defer func() {
		if committed {
			return
		}
		_ = rollbackPlan()
		_, _ = command(g.repo, "git", "add", "-A", "--", plan.Path)
	}()
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
	committed = true
	return false, nil
}

func advancePlanFile(repo, ref string) (func() error, error) {
	itemID, stepID, ok := strings.Cut(ref, "/")
	if !ok || itemID == "" || stepID == "" {
		return nil, fmt.Errorf("advance plan: invalid reference %q", ref)
	}
	path := filepath.Join(repo, filepath.FromSlash(plan.Path))
	original, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	document, err := plan.Load(path)
	if err != nil {
		return nil, err
	}
	updated, err := plan.Advance(document, itemID, stepID)
	if err != nil {
		return nil, err
	}
	if err := plan.Save(path, updated); err != nil {
		return nil, err
	}
	return func() error { return os.WriteFile(path, original, 0o644) }, nil
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

func recordUnbatchableFailure(repo, storePath string) (artifact.ID, error) {
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); err != nil {
		return artifact.ID{}, err
	}
	if err := heartbeat.Validate(); err != nil || heartbeat.State != runrecord.HeartbeatRecordDebt {
		return artifact.ID{}, errors.New("gate: no valid record-debt heartbeat to finalize")
	}
	store, err := repodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	content, ok, err := store.Content(context.Background(), heartbeat.Preparation)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("gate: prepared lifecycle unavailable: %w", err)
	}
	if !ok {
		return artifact.ID{}, errors.New("gate: prepared lifecycle unavailable")
	}
	preparation, err := runrecord.ParseGateLifecycle(content.Data)
	if err != nil || preparation.State != runrecord.GatePrepared || preparation.Environment != heartbeat.Environment {
		return artifact.ID{}, errors.New("gate: record-debt heartbeat contradicts its preparation")
	}
	codeCommit, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return artifact.ID{}, err
	}
	codeCommit = strings.TrimSpace(codeCommit)
	recipeID, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte(gateRecipeSeed))
	if err != nil {
		return artifact.ID{}, err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, preparation.Environment, codeCommit, runrecord.OutcomeFailed, "record", 1,
		[]runrecord.GateStep{{Name: "record", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepFailed, DurationNS: 1}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := record.Batch("gate/final/" + preparation.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	finalized, err := runrecord.NewGateFinalization(preparation, codeCommit, record.Result.ID, runrecord.OutcomeFailed)
	if err != nil {
		return artifact.ID{}, err
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		return artifact.ID{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipeID})
	batch.Contents = append(batch.Contents, finalizedContent)
	batch.Lineage = append(batch.Lineage, finalized.Lineage()...)
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return artifact.ID{}, err
	}
	heartbeat.State, heartbeat.PID, heartbeat.Updated = runrecord.HeartbeatFinalized, os.Getpid(), time.Now().UTC()
	if err := writeJSON(repo, gateHeartbeatFile, heartbeat, 0o644); err != nil {
		return artifact.ID{}, err
	}
	return preparation.ID, nil
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
	if outcome == runrecord.OutcomeSucceeded {
		if err := g.appendProfileEvidence(&batch, codeCommit, record.Result.ID); err != nil {
			return err
		}
	}
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

func (g *gateContext) appendProfileEvidence(batch *artifact.Batch, codeCommit string, gateResult artifact.ID) error {
	if g.profile == nil {
		return nil
	}
	if g.profileDirty {
		g.honesty = append(g.honesty, "code profile evidence not persisted: unplanned Go dirt is outside the committed target")
		return nil
	}
	evidence, err := codeprofile.NewEvidence(codeCommit, gateResult, *g.profile)
	if err != nil {
		return err
	}
	content, err := evidence.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, evidence.Lineage()...)
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

func (g *gateContext) printSummary(output io.Writer, outcome runrecord.Outcome, failure string) {
	var run, skipped []string
	for _, step := range g.steps {
		if step.Outcome == runrecord.StepSkipped {
			skipped = append(skipped, step.Name)
		} else {
			run = append(run, step.Name)
		}
	}
	fmt.Fprintf(output, "GATE %s %.1fs | ran=%s | skipped=%s\n", strings.ToUpper(string(outcome)), time.Since(g.start).Seconds(), strings.Join(run, ","), strings.Join(skipped, ","))
	if failure != "" {
		fmt.Fprintf(output, "failure: %s\n", failure)
	}
	for _, line := range compactHonesty(g.honesty) {
		fmt.Fprintln(output, line)
	}
}

func compactHonesty(lines []string) []string {
	var output []string
	for _, line := range lines {
		label := ""
		switch {
		case strings.Contains(line, "code profile delta vs HEAD"):
			label = "delta: "
		case strings.HasPrefix(line, "automation ROI"):
			label = "roi: "
		case strings.HasPrefix(line, "consumer census"):
			label = "consumer: "
		case strings.HasPrefix(line, "test scope:"):
			label = "scope: "
		case strings.Contains(line, " reused:"):
			label = "reuse: "
		case strings.Contains(line, "exact_clone_") && !strings.Contains(line, "exact_clone_production=none; exact_clone_validator=none; exact_clone_test=none"):
			label = "review: "
		case strings.Contains(line, "uncatalogued") || strings.Contains(line, "unplanned dirty") ||
			strings.Contains(line, "unavailable") || strings.Contains(line, "unreadable") || strings.Contains(line, "not persisted"):
			label = "warning: "
		}
		if label == "" {
			continue
		}
		line = label + line
		if len(line) > 600 {
			line = line[:600] + "..."
		}
		output = append(output, line)
	}
	return output
}

func command(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%s %s: %v: %s", name, strings.Join(args, " "), err, clioptions.Tail(string(out), 2000))
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
