// gate: the commit gate. One command owns scope refusal, hygiene, derived
// test scope, claim/manifest/SBOM verification, the scoped commit, and the
// store record. Never raw `git commit` during a campaign — the guard enforces
// that; this binary is the sanctioned path and marks its own commit
// subprocess with guard.GateEnv=1 (the guard owns that env-var name).
//
// Message files preserve shell-sensitive prose, scope is exact, and RepoDB is
// authoritative; green output names what did not run.
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
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/codeprofile"
	"overgo/internal/finding"
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
	gateProgressLine  = "gate: phase=%s heartbeat=%s\n"
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
	baseSource   *repoanalysis.SourceSnapshot
	profile      *codeprofile.Profile
	profileDirty bool
	stepEvidence map[string]string
	cachePaths   []string
	retryCache   *automationcheck.EvidenceCache
	structural   *codeprofile.FunctionImpact
	packageGraph *packageInputGraph
	selection    automationcheck.SelectionMetrics
	selectionID  string
}

func main() {
	clioptions.MainNamed("gate", run)
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file (required; never -m: the shell eats backticks)")
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships (required unless -merge)")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (relative to repo root)")
	merge := flag.Bool("merge", false, "finalize an in-progress merge: derive the shipped paths from the staged merge set and let the commit record both parents (stage it first with `git merge --no-ff --no-commit <branch>`)")
	planRef := flag.String("plan", "", "item/step this commit serves; MUST equal the current open step, including for -merge. Off-plan commits are refused.")
	reconcile := flag.Bool("reconcile", false, "finalize the deterministic RepoDB batch in bin/gate_debt.json")
	recordFailure := flag.Bool("record-failure", false, "recover an unbatchable post-commit record as a typed failed finalization")
	admitReview := flag.String("admit-review", "", "read-only: admit a RepoDB review-verdict ID against the current HEAD")
	watchdog := flag.Bool("watchdog", false, "print typed JSON liveness from bin/gate_lifecycle.json")
	staleAfter := flag.Duration("stale-after", runrecord.DefaultHeartbeatStaleAfter, "heartbeat age classified stale by -watchdog")
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
		stepEvidence: map[string]string{},
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
		failureCode = g.steps[len(g.steps)-1].Name
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
	if err := plan.Validate(document); err != nil {
		return err
	}
	role, err := plan.AutomationRole("")
	if err != nil {
		return err
	}
	it, st, open := plan.Current(document, role)
	if !open {
		return fmt.Errorf("gate: -plan %s given but the plan is COMPLETE (no open step) -- nothing to commit against", ref)
	}
	if item != it.ID || stepID != st.ID {
		return fmt.Errorf("gate: -plan %s does NOT match the plan's current open step %s/%s -- commit only the dispatched step (off-plan commit REFUSED). If the plan is wrong, fix the plan first; do not commit around it", ref, it.ID, st.ID)
	}
	return nil
}

func (g *gateContext) pipelineChecks(devicePackages ...string) []automationcheck.Check {
	generated := automationcheck.GeneratedChecks(g.repo, command)
	device := automationcheck.DeviceCheck(g.repo, g.paths, devicePackages, command)
	published := automationcheck.PublishedCheck(g.repo, command)
	checks := []automationcheck.Check{
		gateCheck("protection", runrecord.PhaseValidate, g.stepProtection), gateCheck("scope", runrecord.PhaseValidate, g.stepScope),
		gateCheck("profile", runrecord.PhaseValidate, g.stepProfile), gateCheck("fmt", runrecord.PhaseValidate, g.stepFmt),
		gateCheck("style", runrecord.PhaseValidate, g.stepStyle), generated[0], generated[1], generated[2], published,
		gateCheck("docs", runrecord.PhaseValidate, g.stepDocumentation), gateCheck("magics", runrecord.PhaseValidate, g.stepMagics),
		gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance), gateCheck("vet", runrecord.PhaseVet, g.stepVet),
		gateCheck("build", runrecord.PhaseBuild, g.stepBuild), gateCheck("test", runrecord.PhaseTest, g.stepTest),
		device, gateCheck("commit", runrecord.PhasePackage, g.stepCommit),
	}
	dependencies := map[string][]string{
		"scope": {"protection"}, "profile": {"scope"}, "fmt": {"profile"}, "style": {"fmt"},
		"manifest": {"style"}, "sbom": {"style"}, "claims": {"style"},
		"docs": {"manifest", "sbom", "claims"}, "magics": {"docs"}, "acceptance": {"magics"},
		"vet": {"acceptance"}, "build": {"acceptance"}, "test": {"vet", "build"},
		"device": {"test"}, "commit": {"device"},
	}
	for index := range checks {
		checks[index].Descriptor.Dependencies = dependencies[checks[index].Descriptor.Name]
	}
	return checks
}

func gateCheck(name string, phase runrecord.Phase, run func() (bool, error)) automationcheck.Check {
	return automationcheck.Check{
		Descriptor: automationcheck.Descriptor{Name: name, Phase: phase, Always: true},
		Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
			skipped, err := run()
			return skipped, "", err
		},
	}
}

func (g *gateContext) pipeline() error {
	graph, graphErr := g.inputGraph()
	var devicePackages []string
	if graphErr == nil {
		devicePackages, graphErr = graph.dependentDirectories("internal/cuda")
	}
	definitions := g.pipelineChecks(devicePackages...)
	structural, structuralErr := g.deriveStructuralImpact()
	surface := automationcheck.Surface{}
	if structuralErr == nil {
		surface = ownershipSurface(structural)
	} else {
		g.honesty = append(g.honesty, "structural impact unavailable; owned checks defaulted to run: "+structuralErr.Error())
	}
	if graphErr != nil {
		surface.Unknown = append(surface.Unknown, "package ownership: "+graphErr.Error())
		g.honesty = append(g.honesty, "package ownership unavailable; owned checks defaulted to run: "+graphErr.Error())
	}
	impact := automationcheck.OwnershipImpact(definitions, surface)
	g.selection = automationcheck.MeasureSelection(definitions, impact)
	g.selectionID = surface.Identity
	checks, err := automationcheck.Plan(definitions, impact)
	if err != nil {
		return err
	}
	cache := g.loadRetryCache()
	cache.Compact()
	g.retryCache = &cache
	cacheable := map[string]bool{"vet": true, "build": true}
	inputs := make(map[artifact.ID]artifact.ID, len(cacheable))
	for _, check := range checks {
		if cacheable[check.Check.Name] {
			input, inputErr := g.phaseInputFingerprint(check.Check.Name)
			if inputErr != nil {
				return inputErr
			}
			inputs[check.ID] = input
		}
	}
	satisfied := make(map[string]bool, len(impact.Exclusions))
	for _, exclusion := range impact.Exclusions {
		satisfied[exclusion.Check] = true
	}
	var cacheMutex sync.Mutex
	results, err := automationcheck.ExecuteDAG(context.Background(), checks, satisfied, func(ctx context.Context, check automationcheck.Invocation) (automationcheck.Evidence, error) {
		fmt.Fprintf(os.Stderr, gateProgressLine, check.Check.Name, runrecord.HeartbeatRunning)
		input, cacheCheck := inputs[check.ID]
		if cacheCheck {
			cacheMutex.Lock()
			evidence, reused := cache.Lookup(check, input)
			cacheMutex.Unlock()
			if reused {
				return evidence, nil
			}
		}
		evidence, runErr := automationcheck.Run(ctx, check)
		if runErr == nil && cacheCheck {
			cacheMutex.Lock()
			cache.Record(check, input, evidence)
			cacheMutex.Unlock()
		}
		return evidence, runErr
	})
	if err != nil {
		return err
	}
	g.saveRetryCache(cache)
	byName := make(map[string]automationcheck.DAGResult, len(results))
	for _, result := range results {
		if result.Invocation.ID.Valid() {
			byName[result.Invocation.Check.Name] = result
		}
	}
	for _, definition := range definitions {
		name := definition.Descriptor.Name
		result, ran := byName[name]
		if !ran {
			if exclusion, excluded := impact.ExclusionReason(name); excluded {
				g.steps = append(g.steps, runrecord.GateStep{Name: name, Phase: definition.Descriptor.Phase, Outcome: runrecord.StepSkipped, DurationNS: uint64(time.Nanosecond)})
				g.honesty = append(g.honesty, name+" skipped: "+exclusion)
			}
			continue
		}
		g.steps = append(g.steps, gateEvidenceRecord(name, result.Invocation.Check.Phase, result.Evidence, result.Err, g.stepEvidence[name]))
		if result.Evidence.Reused {
			g.honesty = append(g.honesty, name+" reused: derived inputs already passed this step")
		}
		if result.Err != nil {
			return fmt.Errorf("%s: %w", name, result.Err)
		}
	}
	return nil
}

func (g *gateContext) inputGraph() (packageInputGraph, error) {
	if g.packageGraph != nil {
		return *g.packageGraph, nil
	}
	graph, err := loadPackageInputGraph(g.repo)
	if err == nil {
		g.packageGraph = &graph
	}
	return graph, err
}

func (g *gateContext) deriveStructuralImpact() (codeprofile.FunctionImpact, error) {
	if g.structural != nil {
		return *g.structural, nil
	}
	candidate, err := g.sourceSnapshot()
	if err != nil {
		return codeprofile.FunctionImpact{}, err
	}
	base, err := sourceAtHEAD(g.repo, candidate)
	if err != nil {
		return codeprofile.FunctionImpact{}, err
	}
	selection, err := repoanalysis.HostBuildSelection(g.repo, "./cmd/...", "./internal/...")
	if err != nil {
		return codeprofile.FunctionImpact{}, err
	}
	impact, err := codeprofile.DeriveFunctionImpact(base, candidate, selection, selection, g.paths)
	if err == nil {
		g.structural, g.baseSource = &impact, &base
	}
	return impact, err
}

func ownershipSurface(impact codeprofile.FunctionImpact) automationcheck.Surface {
	packages := map[string]bool{}
	for _, packagePath := range impact.Packages {
		packages[packagePath] = true
	}
	symbols := make([]automationcheck.Symbol, 0, len(impact.Reachable))
	for _, symbol := range impact.Reachable {
		packagePath := path.Dir(symbol.File)
		packages[packagePath] = true
		symbols = append(symbols, automationcheck.Symbol{
			Package: packagePath, Receiver: symbol.Receiver, Name: symbol.Name,
		})
	}
	unknown := make([]string, 0, len(impact.Unknown))
	for _, boundary := range impact.Unknown {
		unknown = append(unknown, strings.Join([]string{boundary.Kind, boundary.Path, boundary.Symbol}, ":"))
	}
	return automationcheck.Surface{
		Identity: impact.BaseIdentity + ":" + impact.CandidateIdentity,
		Packages: slices.Sorted(maps.Keys(packages)), Symbols: symbols, Unknown: unknown,
	}
}

func gateEvidenceRecord(name string, phase runrecord.Phase, evidence automationcheck.Evidence, runErr error, storedEvidence string) runrecord.GateStep {
	record := runrecord.GateStep{
		Name: name, Phase: phase, Outcome: runrecord.StepSucceeded,
		DurationNS: max(evidence.DurationNS, uint64(1)), Evidence: storedEvidence,
	}
	if record.Evidence == "" && evidence.ID.Valid() {
		record.Evidence = evidence.ID.String()
	}
	switch {
	case runErr != nil:
		record.Outcome = runrecord.StepFailed
	case evidence.Skipped:
		record.Outcome = runrecord.StepSkipped
	case evidence.Reused:
		record.Outcome = runrecord.StepReused
	}
	return record
}

const (
	historicalRankingMarker = "<!-- overgo-document: historical-ranking -->"
	currentWorkMarker       = "<!-- overgo-current-work: docs/plan.json -->"
)

// stepDocumentation keeps prose assessments from masquerading as current
// work authority. Ranked current work belongs to the failable plan; prose may
// retain its historical ordering only with an explicit warning and redirect.
func (g *gateContext) stepDocumentation() (bool, error) {
	document, err := plan.Load(filepath.Join(g.repo, plan.Path))
	if err != nil {
		return false, err
	}
	if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
		return false, err
	}
	store, err := repodb.OpenReadOnly(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return false, err
	}
	_, found, readErr := closurescan.ReadCensusEvidence(context.Background(), store, *document.Census)
	closeErr := store.Close()
	if readErr != nil || closeErr != nil {
		return false, errors.Join(readErr, closeErr)
	}
	if !found {
		return false, errors.New("gate: campaign census evidence is absent from RepoDB")
	}
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
	profile.Impact = codeprofile.ImpactSelection{
		Identity: g.selectionID, Owned: g.selection.Owned, Triggered: g.selection.Triggered,
		Excluded: g.selection.Excluded, Unresolved: g.selection.Unresolved,
	}
	baseSource, err := sourceAtHEAD(g.repo, snapshot)
	if err != nil {
		return false, err
	}
	base, err := codeprofile.Build(baseSource)
	if err != nil {
		return false, err
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"code profile: runtime=%d files/%d nodes automation=%d/%d generated=%d/%d test=%d/%d duplicate_excess=%d clones=%d functions=%d exported=%d imports=%d",
		profile.Runtime.Files, profile.Runtime.Nodes, profile.Automation.Files, profile.Automation.Nodes,
		profile.Generated.Files, profile.Generated.Nodes, profile.Test.Files, profile.Test.Nodes, profile.DuplicateExcessNodes,
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
	g.baseSource = &baseSource
	g.profile = &profile
	return false, nil
}

func (g *gateContext) appendConsumerCensus(candidate, head repoanalysis.SourceSnapshot, changed []string, profile *codeprofile.Profile) error {
	selection, err := repoanalysis.HostBuildSelection(g.repo, "./cmd/...", "./internal/...")
	if err != nil {
		return err
	}
	impact := codeprofile.FunctionImpact{}
	if g.structural != nil {
		impact = *g.structural
	} else {
		impact, err = codeprofile.DeriveFunctionImpact(head, candidate, selection, selection, g.paths)
		if err != nil {
			return err
		}
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"function impact: base=%s candidate=%s seeds=%d reachable=%d unknown=%d",
		impact.BaseIdentity, impact.CandidateIdentity, len(impact.Seeds), len(impact.Reachable), len(impact.Unknown),
	))
	g.honesty = append(g.honesty, impactSelectionHonesty(profile.Impact))
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

func impactSelectionHonesty(selection codeprofile.ImpactSelection) string {
	return fmt.Sprintf(
		"impact selection: excluded=%d/%d triggered=%d unresolved=%d snapshot=%s",
		selection.Excluded, selection.Owned, selection.Triggered, selection.Unresolved, selection.Identity,
	)
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
	declarations = slices.DeleteFunc(slices.Clone(declarations), func(value codeprofile.ConsumerDeclaration) bool {
		return value.ProductionReferences > 0 || value.Boundary != ""
	})
	return fmt.Sprintf(
		"consumer census %s context=%s delta: production=%+d test_only=%+d boundary=%+d zero=%+d; current=%d/%d/%d/%d; candidates=%s; advisory_only=ambiguous dispatch is a boundary, tests are not production consumers",
		scope, context, current.Production-base.Production, current.TestOnly-base.TestOnly,
		current.Boundary-base.Boundary, current.Zero-base.Zero,
		current.Production, current.TestOnly, current.Boundary, current.Zero, consumerCandidates(declarations),
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

func surfaceDeltaHonesty(base, candidate codeprofile.Profile) string {
	runtimeFiles, runtimeNodes := candidate.Runtime.Files-base.Runtime.Files, candidate.Runtime.Nodes-base.Runtime.Nodes
	automationFiles, automationNodes := candidate.Automation.Files-base.Automation.Files, candidate.Automation.Nodes-base.Automation.Nodes
	duplicateExcess := candidate.DuplicateExcessNodes - base.DuplicateExcessNodes
	adverse := "none"
	if duplicateExcess < 0 && (runtimeFiles+automationFiles > 0 || runtimeNodes+automationNodes > 0) {
		adverse = "duplication fell while production grew; reduction does not offset surface growth"
	}
	return fmt.Sprintf(
		"code profile delta vs HEAD: runtime=%+d files/%+d nodes automation=%+d/%+d generated=%+d/%+d test=%+d/%+d duplicate_excess=%+d clones=%+d function_count=%+d exported=%+d imports=%+d; adverse_pattern=%s",
		runtimeFiles, runtimeNodes, automationFiles, automationNodes,
		candidate.Generated.Files-base.Generated.Files, candidate.Generated.Nodes-base.Generated.Nodes,
		candidate.Test.Files-base.Test.Files, candidate.Test.Nodes-base.Test.Nodes, duplicateExcess,
		len(candidate.Clones)-len(base.Clones), len(candidate.Functions)-len(base.Functions),
		candidate.ExportedDeclarations-base.ExportedDeclarations, candidate.PackageImportEdges-base.PackageImportEdges, adverse,
	)
}

func profileReviewFocus(profile codeprofile.Profile, changed []string) string {
	paths := pathSet(changed)
	cloneFocus := "none"
	for _, candidate := range profile.Clones {
		if !cliMainClone(candidate) && slices.ContainsFunc(candidate.Functions, func(function string) bool {
			path, _, ok := strings.Cut(function, ":")
			return ok && paths[path]
		}) {
			cloneFocus = fmt.Sprintf("nodes=%d functions=%s", candidate.Nodes, strings.Join(candidate.Functions, ","))
			break
		}
	}
	return fmt.Sprintf("code review candidate: exact_clone=%s; advisory_only=inspect semantic ownership and numerical contracts, migrate callers and delete displaced paths, require parity evidence", cloneFocus)
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

func (g *gateContext) loadRetryCache() automationcheck.EvidenceCache {
	empty := automationcheck.NewEvidenceCache(g.environment.ID)
	var cache automationcheck.EvidenceCache
	if readJSON(g.repo, gateRetryFile, &cache) != nil || !cache.Reusable(g.environment.ID) {
		return empty
	}
	return cache
}

func (g *gateContext) saveRetryCache(cache automationcheck.EvidenceCache) {
	if cache.Environment.Valid() {
		_ = writeJSON(g.repo, gateRetryFile, cache, 0o644)
	}
}

func (g *gateContext) phaseInputFingerprint(phase string) (artifact.ID, error) {
	paths := g.cachePaths
	var err error
	if paths == nil {
		paths, err = gitLines(g.repo, "ls-files", "-co", "--exclude-standard")
		if err != nil {
			return artifact.ID{}, err
		}
		g.cachePaths = paths
	}
	return fingerprintPhaseInputs(g.repo, phase, paths)
}

func fingerprintPhaseInputs(root, phase string, paths []string) (artifact.ID, error) {
	var selected []string
	for _, path := range paths {
		path = filepath.ToSlash(path)
		if phaseOwnsPath(phase, path) {
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
			return artifact.ID{}, err
		}
		hasher.Write(data)
		hasher.Write([]byte{0})
	}
	return artifact.IdentifyBytes(artifact.KindEvidence, hasher.Sum(nil))
}

func phaseOwnsPath(phase, path string) bool {
	goSource := path == "go.mod" || path == "go.sum" || strings.HasSuffix(path, ".go")
	goInput := goSource ||
		(strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "cmd/")) && !strings.HasSuffix(path, ".md")
	switch phase {
	case "vet", "build":
		return goInput
	case "test":
		return goInput || path == "README.md" || strings.HasPrefix(path, "docs/") && path != plan.Path
	default:
		return false
	}
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

func (g *gateContext) stepStyle() (bool, error) {
	if len(g.changedGoFiles()) == 0 {
		return true, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	if g.baseSource == nil {
		baseline, err := sourceAtHEAD(g.repo, snapshot)
		if err != nil {
			return false, err
		}
		g.baseSource = &baseline
	}
	return false, repoanalysis.ValidateGoStyleDelta(snapshot, *g.baseSource)
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
	inputGraph, err := g.inputGraph()
	if err != nil {
		return false, err
	}
	directInputs, err := packageInputIdentities(inputGraph, direct)
	if err != nil {
		return false, err
	}
	directPending, directReused, err := g.packageCachePartition(direct, "short", directInputs)
	if err != nil {
		return false, err
	}
	if len(directPending) > 0 {
		if _, err := runGoTests(g.repo, directPending); err != nil {
			return false, err
		}
		if err := g.recordPackagePasses(directPending, "short", directInputs); err != nil {
			return false, err
		}
	}
	if len(dependent) == 0 {
		g.packageCacheHonesty(directReused, len(directPending))
		return false, nil
	}
	dependentInputs, err := packageInputIdentities(inputGraph, dependent)
	if err != nil {
		return false, err
	}
	dependentPending, dependentReused, err := g.packageCachePartition(dependent, "complete", dependentInputs)
	if err != nil {
		return false, err
	}
	report := testevidence.GoTestReport{}
	if len(dependentPending) > 0 {
		report, err = runGoTestsAdvisory(g.repo, dependentPending)
	}
	if err != nil {
		return false, err
	}
	if len(report.Skipped)+len(report.Unavailable) > 0 {
		g.honesty = append(g.honesty, fmt.Sprintf(
			"dependent fixture evidence not credited: %d skipped, %d unavailable",
			len(report.Skipped), len(report.Unavailable),
		))
	} else if err := g.recordPackagePasses(dependentPending, "complete", dependentInputs); err != nil {
		return false, err
	}
	g.packageCacheHonesty(directReused+dependentReused, len(directPending)+len(dependentPending))
	return false, nil
}

func (g *gateContext) packageCachePartition(packages []string, mode string, inputs map[string]artifact.ID) ([]string, int, error) {
	if g.retryCache == nil {
		cache := g.loadRetryCache()
		cache.Compact()
		g.retryCache = &cache
	}
	pending := make([]string, 0, len(packages))
	reused := 0
	for _, packagePath := range packages {
		hit, err := g.retryCache.PackageReusable(packagePath, mode, inputs[packagePath])
		if err != nil {
			return nil, 0, err
		}
		if hit {
			reused++
		} else {
			pending = append(pending, packagePath)
		}
	}
	return pending, reused, nil
}

func (g *gateContext) recordPackagePasses(packages []string, mode string, inputs map[string]artifact.ID) error {
	for _, packagePath := range packages {
		if err := g.retryCache.RecordPackagePass(packagePath, mode, inputs[packagePath]); err != nil {
			return err
		}
	}
	g.saveRetryCache(*g.retryCache)
	return nil
}

func (g *gateContext) packageCacheHonesty(reused, executed int) {
	if reused+executed > 0 {
		g.honesty = append(g.honesty, fmt.Sprintf("package test evidence: %d reused + %d executed", reused, executed))
	}
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

// stepMagics enforces repository-wide zero debt.
func (g *gateContext) stepMagics() (bool, error) {
	goSource := false
	for _, path := range g.paths {
		if strings.HasSuffix(path, ".go") {
			goSource = true
		}
	}
	if !goSource {
		return true, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	documents, err := activeMagicBindings(g.repo, g.storePath)
	if err != nil {
		return false, err
	}
	report, err := closurescan.ValidatePermanentAuthority(snapshot, documents)
	if err != nil {
		return false, err
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"permanent magic authority: production=%d classified=%d tests=%d open=0 stale=0 policy_copies=0",
		report.ProductionSites, report.ClassifiedSites, report.TestSites,
	))
	return false, nil
}

func activeMagicBindings(repo, storePath string) ([]closureledger.Document, error) {
	store, err := repodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return nil, err
	}
	defer store.Close()
	result, err := store.Query(context.Background(), repodb.Query{
		Kind: artifact.KindEvidence, MediaType: closureledger.MediaType,
		Schema: closureledger.Schema, MaxResults: store.QueryExtent(),
		Projection: repodb.ProjectAliases | repodb.ProjectContentData,
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
		data, found := result.Content(alias.Target)
		if !found {
			return nil, errors.New("magic scan: active document unavailable")
		}
		document, err := closureledger.Parse(data)
		if err != nil || document.ID != alias.Target {
			return nil, errors.New("magic scan: active document identity mismatch")
		}
		documents = append(documents, document)
	}
	return documents, nil
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
	role, roleErr := plan.AutomationRole("")
	if roleErr != nil {
		return testevidence.VerdictBitwiseDeterministic
	}
	_, step, open := plan.Current(document, role)
	if !open {
		return testevidence.VerdictBitwiseDeterministic
	}
	return testevidence.ClassifyVerifyCommand(step.Verify)
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
		Version: artifact.InitialDocumentVersion, State: state, Preparation: g.preparation.ID,
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
	debt := gateDebtEnvelope{Version: artifact.InitialDocumentVersion, Preparation: g.preparation.ID, Batch: batch}
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
	if debt.Version != artifact.InitialDocumentVersion || debt.Preparation.Kind() != artifact.KindEvidence {
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
		return printJSON(watchdogStatus{Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatAbsent})
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
	if err := appendGateAdvisoryFinding(context.Background(), store, &batch, g.paths, g.honesty); err != nil {
		return g.oweRecord(batch, err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return g.oweRecord(batch, err)
	}
	_ = os.Remove(filepath.Join(g.repo, filepath.FromSlash(gateDebtFile)))
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

func (g *gateContext) printSummary(output io.Writer, outcome runrecord.Outcome, failure string) {
	var run, reused, skipped []string
	for _, step := range g.steps {
		switch step.Outcome {
		case runrecord.StepSkipped:
			skipped = append(skipped, step.Name)
		case runrecord.StepReused:
			reused = append(reused, step.Name)
		default:
			run = append(run, step.Name)
		}
	}
	fmt.Fprintf(output, "GATE %s %.1fs | ran=%s | reused=%s | skipped=%s\n", strings.ToUpper(string(outcome)), time.Since(g.start).Seconds(), strings.Join(run, ","), strings.Join(reused, ","), strings.Join(skipped, ","))
	if failure != "" {
		fmt.Fprintf(output, "blocker: %s\n", failure)
	}
	for _, line := range compactHonesty(g.honesty) {
		fmt.Fprintln(output, line)
	}
}

func appendGateAdvisoryFinding(ctx context.Context, store *repodb.Store, batch *artifact.Batch, owners, honesty []string) error {
	evidence := slices.DeleteFunc(compactHonesty(honesty), func(line string) bool {
		return !strings.HasPrefix(line, "advisory: review:") && !strings.HasPrefix(line, "advisory: warning:") &&
			(!strings.HasPrefix(line, "advisory: consumer:") || strings.Contains(line, "candidates=;"))
	})
	if len(evidence) == 0 {
		return nil
	}
	if len(owners) == 0 {
		owners = []string{"repository"}
	}
	document, findingBatch, err := finding.NewTextBatch("Actionable gate advisories", finding.SeverityMedium, owners, evidence,
		"Resolve each advisory at its owning source and retain a failable regression check.", "The gate emits no actionable advisory for the same owner surface.")
	if err != nil {
		return err
	}
	alias := artifact.AliasBinding{Name: "finding/active/gate-advisories", Target: document.ID}
	if previous, found, err := artifact.ResolveAlias(ctx, store, alias.Name); err != nil {
		return err
	} else if found {
		alias.Previous = &previous
	}
	batch.Contents = append(batch.Contents, findingBatch.Contents...)
	batch.Lineage = append(batch.Lineage, findingBatch.Lineage...)
	batch.Aliases = append(batch.Aliases, alias)
	return nil
}

func compactHonesty(lines []string) []string {
	var output []string
	for _, line := range lines {
		label := ""
		switch {
		case strings.Contains(line, "code profile delta vs HEAD"):
			label = "advisory: delta: "
		case strings.HasPrefix(line, "automation ROI"):
			label = "advisory: roi: "
		case strings.HasPrefix(line, "consumer census"):
			label = "advisory: consumer: "
		case strings.HasPrefix(line, "impact selection:"):
			label = "advisory: impact: "
		case strings.HasPrefix(line, "test scope:"):
			label = "advisory: scope: "
		case strings.HasPrefix(line, "package test evidence:"):
			label = "advisory: reuse: "
		case strings.Contains(line, " reused:"):
			label = "advisory: reuse: "
		case strings.Contains(line, "exact_clone=") && !strings.Contains(line, "exact_clone=none"):
			label = "advisory: review: "
		case strings.Contains(line, "uncatalogued") || strings.Contains(line, "unplanned dirty") ||
			strings.Contains(line, "unavailable") || strings.Contains(line, "unreadable") || strings.Contains(line, "not persisted"):
			label = "advisory: warning: "
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
