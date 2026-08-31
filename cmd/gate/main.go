// gate: the commit gate. One command owns scope refusal, hygiene, derived
// test scope, claim/manifest/SBOM verification, the scoped commit, and the
// store record. Never raw `git commit` during a campaign — the guard enforces
// that; this binary is the sanctioned path and constructs the exact accepted
// commit object before compare-and-swapping the captured branch reference.
//
// Message files preserve shell-sensitive prose, scope is exact, and OvergoDB is
// authoritative; green output names what did not run.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
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
	"strconv"
	"strings"
	"sync"
	"time"

	"overgo/internal/agentworkflow"
	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/authoritylock"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/finding"
	"overgo/internal/fsatomic"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/processlock"
	"overgo/internal/protection"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testscope"
)

const (
	gateRecipeSeed   = "overgo-gate/v1"
	gateWorkloadSeed = "overgo-gate-workload/v1"
	gateStorePath    = "overgodb-store"
	// Gate state lives in tmp/, the sanctioned scrap home: bin/ holds
	// executables only and the release refuses anything else in it.
	gateDebtFile         = "tmp/gate_debt.json"
	gateCommitIntentFile = "tmp/gate_commit_intent.json"
	gateHeartbeatFile    = "tmp/gate_lifecycle.json"
	gateRetryFile        = "tmp/gate_cache.json"
	gatePlanScratchDir   = "tmp/gate_plan_scratch"
	gateGitStateLockFile = "overgo-gate-git-state.lock"
	gateProgressLine     = "gate: phase=%s heartbeat=%s\n"
	// Recovery roots and transient Git authority must not be readable by
	// other users. Directory traversal is likewise restricted to the owner.
	gatePrivateFileMode      = fs.FileMode(0o600)
	gatePrivateDirectoryMode = fs.FileMode(0o700)
)

type gateContext struct {
	repo                string
	paths               []string
	planRef             string
	messageFile         string
	storePath           string
	steps               []runrecord.GateStep
	honesty             []string
	start               time.Time
	environment         runrecord.Environment
	preparation         runrecord.GateLifecycle
	preparationCommit   artifact.CommitID
	source              *repoanalysis.SourceSnapshot
	baseSource          *repoanalysis.SourceSnapshot
	profile             *codeprofile.Profile
	profileDirty        bool
	stepEvidence        map[string]string
	cachePaths          []string
	retryCache          *automationcheck.EvidenceCache
	structural          *codeprofile.FunctionImpact
	packageGraph        *packageInputGraph
	selection           automationcheck.SelectionMetrics
	selectionID         string
	manifestPlan        *automationcheck.ManifestPlan
	terminal            map[string]automationcheck.Evidence
	baseManifest        *codemanifest.Manifest
	candidateManifest   *codemanifest.Manifest
	manifestDelta       *codemanifest.Delta
	manifestImpact      *codemanifest.Impact
	manifestMetrics     automationcheck.ManifestMeasurements
	strategy            *loop.Strategy
	diff                runrecord.AttemptDiff
	completionAuthority plan.CompletionAuthority
	completionStore     *overgodb.Store
	indexBefore         gateIndexSnapshot
	mergeBefore         *gateMergeIntent
	planProjection      plan.MergeProjection
	mergeSourceStore    string
	mergeAuthority      *plan.FirstParentTargetMergeAuthority
	planHead            string
	committedHead       string
	commitInterrupted   bool
	acceptedTree        string
	// runCommand overrides supervised command execution for remediation
	// tests; nil routes through the package command runner.
	runCommand func(repo, name string, args ...string) (string, error)
}

func (g *gateContext) runGateCommand(name string, args ...string) (string, error) {
	if g.runCommand != nil {
		return g.runCommand(g.repo, name, args...)
	}
	return command(g.repo, name, args...)
}

func main() {
	clioptions.MainNamed("gate", run)
}

func run() error {
	messageFile := flag.String("message-file", "", "commit message file (required; never -m: the shell eats backticks)")
	pathsCSV := flag.String("paths", "", "comma-separated repo-relative paths this commit ships (required unless -merge)")
	storePath := flag.String("store", gateStorePath, "canonical OvergoDB store directory")
	merge := flag.Bool("merge", false, "finalize an in-progress merge: derive the shipped paths from the staged merge set and let the commit record both parents (stage it first with `git merge --no-ff --no-commit <branch>`)")
	planProjectionFlag := flag.String("plan-projection", "", "with -merge only: explicit target-plan projection (first-parent-target); empty keeps semantic union")
	mergeSourceStoreFlag := flag.String("merge-source-store", "", "with a first-parent-target merge: canonical OvergoDB store of the registered source worktree")
	planRef := flag.String("plan", "", "item/step this commit serves; MUST equal the current open step, including for -merge. Off-plan commits are refused.")
	reconcile := flag.Bool("reconcile", false, "finalize the deterministic OvergoDB batch in tmp/gate_debt.json")
	recordFailure := flag.Bool("record-failure", false, "recover an unbatchable post-commit record as a typed failed finalization")
	recoverPreparation := flag.String("preparation", "", "exact stale preparation ID -record-failure closes when recovery debt is ambiguous")
	recoverInterrupted := flag.Bool("recover-interrupted", false, "roll back an interrupted gate commit and cancel its prepared store lifecycle")
	admitReview := flag.String("admit-review", "", "read-only: admit a OvergoDB review-verdict ID against the current HEAD")
	watchdog := flag.Bool("watchdog", false, "print typed JSON liveness from tmp/gate_lifecycle.json")
	inspectPlan := flag.Bool("inspect-plan", false, "non-committing: construct and print the exact manifest-bound verification plan without executing checks; may write temporary Git object/index state")
	staleAfter := flag.Duration("stale-after", runrecord.DefaultHeartbeatStaleAfter, "heartbeat age classified stale by -watchdog")
	flag.Parse()
	repo, err := os.Getwd()
	if err != nil {
		return err
	}
	if err := gitauthority.RequireRepositoryRoot(context.Background(), repo); err != nil {
		return fmt.Errorf("gate: establish exact repository authority: %w", err)
	}
	cleanStore := filepath.Clean(*storePath)
	if cleanStore == "." || filepath.IsAbs(cleanStore) || cleanStore == ".." || strings.HasPrefix(cleanStore, ".."+string(filepath.Separator)) {
		return errors.New("gate: store path must stay below the repository root")
	}
	if err := requireExclusiveGateMode(
		*reconcile, *recordFailure, *recoverInterrupted, *admitReview != "", *watchdog, *inspectPlan, *merge,
	); err != nil {
		return err
	}
	planProjection, err := gatePlanProjection(*planProjectionFlag, *merge)
	if err != nil {
		return err
	}
	mergeSourceStore, err := gateMergeSourceStore(*mergeSourceStoreFlag, *merge, planProjection)
	if err != nil {
		return err
	}
	mutating := *reconcile || *recordFailure || *recoverInterrupted || *admitReview != "" || !*watchdog && !*inspectPlan
	writesGit := gateWritesGit(
		*reconcile, *recordFailure, *recoverInterrupted, *admitReview != "", *watchdog, *inspectPlan, *merge,
	)
	if writesGit {
		if err := gitauthority.RequireWriterSupport(context.Background(), repo); err != nil {
			return fmt.Errorf("gate: durable Git writer unavailable: %w", err)
		}
	}
	if err := requireCanonicalGateStore(cleanStore, mutating); err != nil {
		return err
	}
	if mutating {
		lock, err := authoritylock.Acquire(repo)
		if err != nil {
			return err
		}
		defer lock.Close()
	}
	if *reconcile {
		preparation, err := reconcileGateDebt(repo, cleanStore)
		if err == nil {
			fmt.Printf("gate: reconciled OvergoDB record debt for %s\n", preparation)
		}
		return err
	}
	if *recoverPreparation != "" && !*recordFailure {
		return errors.New("gate: -preparation requires -record-failure")
	}
	if *recordFailure {
		preparation, err := recordSelectedUnbatchableFailure(repo, cleanStore, *recoverPreparation)
		if err == nil {
			fmt.Printf("gate: recorded failed finalization for %s\n", preparation)
		}
		return err
	}
	if *recoverInterrupted {
		preparation, err := recoverInterruptedCommit(repo, cleanStore)
		if err == nil {
			fmt.Printf("gate: recovered interrupted commit for %s\n", preparation)
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
	if *inspectPlan && *merge {
		return errors.New("gate: -inspect-plan requires explicit -paths and cannot inspect an in-progress merge")
	}
	if (*pathsCSV == "" && !*merge) || (!*inspectPlan && *messageFile == "") {
		return fmt.Errorf("usage: gate -message-file <path> (-paths <csv> | -merge) -plan <item>/<step> [-store <dir>]")
	}
	// Every commit -- including a merge finalize -- is bound to the plan's current
	// open step. Merges are no longer exempt: a sync/merge is a first-class plan
	// task (inject it with `plan -add`, then finalize with -plan <item>/do).
	var completionAuthority plan.CompletionAuthority
	var planHead string
	var indexBefore gateIndexSnapshot
	var mergeBefore *gateMergeIntent
	if !*inspectPlan {
		mergeBefore, indexBefore, err = captureGateStartState(repo)
		if err != nil {
			return err
		}
		if err := requireNoPendingGateState(repo, cleanStore); err != nil {
			return err
		}
		completionAuthority, planHead, err = resolvePlanBinding(repo, cleanStore, *planRef)
		if err != nil {
			return err
		}
	}
	g := &gateContext{
		repo: repo, planRef: *planRef, messageFile: *messageFile, storePath: cleanStore, start: time.Now(),
		stepEvidence: map[string]string{}, terminal: map[string]automationcheck.Evidence{},
		completionAuthority: completionAuthority, planHead: planHead,
		indexBefore: indexBefore, mergeBefore: mergeBefore,
		planProjection: planProjection, mergeSourceStore: mergeSourceStore,
	}
	defer g.closeCompletionStore()
	if *merge {
		// Merge mode: the staged merge IS the plan. Deriving -paths from the
		// staged set makes the scope step trivially pass, and the existing
		// commit step constructs the exact two-parent commit and advances the
		// captured branch by compare-and-swap. Merges otherwise
		// bypass the gate entirely; this runs full hygiene + impacted tests on
		// the merged tree before the commit lands.
		if _, err := command(repo, "git", "rev-parse", "--verify", "-q", "MERGE_HEAD"); err != nil {
			return fmt.Errorf("-merge needs an in-progress merge (no MERGE_HEAD): run `git merge --no-ff --no-commit <branch>` first")
		}
		if g.mergeSourceStore != "" {
			if g.mergeBefore == nil {
				return errors.New("-merge-source-store requires captured merge metadata")
			}
			parents := g.mergeBefore.parents()
			if len(parents) != 1 {
				return errors.New("-merge-source-store requires exactly one captured merge parent")
			}
			registered, err := gitauthority.RequireRegisteredWorktreeStore(
				context.Background(), repo, g.mergeSourceStore, parents[0],
			)
			if err != nil {
				return fmt.Errorf("-merge-source-store: %w", err)
			}
			localStore, err := filepath.Abs(filepath.Join(repo, gateStorePath))
			if err != nil {
				return err
			}
			if strings.EqualFold(filepath.Clean(registered), filepath.Clean(localStore)) {
				return errors.New("-merge-source-store must be owned by the distinct source worktree")
			}
			g.mergeSourceStore = registered
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
	if err := validatePlannedPaths(g.paths); err != nil {
		return err
	}
	if err := g.expandDirectoryPaths(); err != nil {
		return err
	}
	if !slices.Contains(g.paths, plan.Path) {
		g.paths = append(g.paths, plan.Path)
	}
	if err := validatePlannedPaths(g.paths); err != nil {
		return err
	}
	if *inspectPlan {
		planned, err := g.planPipeline()
		if err != nil {
			return err
		}
		return writeGatePlanReport(os.Stdout, planned)
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

	// The candidate's change size is observed before the pipeline can
	// commit it; after a successful commit the worktree diff is gone.
	g.diff = observeDiff(repo)
	outcome := runrecord.OutcomeSucceeded
	failureCode := ""
	var pipelineErr error
	if pipelineErr = g.pipeline(); pipelineErr != nil {
		outcome = runrecord.OutcomeFailed
		failureCode = "planning"
		if len(g.steps) != 0 {
			failureCode = g.steps[len(g.steps)-1].Name
		}
	}
	// No running heartbeat writer may outlive the gate execution and race a
	// terminal store/heartbeat publication. A crash after this point leaves the
	// last running locator for the explicit recovery path.
	stopHeartbeat()
	if g.commitInterrupted {
		closeErr := g.closeCompletionStore()
		_, recoveryErr := recoverInterruptedCommit(repo, cleanStore)
		g.printSummary(os.Stdout, outcome, pipelineErr.Error())
		if closeErr != nil || recoveryErr != nil {
			_ = g.writeHeartbeat(runrecord.HeartbeatRecordDebt)
			return errors.Join(
				pipelineErr,
				fmt.Errorf(
					"commit landed but automatic recovery failed; run `go run ./cmd/gate -recover-interrupted`: %w",
					errors.Join(closeErr, recoveryErr),
				),
			)
		}
		_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
		return pipelineErr
	}
	recordErr := g.record(outcome, failureCode)
	recordErr = errors.Join(recordErr, g.closeCompletionStore())
	// Intent removal mutates recovery authority and therefore must remain lazy;
	// cmp.Or would evaluate the removal even when record publication failed.
	switch {
	case recordErr != nil:
	default:
		recordErr = removeGateCommitIntent(repo)
	}
	rolledBackRecord := false
	if recordErr != nil && g.committedHead != "" {
		if _, debtErr := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); errors.Is(debtErr, os.ErrNotExist) {
			if _, recoveryErr := recoverInterruptedCommit(repo, cleanStore); recoveryErr != nil {
				recordErr = errors.Join(
					recordErr,
					fmt.Errorf("automatic rollback after unbatchable record failure: %w", recoveryErr),
				)
			} else if recoveredHead, headErr := command(repo, "git", "rev-parse", "HEAD"); headErr != nil {
				recordErr = errors.Join(recordErr, fmt.Errorf("inspect automatic record recovery: %w", headErr))
			} else {
				switch strings.TrimSpace(recoveredHead) {
				case g.planHead:
					rolledBackRecord = true
					outcome = runrecord.OutcomeCancelled
				case g.committedHead:
					// The final batch landed and recovery only cleared a stale
					// marker, so the originally reported record error is resolved.
					recordErr = nil
				default:
					recordErr = errors.Join(recordErr, errors.New("automatic record recovery left HEAD at an unknown commit"))
				}
			}
		} else if debtErr != nil {
			recordErr = errors.Join(recordErr, fmt.Errorf("inspect gate record debt: %w", debtErr))
		}
	}
	failureDetail := ""
	if pipelineErr != nil {
		failureDetail = pipelineErr.Error()
	}
	g.printSummary(os.Stdout, outcome, failureDetail)
	if recordErr != nil {
		if rolledBackRecord {
			_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
			fmt.Fprintf(os.Stderr, "gate: store record failed; commit was rolled back and the preparation cancelled: %v\n", recordErr)
		} else {
			_ = g.writeHeartbeat(runrecord.HeartbeatRecordDebt)
			fmt.Fprintf(os.Stderr, "gate: store record failed (recovery or record debt remains): %v\n", recordErr)
		}
	} else {
		_ = g.writeHeartbeat(runrecord.HeartbeatFinalized)
	}
	if pipelineErr != nil {
		return pipelineErr
	}
	if recordErr != nil {
		if rolledBackRecord {
			return fmt.Errorf("commit rolled back because its OvergoDB record could not be constructed: %w", recordErr)
		}
		return fmt.Errorf("commit landed but OvergoDB record debt remains: %w", recordErr)
	}
	return nil
}

func gateWritesGit(
	reconcile, recordFailure, recoverInterrupted, admitReview, watchdog, inspectPlan, merge bool,
) bool {
	return reconcile || recoverInterrupted || inspectPlan || merge ||
		!recordFailure && !admitReview && !watchdog
}

func requireCanonicalGateStore(path string, writesAuthority bool) error {
	if writesAuthority && filepath.ToSlash(filepath.Clean(path)) != gateStorePath {
		return fmt.Errorf("gate: completion authority requires the canonical %s store", gateStorePath)
	}
	return nil
}

func requireExclusiveGateMode(
	reconcile, recordFailure, recoverInterrupted, admitReview, watchdog, inspectPlan, merge bool,
) error {
	selected := 0
	for _, enabled := range []bool{reconcile, recordFailure, recoverInterrupted, admitReview, watchdog, inspectPlan} {
		if enabled {
			selected++
		}
	}
	if selected > 1 || selected != 0 && merge {
		return errors.New("gate: operation modes are mutually exclusive")
	}
	return nil
}

func gatePlanProjection(value string, merge bool) (plan.MergeProjection, error) {
	projection, err := plan.ParseMergeProjection(value)
	if err != nil {
		return plan.MergeProjectionSemanticUnion, fmt.Errorf("gate: -plan-projection: %w", err)
	}
	if projection == plan.MergeProjectionFirstParentTarget && !merge {
		return plan.MergeProjectionSemanticUnion, errors.New("gate: -plan-projection requires -merge")
	}
	return projection, nil
}

func gateMergeSourceStore(value string, merge bool, projection plan.MergeProjection) (string, error) {
	value = strings.TrimSpace(value)
	if projection != plan.MergeProjectionFirstParentTarget {
		if value != "" {
			return "", errors.New("gate: -merge-source-store requires -merge with first-parent-target")
		}
		return "", nil
	}
	if !merge || value == "" {
		return "", errors.New("gate: first-parent-target requires -merge-source-store")
	}
	resolved, err := filepath.Abs(filepath.Clean(value))
	if err != nil {
		return "", fmt.Errorf("gate: resolve -merge-source-store: %w", err)
	}
	if filepath.Base(resolved) != gitauthority.CanonicalOvergoDBDirectory {
		return "", errors.New("gate: -merge-source-store must name a canonical overgodb-store directory")
	}
	return resolved, nil
}

func (g *gateContext) closeCompletionStore() error {
	if g == nil || g.completionStore == nil {
		return nil
	}
	err := g.completionStore.Close()
	g.completionStore = nil
	return err
}

// checkPlanBinding refuses any commit whose -plan is not the plan's current open
// step. This is the enforcement that makes off-plan work impossible to commit:
// the shared internal/plan.Current is the same "current step" cmd/plan dispatches
// and verifies, so the gate and the dispatcher can never disagree.
func checkPlanBinding(repo, ref string) error {
	_, _, err := resolvePlanBinding(repo, gateStorePath, ref)
	return err
}

func resolvePlanBinding(repo, storePath, ref string) (plan.CompletionAuthority, string, error) {
	if ref == "" {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan <item>/<step> is required (the plan's current open step; run `go run ./cmd/plan -next`)")
	}
	item, stepID, ok := strings.Cut(ref, "/")
	if !ok || item == "" || stepID == "" {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan must be <item>/<step>, got %q", ref)
	}
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	if err := plan.Validate(document); err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	role, err := plan.AutomationRole("")
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	head, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	head = strings.TrimSpace(head)
	store, err := overgodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	defer store.Close()
	authority, err := plan.ResolveCompletionAuthority(context.Background(), repo, head, document, store)
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	it, st, open := plan.Current(document, role, authority)
	if !open {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan %s given but the plan is COMPLETE (no open step) -- nothing to commit against", ref)
	}
	if item != it.ID || stepID != st.ID {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan %s does NOT match the plan's current open step %s/%s -- commit only the dispatched step (off-plan commit REFUSED). If the plan is wrong, fix the plan first; do not commit around it", ref, it.ID, st.ID)
	}
	return authority, head, nil
}

func (g *gateContext) pipelineChecks(devicePackages ...string) []automationcheck.Check {
	generated := automationcheck.GeneratedChecks(g.repo, command)
	device := automationcheck.DeviceCheck(g.repo, g.paths, devicePackages, command)
	published := automationcheck.PublishedCheck(g.repo, command)
	checks := []automationcheck.Check{
		gateCheck("protection", runrecord.PhaseValidate, g.stepProtection), gateCheck("scope", runrecord.PhaseValidate, g.stepScope),
		gateCheck("architecture", runrecord.PhaseValidate, g.stepArchitectureRatchet),
		gateCheck("profile", runrecord.PhaseValidate, g.stepProfile), gateCheck("fmt", runrecord.PhaseValidate, g.stepFmt),
		gateCheck("style", runrecord.PhaseValidate, g.stepStyle), generated[0], generated[1], generated[2], published,
		gateCheck("docs", runrecord.PhaseValidate, g.stepDocumentation), gateCheck("magics", runrecord.PhaseValidate, g.stepMagics),
		gateCheck("modern-go", runrecord.PhaseValidate, g.stepModernGoRatchet),
		gateCheck("acceptance", runrecord.PhaseTest, g.stepAcceptance), gateCheck("vet", runrecord.PhaseVet, g.stepVet),
		gateCheck("build", runrecord.PhaseBuild, g.stepBuild), gateCheck("test", runrecord.PhaseTest, g.stepTest),
		device, gateCheck("commit", runrecord.PhasePackage, g.stepCommit),
	}
	dependencies := map[string][]string{
		"scope": {"protection"}, "architecture": {"scope"}, "profile": {"architecture"}, "fmt": {"profile"}, "style": {"fmt"},
		"manifest": {"style"}, "sbom": {"style"}, "claims": {"style"},
		"docs": {"manifest", "sbom", "claims"}, "magics": {"docs"}, "modern-go": {"magics"}, "acceptance": {"modern-go"},
		"vet": {"acceptance"}, "build": {"acceptance"}, "test": {"vet", "build"},
		"device": {"test"}, "commit": {"test", "device"},
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
	planningStarted := time.Now()
	planned, err := g.planPipeline()
	if err != nil {
		return err
	}
	definitions, checks, impact := planned.definitions, planned.invocations, planned.impact
	planningDuration := time.Since(planningStarted)
	if g.terminal == nil {
		g.terminal = map[string]automationcheck.Evidence{}
	}
	cache := g.loadRetryCache()
	cache.Compact()
	g.retryCache = &cache
	cacheable := func(name string) bool { return phaseReusesEvidence(name) }
	inputs := make(map[artifact.ID]artifact.ID, len(checks))
	for index, check := range checks {
		input, inputErr := g.phaseInputFingerprint(check.Check.Name)
		if inputErr != nil {
			return inputErr
		}
		if planned.manifest != nil {
			bound, bindErr := automationcheck.BindManifestExecution(*planned.manifest, check, []artifact.ID{input})
			if bindErr != nil {
				return bindErr
			}
			checks[index] = bound
			check = bound
		}
		inputs[check.ID] = input
	}
	satisfied := make(map[string]bool, len(impact.Exclusions))
	for _, exclusion := range impact.Exclusions {
		satisfied[exclusion.Check] = true
	}
	var cacheMutex sync.Mutex
	var terminalMutex sync.Mutex
	results, err := automationcheck.ExecuteDAG(context.Background(), checks, satisfied, func(ctx context.Context, check automationcheck.Invocation) (automationcheck.Evidence, error) {
		fmt.Fprintf(os.Stderr, gateProgressLine, check.Check.Name, runrecord.HeartbeatRunning)
		if planned.manifest != nil {
			if driftErr := g.requireCandidateTree(planned.manifest.CandidateTree); driftErr != nil {
				return automationcheck.Evidence{}, driftErr
			}
		}
		input, hasInput := inputs[check.ID]
		cacheCheck := cacheable(check.Check.Name) && hasInput
		if cacheCheck {
			cacheMutex.Lock()
			evidence, reused := cache.Lookup(check, input)
			cacheMutex.Unlock()
			if reused {
				terminalMutex.Lock()
				g.terminal[check.Check.Name] = evidence
				terminalMutex.Unlock()
				return evidence, nil
			}
		}
		evidence, runErr := automationcheck.Run(ctx, check)
		if runErr == nil && cacheCheck {
			cacheMutex.Lock()
			cache.Record(check, input, evidence)
			cacheMutex.Unlock()
		}
		if evidence.ID.Valid() {
			terminalMutex.Lock()
			g.terminal[check.Check.Name] = evidence
			terminalMutex.Unlock()
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
	cacheHits := 0
	cacheEligible := 0
	for _, result := range results {
		if cacheable(result.Invocation.Check.Name) {
			cacheEligible++
		}
		if result.Evidence.Reused {
			cacheHits++
		}
	}
	g.manifestMetrics = automationcheck.MeasureManifest(
		len(definitions), len(checks), len(impact.Exclusions), len(planned.surface.Unknown),
		cacheEligible, cacheHits, planningDuration, time.Since(g.start),
	)
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
		if planned.manifest != nil && result.Evidence.ID.Valid() &&
			!slices.Contains(automationcheck.EvidenceLineage(result.Evidence), planned.manifest.ID) {
			return fmt.Errorf("%s: terminal evidence omits manifest plan %s", name, planned.manifest.ID)
		}
		g.terminal[name] = result.Evidence
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

type plannedPipeline struct {
	definitions []automationcheck.Check
	invocations []automationcheck.Invocation
	impact      automationcheck.Impact
	surface     automationcheck.Surface
	manifest    *automationcheck.ManifestPlan
	structural  codemanifest.Impact
}

type planDisposition struct {
	Name       string                 `json:"name"`
	Phase      runrecord.Phase        `json:"phase"`
	Reason     string                 `json:"reason"`
	Matched    []automationcheck.Fact `json:"matched,omitempty"`
	RequiredBy []string               `json:"required_by,omitempty"`
}

type gatePlanReport struct {
	Kind              string                         `json:"kind"`
	Schema            int                            `json:"schema"`
	PlanID            string                         `json:"plan_id,omitzero"`
	BaseManifest      string                         `json:"base_manifest,omitzero"`
	CandidateManifest string                         `json:"candidate_manifest,omitzero"`
	CandidateSource   string                         `json:"candidate_source,omitzero"`
	CandidateTree     string                         `json:"candidate_tree,omitzero"`
	Selected          []planDisposition              `json:"selected"`
	Excluded          []planDisposition              `json:"excluded"`
	Unresolved        []planDisposition              `json:"unresolved"`
	AgentContext      *agentworkflow.ManifestContext `json:"agent_context,omitempty"`
}

func buildGatePlanReport(planned plannedPipeline) gatePlanReport {
	report := gatePlanReport{
		Kind: "overgo.gate-plan-inspection", Schema: 2,
		Selected: []planDisposition{}, Excluded: []planDisposition{},
		Unresolved: []planDisposition{},
	}
	if planned.manifest != nil {
		report.PlanID = planned.manifest.ID.String()
		report.BaseManifest = planned.manifest.BaseManifest.String()
		report.CandidateManifest = planned.manifest.CandidateManifest.String()
		report.CandidateSource = planned.manifest.CandidateSource
		report.CandidateTree = planned.manifest.CandidateTree
		if context, err := agentworkflow.NewManifestContext(
			planned.structural, *planned.manifest, nil, agentworkflow.DefaultManifestContextLimits(),
		); err == nil {
			report.AgentContext = &context
		}
	}
	definitions := make(map[string]automationcheck.Descriptor, len(planned.definitions))
	for _, check := range planned.definitions {
		definitions[check.Descriptor.Name] = check.Descriptor
	}
	excluded := make(map[string]string, len(planned.impact.Exclusions))
	for _, exclusion := range planned.impact.Exclusions {
		excluded[exclusion.Check] = exclusion.Reason
		descriptor := definitions[exclusion.Check]
		report.Excluded = append(report.Excluded, planDisposition{
			Name: exclusion.Check, Phase: descriptor.Phase, Reason: exclusion.Reason,
		})
	}
	unknownReason := "no impact producer proved this check independent"
	if len(planned.surface.Unknown) != 0 {
		unknownReason = "impact analysis unresolved: " + strings.Join(planned.surface.Unknown, "; ")
	}
	requiredBy := make(map[string][]string, len(planned.invocations))
	for _, invocation := range planned.invocations {
		for _, dependency := range invocation.Check.Dependencies {
			requiredBy[dependency] = append(requiredBy[dependency], invocation.Check.Name)
		}
	}
	for _, invocation := range planned.invocations {
		disposition := planDisposition{
			Name: invocation.Check.Name, Phase: invocation.Check.Phase,
			Matched: slices.Clone(invocation.Matched), RequiredBy: slices.Clone(requiredBy[invocation.Check.Name]),
		}
		switch {
		case invocation.Check.Always:
			disposition.Reason = "always required by check definition"
		case len(invocation.Matched) != 0:
			disposition.Reason = "matched derived impact facts"
		default:
			disposition.Reason = unknownReason
			report.Unresolved = append(report.Unresolved, disposition)
		}
		report.Selected = append(report.Selected, disposition)
	}
	slices.SortFunc(report.Excluded, func(left, right planDisposition) int { return strings.Compare(left.Name, right.Name) })
	slices.SortFunc(report.Unresolved, func(left, right planDisposition) int { return strings.Compare(left.Name, right.Name) })
	return report
}

func writeGatePlanReport(output io.Writer, planned plannedPipeline) error {
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(buildGatePlanReport(planned))
}

func (g *gateContext) planPipeline() (plannedPipeline, error) {
	tree, err := g.plannedTree()
	if err != nil {
		return plannedPipeline{}, err
	}
	var graphErr error
	var devicePackages []string
	var structural codemanifest.Impact
	var baseManifest, candidateManifest codemanifest.Manifest
	var structuralErr error
	var legacy codeprofile.FunctionImpact
	var legacyErr error
	analysisErr := g.withCandidateWorktree(tree, func(root string) error {
		original := g.repo
		g.repo = root
		defer func() { g.repo = original }()
		graph, err := g.inputGraph()
		graphErr = err
		if graphErr == nil {
			devicePackages, graphErr = graph.dependentDirectories("internal/cuda")
		}
		structural, baseManifest, candidateManifest, structuralErr = g.deriveManifestImpact()
		if structuralErr != nil {
			legacy, legacyErr = g.deriveStructuralImpact()
		}
		return nil
	})
	if analysisErr != nil {
		return plannedPipeline{}, analysisErr
	}
	// The analysis graph contains paths inside the temporary immutable
	// worktree. Keep its derived package selection, but reload a live graph for
	// later package-cache execution after that worktree is removed.
	g.packageGraph = nil
	definitions := g.pipelineChecks(devicePackages...)
	surface := automationcheck.Surface{}
	if structuralErr == nil {
		surface = automationcheck.ManifestSurface(structural)
		if requiresManifestBootstrap(g.paths) {
			surface.Unknown = append(surface.Unknown, "manifest analyzer or planner implementation changed")
			g.honesty = append(g.honesty, "manifest bootstrap: analyzer-owned change forced the complete selectable plan")
		}
	} else {
		if legacyErr == nil {
			surface = ownershipSurface(legacy)
		}
		surface.Unknown = append(surface.Unknown, "code manifest unavailable: "+structuralErr.Error())
		g.honesty = append(g.honesty, "code manifest unavailable; owned checks defaulted to run: "+structuralErr.Error())
	}
	if graphErr != nil {
		surface.Unknown = append(surface.Unknown, "package ownership: "+graphErr.Error())
		g.honesty = append(g.honesty, "package ownership unavailable; owned checks defaulted to run: "+graphErr.Error())
	}
	definitions, surface, coverage, completenessErr := automationcheck.CompleteOwnership(definitions, surface, nil)
	if completenessErr != nil {
		return plannedPipeline{}, completenessErr
	}
	if len(coverage.UncoveredPackages)+len(coverage.UncoveredSymbols) != 0 {
		g.honesty = append(g.honesty, fmt.Sprintf(
			"ownership incomplete; owned checks defaulted to run: packages=%d symbols=%d",
			len(coverage.UncoveredPackages), len(coverage.UncoveredSymbols),
		))
	}
	impact := automationcheck.OwnershipImpact(definitions, surface)
	g.selection = automationcheck.MeasureSelection(definitions, impact)
	g.selectionID = surface.Identity
	checks, err := automationcheck.Plan(definitions, impact)
	if err != nil {
		return plannedPipeline{}, err
	}
	var boundPlan *automationcheck.ManifestPlan
	if structuralErr == nil {
		candidateKey := candidateTreeKey(tree)
		if err := g.requirePreparedCandidate(candidateKey); err != nil {
			return plannedPipeline{}, err
		}
		bound, err := automationcheck.BindManifestPlan(
			baseManifest.ID, candidateManifest.ID, candidateManifest.SourceIdentity, candidateKey,
			surface, impact, checks,
		)
		if err != nil {
			return plannedPipeline{}, err
		}
		g.manifestPlan = &bound
		g.baseManifest, g.candidateManifest = &baseManifest, &candidateManifest
		boundPlan = &bound
		g.selectionID = bound.ID.String()
		g.honesty = append(g.honesty, "manifest plan: "+bound.ID.String())
	}
	return plannedPipeline{definitions: definitions, invocations: checks, impact: impact, surface: surface, manifest: boundPlan, structural: structural}, nil
}

func requiresManifestBootstrap(paths []string) bool {
	for _, name := range paths {
		name = filepath.ToSlash(name)
		for _, owner := range []string{
			"internal/codemanifest/", "internal/codeprofile/", "internal/repoanalysis/",
			"internal/automationcheck/", "cmd/code-manifest/", "cmd/gate/",
		} {
			if strings.HasPrefix(name, owner) {
				return true
			}
		}
	}
	return false
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

func (g *gateContext) deriveManifestImpact() (codemanifest.Impact, codemanifest.Manifest, codemanifest.Manifest, error) {
	candidate, err := g.sourceSnapshot()
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	base, err := sourceAtHEAD(g.repo, candidate)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	selection, err := repoanalysis.HostBuildSelection(g.repo, "./cmd/...", "./internal/...")
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	baseInputs, err := g.manifestExternalInputs(false)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	candidateInputs, err := g.manifestExternalInputs(true)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	manifestCache, err := codemanifest.NewCache(len([]repoanalysis.SourceSnapshot{base, candidate}))
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	baseManifest, _, err := manifestCache.Generate(base, []repoanalysis.BuildSelection{selection}, baseInputs)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	candidateManifest, reused, err := manifestCache.Generate(candidate, []repoanalysis.BuildSelection{selection}, candidateInputs)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	if reused {
		g.honesty = append(g.honesty, "candidate code manifest reused by exact analysis authority")
	}
	delta, err := codemanifest.Diff(baseManifest, candidateManifest)
	if err != nil {
		return codemanifest.Impact{}, codemanifest.Manifest{}, codemanifest.Manifest{}, err
	}
	impact, err := codemanifest.Close(baseManifest, candidateManifest, delta)
	if err == nil {
		g.manifestDelta, g.manifestImpact = &delta, &impact
	}
	return impact, baseManifest, candidateManifest, err
}

func (g *gateContext) manifestExternalInputs(candidate bool) ([]codemanifest.ExternalInput, error) {
	var inputs []codemanifest.ExternalInput
	for _, name := range g.paths {
		if strings.HasSuffix(name, ".go") {
			continue
		}
		var digest string
		var found bool
		var err error
		if candidate {
			digest, found, err = worktreeFileDigest(g.repo, name)
		} else {
			digest, found, err = revisionFileDigest(g.repo, "HEAD", name)
		}
		if err != nil {
			return nil, err
		}
		if found {
			inputs = append(inputs, codemanifest.ExternalInput{
				Path: name, ContentID: digest, Kind: "repository-file", Owner: path.Dir(name),
			})
		}
	}
	return inputs, nil
}

func worktreeFileDigest(root, relative string) (string, bool, error) {
	file, err := os.Open(filepath.Join(root, filepath.FromSlash(relative)))
	if errors.Is(err, os.ErrNotExist) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", false, err
	}
	return hex.EncodeToString(hasher.Sum(nil)), true, nil
}

func revisionFileDigest(root, revision, relative string) (string, bool, error) {
	object := revision + ":" + filepath.ToSlash(relative)
	probe := newGateGitReaderCommand(root, "cat-file", "-e", object)
	if err := probe.Run(); err != nil {
		if _, missing := err.(*exec.ExitError); missing {
			return "", false, nil
		}
		return "", false, err
	}
	hasher := sha256.New()
	show := newGateGitReaderCommand(root, "show", object)
	show.Stdout = hasher
	if err := show.Run(); err != nil {
		return "", false, err
	}
	return hex.EncodeToString(hasher.Sum(nil)), true, nil
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
	case evidence.Inapplicable:
		record.Outcome = runrecord.StepInapplicable
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
	store, err := overgodb.OpenReadOnly(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return false, err
	}
	_, found, readErr := closurescan.ReadCensusEvidence(context.Background(), store, *document.Census)
	closeErr := store.Close()
	if readErr != nil || closeErr != nil {
		return false, errors.Join(readErr, closeErr)
	}
	if !found {
		return false, errors.New("gate: campaign census evidence is absent from OvergoDB")
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
	if stray, err := strayRootExecutables(g.repo); err != nil {
		return false, err
	} else if len(stray) > 0 {
		return false, fmt.Errorf(
			"gate: executables outside bin at the repository root: %s -- a single-package `go build ./cmd/x` drops its binary at the working directory; build with -o bin/<name>.exe or delete the stray",
			strings.Join(stray, ", "),
		)
	}
	g.stepEvidence["protection"] = configured + ";activation=" + activated
	g.honesty = append(g.honesty, "protection: "+g.stepEvidence["protection"])
	return false, nil
}

// strayRootExecutables lists .exe files at the repository root: the
// gitignore hides them from status and only bin/ holds executables, so
// a root binary is always an accident this gate surfaces at commit
// time instead of leaving it for the release check.
func strayRootExecutables(repo string) ([]string, error) {
	entries, err := os.ReadDir(repo)
	if err != nil {
		return nil, err
	}
	var stray []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.EqualFold(filepath.Ext(entry.Name()), ".exe") {
			stray = append(stray, entry.Name())
		}
	}
	return stray, nil
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

// stepArchitectureRatchet audits the complete candidate source on every gate
// run, including documentation-only changes. It is deliberately in-process:
// the manifest already binds this check to the candidate tree, and a second
// test process would only repeat source discovery while weakening ordering.
func (g *gateContext) stepArchitectureRatchet() (bool, error) {
	if _, err := g.stepArchitecture(); err != nil {
		return false, err
	}
	staged, err := codeprofile.LoadStagedSurface(filepath.Join(g.repo, "docs", "staged_surface.json"))
	if err != nil {
		return false, err
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	report, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		return false, err
	}
	if err := runrecord.ValidateTriggerRegistry(); err != nil {
		return false, err
	}
	// The staged-surface declaration is gate authority even when no Go file
	// changed: every deferred export must still resolve to an open canonical
	// step or an evidence-bound retained classification.
	report.Rules += len(staged.Staged) + 1
	report.Sites += len(staged.Staged)
	g.honesty = append(g.honesty, fmt.Sprintf(
		"architecture ratchet: source=%s paths=%d rules=%d sites=%d findings=%d",
		report.SourceIdentity, len(g.paths), report.Rules, report.Sites, len(report.Findings),
	))
	return false, report.Error()
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
	baseline, ratcheted, err := codeprofile.LoadCloneBaseline(filepath.Join(g.repo, filepath.FromSlash(codeprofile.CloneBaselineFile)))
	if err != nil {
		return false, err
	}
	if ratcheted {
		if err := codeprofile.AdmitCloneBaseline(baseline, profile.DuplicateExcessNodes); err != nil {
			return false, err
		}
		g.honesty = append(g.honesty, fmt.Sprintf(
			"clone ratchet: duplicate_excess=%d ceiling=%d headroom=%d",
			profile.DuplicateExcessNodes, baseline.DuplicateExcessNodes,
			baseline.DuplicateExcessNodes-profile.DuplicateExcessNodes,
		))
	}
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
		// docs/staged_surface.json is the reviewed acceptance for new
		// surface whose consumer is deliberately deferred (owner ruling
		// 2026-08-27: valuable new elements land declared, not blocked);
		// undeclared new surface still refuses.
		staged, err := codeprofile.LoadStagedSurface(filepath.Join(g.repo, "docs", "staged_surface.json"))
		if err != nil {
			return err
		}
		accepted, blocking := codeprofile.PartitionStagedSurface(unconsumed, staged)
		if len(accepted) > 0 {
			g.honesty = append(g.honesty, fmt.Sprintf(
				"staged surface accepted per docs/staged_surface.json: %s", consumerCandidates(accepted)))
		}
		if len(blocking) > 0 {
			return fmt.Errorf("new unconsumed production surface: %s", consumerCandidates(blocking))
		}
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
	slices.Sort(candidates)
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
		cmd := newGateGitReaderCommand(repo, "show", revision+":"+path)
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

// treeStateKey identifies the exact full Git tree produced by overlaying only
// planned paths onto HEAD. Git's tree encoding provides the path/type/length
// framing; the outer domain-separated SHA-256 remains the durable gate key.
func (g *gateContext) treeStateKey() (string, error) {
	tree, err := g.plannedTree()
	if err != nil {
		return "", err
	}
	return candidateTreeKey(tree), nil
}

func candidateTreeKey(tree string) string {
	digest := sha256.Sum256([]byte("overgo-candidate-tree/v1\x00" + tree))
	return hex.EncodeToString(digest[:])
}

func (g *gateContext) requirePreparedCandidate(candidateKey string) error {
	if g.preparation.TreeKey != "" && g.preparation.TreeKey != candidateKey {
		return errors.New("gate: candidate tree changed after durable preparation")
	}
	return nil
}

// plannedTree snapshots the commit candidate into an isolated temporary index.
// It never reads unplanned worktree paths and never mutates the caller's index.
func (g *gateContext) plannedTree() (string, error) {
	temporary, err := os.MkdirTemp("", "overgo-gate-index-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(temporary)
	environment := gitIndexEnvironment(filepath.Join(temporary, "index"))
	if _, err := gitWriterCommandEnvironment(g.repo, environment, "read-tree", "HEAD"); err != nil {
		return "", err
	}
	for _, chunk := range chunkByArgBudget(g.paths) {
		arguments := append([]string{"--literal-pathspecs", "add", "-A", "--"}, chunk...)
		if _, err := gitWriterCommandEnvironment(g.repo, environment, arguments...); err != nil {
			return "", err
		}
	}
	tree, err := gitWriterCommandEnvironment(g.repo, environment, "write-tree")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(tree), nil
}

func (g *gateContext) requireCandidateTree(expected string) error {
	current, err := g.treeStateKey()
	if err != nil {
		return err
	}
	if current != expected {
		return fmt.Errorf("gate: candidate drifted after manifest planning: planned=%s current=%s", expected, current)
	}
	return nil
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
		_ = writeJSON(g.repo, gateRetryFile, cache, clioptions.OutputFileMode)
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
	slices.Sort(selected)
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
	documentation := strings.HasSuffix(path, ".md") || strings.HasPrefix(path, "docs/") ||
		path == "compatibility.json" || path == "SBOM.cdx.json"
	switch phase {
	case "vet", "build", "fmt", "style", "profile", "architecture":
		return goInput
	case "scope", "protection":
		return goInput || strings.HasPrefix(path, "scripts/") || strings.HasPrefix(path, protection.HarnessConfigDirectory)
	case "manifest", "sbom", "claims", "docs":
		return goInput || documentation
	case "test":
		return goInput || path == "README.md" || strings.HasPrefix(path, "docs/") && path != plan.Path
	default:
		return false
	}
}

// phaseReusesEvidence reports whether a check's terminal evidence may be
// reused across consecutive gate attempts when its exact manifest-bound input
// fingerprint is byte-identical. Store-coupled checks (magics, acceptance,
// published) stay excluded because every attempt's preparation commit moves
// the store their verdicts read; device stays excluded because hardware is
// not a fingerprintable input; the commit check is never reused.
func phaseReusesEvidence(phase string) bool {
	switch phase {
	case "vet", "build", "fmt", "style", "profile", "architecture", "scope", "protection",
		"manifest", "sbom", "claims", "docs", "test":
		return true
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
		files, err := gitLines(g.repo, "--literal-pathspecs", "ls-files", "-co", "--exclude-standard", "--", p)
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

func validatePlannedPaths(paths []string) error {
	if len(paths) == 0 {
		return errors.New("gate: planned paths are empty")
	}
	seen := make(map[string]bool, len(paths))
	for _, candidate := range paths {
		trimmed := strings.TrimSuffix(candidate, "/")
		if trimmed == "" || trimmed == "." || path.Clean(trimmed) != trimmed || filepath.IsAbs(trimmed) ||
			strings.HasPrefix(trimmed, "../") || strings.HasPrefix(trimmed, ":") ||
			strings.ContainsAny(trimmed, "*?[\x00\r\n") {
			return fmt.Errorf("gate: planned path is not one canonical literal repository path: %q", candidate)
		}
		if seen[trimmed] {
			return fmt.Errorf("gate: planned path is duplicated: %q", candidate)
		}
		seen[trimmed] = true
	}
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
	var verificationInputs []string
	for _, path := range unplanned {
		g.honesty = append(g.honesty, "unplanned dirty (not shipped): "+path)
		g.profileDirty = g.profileDirty || strings.HasSuffix(path, ".go")
		if unplannedVerificationInput(path) {
			verificationInputs = append(verificationInputs, path)
		}
	}
	if len(verificationInputs) != 0 {
		return false, fmt.Errorf(
			"unplanned verification input could affect evidence without entering the commit: %s",
			strings.Join(verificationInputs, ", "),
		)
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

func unplannedVerificationInput(path string) bool {
	return path == "go.mod" || path == "go.sum" || strings.HasSuffix(path, ".go") ||
		(strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "cmd/") || strings.HasPrefix(path, "kernels/")) &&
			!strings.HasSuffix(path, ".md")
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

// argBudget bounds the cumulative path-argument bytes per spawned
// process: Windows caps the CreateProcess command line at 32767
// characters, and a repo-wide commit can plan more paths than one
// invocation carries; 28000 leaves headroom for the executable path
// and the leading flags.
const argBudget = 28_000

// chunkByArgBudget splits files into runs whose joined length fits one
// command line under argBudget; every run is non-empty.
func chunkByArgBudget(files []string) [][]string {
	var chunks [][]string
	for start := 0; start < len(files); {
		end, used := start, 0
		for end < len(files) && (end == start || used+len(files[end])+1 <= argBudget) {
			used += len(files[end]) + 1
			end++
		}
		chunks = append(chunks, files[start:end])
		start = end
	}
	return chunks
}

func (g *gateContext) stepFmt() (bool, error) {
	files := g.changedGoFiles()
	if len(files) == 0 {
		return true, nil
	}
	var unformatted []string
	for _, chunk := range chunkByArgBudget(files) {
		out, err := command(g.repo, "gofmt", append([]string{"-l"}, chunk...)...)
		if err != nil {
			return false, err
		}
		if s := strings.TrimSpace(out); s != "" {
			unformatted = append(unformatted, s)
		}
	}
	if len(unformatted) > 0 {
		files := strings.Fields(strings.Join(unformatted, "\n"))
		return false, fmt.Errorf(
			"unformatted: %s; remediate with `gofmt -w %s`",
			strings.Join(files, " "), strings.Join(files, " "),
		)
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

func (g *gateContext) stepModernGoRatchet() (bool, error) {
	baseline, err := repoanalysis.LoadModernGoBaseline(filepath.Join(g.repo, filepath.FromSlash(repoanalysis.ModernGoBaselineFile)))
	if err != nil {
		return false, err
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	selection, err := repoanalysis.HostBuildSelection(g.repo, "./cmd/...", "./internal/...")
	if err != nil {
		return false, err
	}
	candidate, err := repoanalysis.ModernGoCensusSnapshot(snapshot, selection, baseline.TargetGo)
	if err != nil {
		return false, err
	}
	if err := repoanalysis.AdmitModernGoRatchet(baseline, candidate, time.Now().UTC()); err != nil {
		return false, err
	}
	if _, err := command(g.repo, "git", "cat-file", "-e", "HEAD:"+repoanalysis.ModernGoBaselineFile); err == nil {
		if g.baseSource == nil {
			base, err := sourceAtHEAD(g.repo, snapshot)
			if err != nil {
				return false, err
			}
			g.baseSource = &base
		}
		previous, err := repoanalysis.ModernGoCensusSnapshot(*g.baseSource, selection, baseline.TargetGo)
		if err != nil {
			return false, err
		}
		if err := repoanalysis.AdmitModernGoDelta(baseline, previous, candidate, g.paths); err != nil {
			return false, err
		}
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"modern-Go ratchet: findings=%d candidates=%d inspected=%d typed=%d catalog=%s",
		len(candidate.Findings), candidate.CandidateCount(), baseline.Coverage.InspectedFiles,
		baseline.Coverage.TypedFiles, baseline.CatalogCommit,
	))
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
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	boundaryCoverage, err := automationcheck.AgentHarnessBoundaryCoverage(snapshot, g.paths)
	if err != nil {
		return false, err
	}
	selectedTests := append(slices.Clone(direct), dependent...)
	if err := automationcheck.RequireAgentHarnessBoundaries(boundaryCoverage, selectedTests); err != nil {
		return false, err
	}
	if len(boundaryCoverage.Boundaries) != 0 {
		g.honesty = append(g.honesty, "assembled agent boundaries: "+strings.Join(boundaryCoverage.Boundaries, ","))
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
	documents, aliases, err := activeMagicBindings(g.repo, g.storePath)
	if err != nil {
		return false, err
	}
	report, err := closurescan.ValidatePermanentActiveAuthority(snapshot, documents, aliases)
	if err != nil && staleClosureAuthorityFailure(err) {
		// The safe deterministic remediation: rebind unambiguous history to
		// current offsets in the same store, then revalidate exactly once.
		// Admitted content never changes; only stale alias offsets move.
		if remediationErr := g.remediateStaleClosureBindings(); remediationErr != nil {
			return false, errors.Join(err, remediationErr)
		}
		documents, aliases, err = activeMagicBindings(g.repo, g.storePath)
		if err != nil {
			return false, err
		}
		report, err = closurescan.ValidatePermanentActiveAuthority(snapshot, documents, aliases)
	}
	if err != nil {
		return false, fmt.Errorf(
			"%w; remediate with `go run ./cmd/closure-scan -import-store %s` and catalog what remains, then re-run the gate",
			err, gateStorePath,
		)
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"permanent magic authority: production=%d classified=%d tests=%d open=0 stale=0 policy_copies=0",
		report.ProductionSites, report.ClassifiedSites, report.TestSites,
	))
	return false, nil
}

// staleClosureAuthorityFailure classifies magic-authority refusals whose fix
// is the deterministic same-store rebind: catalogued sites whose byte offsets
// moved under the active aliases, either reported stale directly or surfacing
// as an uncatalogued site that exact prior history still covers.
func staleClosureAuthorityFailure(err error) bool {
	message := err.Error()
	return strings.Contains(message, "stale active binding") ||
		strings.Contains(message, "uncatalogued production policy")
}

func (g *gateContext) remediateStaleClosureBindings() error {
	started := time.Now()
	out, err := g.runGateCommand("go", "run", "./cmd/closure-scan", "-import-store", gateStorePath)
	if err != nil {
		return fmt.Errorf("gate: closure rebind remediation: %w", err)
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"remediation: closure rebind applied (%s) wall=%dms",
		strings.TrimSpace(out), time.Since(started).Milliseconds(),
	))
	return nil
}

func activeMagicBindings(repo, storePath string) ([]closureledger.Document, map[string]artifact.ID, error) {
	store, err := overgodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return nil, nil, err
	}
	defer store.Close()
	var documents []closureledger.Document
	aliases := map[string]artifact.ID{}
	_, err = overgodb.VisitDecodedDocuments(context.Background(), store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: closureledger.MediaType, Schema: closureledger.Schema,
		}}, AliasPrefixes: []string{closureledger.ActiveAliasPrefix}, Order: overgodb.DocumentOldestFirst,
	}, closureledger.Parse, func(view overgodb.DocumentView, document closureledger.Document) error {
		if document.ID != view.Content.Descriptor.ID {
			return errors.New("magic scan: active document identity mismatch")
		}
		documents = append(documents, document)
		for _, alias := range view.Aliases {
			aliases[alias] = document.ID
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return documents, aliases, nil
}

// stepArchitecture is the permanent entry-authority ratchet. Unlike magics it
// never skips: a documentation-only commit still validates the complete
// candidate source inventory, so no commit shape can land a bypass around the
// established storage, process, capability, tool, trigger, and promotion
// entry authorities. It reuses the gate's cached source snapshot in-process
// and launches no second verification process; the check retires only when
// the authorities it guards disappear.
func (g *gateContext) stepArchitecture() (bool, error) {
	started := time.Now()
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return false, err
	}
	required := []closurescan.EntryAuthorityDomain{
		closurescan.EntryAuthorityStorage, closurescan.EntryAuthorityProcess,
		closurescan.EntryAuthorityCapability, closurescan.EntryAuthorityTool,
		closurescan.EntryAuthorityTrigger, closurescan.EntryAuthorityPromotion,
		closurescan.EntryAuthorityGoOnly,
	}
	rules := make([]closurescan.EntryAuthorityRule, 0, len(required))
	for _, domain := range required {
		rule, found := closurescan.EntryAuthorityRuleFor(domain)
		if !found {
			return false, fmt.Errorf("architecture: entry authority table lost the %s domain", domain)
		}
		rules = append(rules, rule)
	}
	report, err := closurescan.ValidateEntryAuthorities(snapshot, rules)
	if err != nil {
		return false, fmt.Errorf("architecture: %w", err)
	}
	domains := make([]string, 0, len(report.Domains))
	for _, domain := range report.Domains {
		domains = append(domains, string(domain.Domain))
	}
	g.honesty = append(g.honesty, fmt.Sprintf(
		"entry authority ratchet: domains=%s wall=%dms",
		strings.Join(domains, ","), time.Since(started).Milliseconds(),
	))
	return false, nil
}

func (g *gateContext) stepAcceptance() (bool, error) {
	contract, err := completionAcceptanceContract(g.repo, g.planRef, g.completionAuthority)
	if err != nil {
		return false, err
	}
	tree, err := g.plannedTree()
	if err != nil {
		return false, err
	}
	if g.manifestPlan == nil || candidateTreeKey(tree) != g.manifestPlan.CandidateTree {
		return false, errors.New("acceptance: immutable candidate tree differs from the manifest plan")
	}
	if err := g.requirePreparedCandidate(g.manifestPlan.CandidateTree); err != nil {
		return false, err
	}
	verdict, err := g.executeCandidateVerifier(tree, contract.verify)
	if err != nil {
		return false, err
	}
	if verdict != testevidence.ClassifyVerifyCommand(contract.verify) {
		return false, errors.New("acceptance: verifier returned the wrong classifier verdict")
	}
	g.acceptedTree = tree
	g.stepEvidence["acceptance"] = contract.evidence
	return false, nil
}

type acceptanceContract struct {
	evidence string
	verify   string
}

func completionAcceptanceContract(repo, reference string, completions plan.CompletionAuthority) (acceptanceContract, error) {
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return acceptanceContract{}, err
	}
	return completionAcceptanceContractForPlan(document, reference, completions)
}

func completionAcceptanceEvidenceForPlan(
	document plan.Plan,
	reference string,
	completions plan.CompletionAuthority,
) (string, error) {
	contract, err := completionAcceptanceContractForPlan(document, reference, completions)
	return contract.evidence, err
}

func completionAcceptanceContractForPlan(
	document plan.Plan,
	reference string,
	completions plan.CompletionAuthority,
) (acceptanceContract, error) {
	role, roleErr := plan.AutomationRole("")
	if roleErr != nil {
		return acceptanceContract{}, roleErr
	}
	item, step, open := plan.Current(document, role, completions)
	if !open {
		return acceptanceContract{}, errors.New("acceptance: plan has no current open step")
	}
	if current := item.ID + "/" + step.ID; current != reference {
		return acceptanceContract{}, fmt.Errorf("acceptance: current plan step %s differs from gate authority %s", current, reference)
	}
	evidence, err := runrecord.FormatCompletionAcceptanceEvidence(
		testevidence.CurrentVerifyPolicy, reference, step.Verify,
	)
	if err != nil {
		return acceptanceContract{}, err
	}
	return acceptanceContract{evidence: evidence, verify: step.Verify}, nil
}

func (g *gateContext) executeCandidateVerifier(
	tree, verify string,
) (verdict testevidence.VerdictClass, err error) {
	err = g.withCandidateWorktree(tree, func(worktree string) error {
		var executeErr error
		verdict, executeErr = planverify.Execute(context.Background(), worktree, verify)
		if executeErr != nil {
			return executeErr
		}
		return requireCandidateWorktreeUnchanged(worktree, tree)
	})
	return verdict, err
}

func requireCandidateWorktreeUnchanged(worktree, tree string) error {
	for _, arguments := range [][]string{
		{"diff", "--name-only", "-z", "--"},
		{"ls-files", "--others", "--exclude-standard", "-z"},
		{"ls-files", "--others", "--ignored", "--exclude-standard", "-z"},
	} {
		changed, err := command(worktree, "git", arguments...)
		if err != nil {
			return err
		}
		if changed != "" {
			return errors.New("acceptance: verifier mutated its immutable candidate worktree")
		}
	}
	indexTree, err := gitWriterCommand(worktree, "write-tree")
	if err != nil {
		return err
	}
	if strings.TrimSpace(indexTree) != tree {
		return errors.New("acceptance: verifier changed the accepted candidate index")
	}
	return nil
}

func (g *gateContext) withCandidateWorktree(tree string, use func(string) error) (err error) {
	if !validGitObjectID(tree) || use == nil {
		return errors.New("gate: candidate worktree requires an exact tree and callback")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-acceptance-*")
	if err != nil {
		return err
	}
	worktree := filepath.Join(temporary, "candidate")
	added := false
	defer func() {
		if added {
			_, cleanupErr := gitWriterCommand(g.repo, "worktree", "remove", "--force", worktree)
			err = errors.Join(err, cleanupErr)
		}
		err = errors.Join(err, os.RemoveAll(temporary))
	}()
	if _, err := gitWriterCommand(
		g.repo, "worktree", "add", "--quiet", "--detach", "--no-checkout", worktree, "HEAD",
	); err != nil {
		return err
	}
	added = true
	if _, err := gitWriterCommand(worktree, "read-tree", tree); err != nil {
		return err
	}
	if _, err := gitWriterCommand(worktree, "checkout-index", "--all", "--force"); err != nil {
		return err
	}
	return use(worktree)
}

func (g *gateContext) stepCommit() (bool, error) {
	if g.manifestPlan == nil {
		return false, errors.New("commit admission: manifest plan is absent")
	}
	if g.acceptedTree == "" || candidateTreeKey(g.acceptedTree) != g.manifestPlan.CandidateTree {
		return false, errors.New("commit admission: accepted immutable candidate tree is absent")
	}
	if err := g.requirePreparedCandidate(g.manifestPlan.CandidateTree); err != nil {
		return false, err
	}
	if err := g.requireCandidateTree(g.manifestPlan.CandidateTree); err != nil {
		return false, err
	}
	currentTree, err := g.plannedTree()
	if err != nil {
		return false, err
	}
	if currentTree != g.acceptedTree {
		return false, errors.New("commit admission: planned content changed after acceptance")
	}
	if err := validateManifestCommitAdmission(*g.manifestPlan, g.terminal); err != nil {
		return false, err
	}
	// Validate the immutable result shape before Git advances. A schema error
	// discovered after commit cannot be represented by the normal debt batch.
	recipeID, err := g.gateRecipeID()
	if err != nil {
		return false, err
	}
	validationDuration, err := completedGateMeasurement(g.start, "pre-commit record validation")
	if err != nil {
		return false, err
	}
	steps := append(slices.Clone(g.steps), runrecord.GateStep{
		Name: "commit", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: validationDuration,
	})
	if _, err := runrecord.NewGateRecord(
		recipeID, g.environment.ID, strings.Repeat("0", hex.EncodedLen(sha1.Size)), runrecord.OutcomeSucceeded, "", validationDuration, steps,
	); err != nil {
		return false, fmt.Errorf("pre-commit record validation: %w", err)
	}
	planFile := filepath.Join(g.repo, filepath.FromSlash(plan.Path))
	planBefore, err := acceptedPlanBytes(g.repo, g.acceptedTree)
	if err != nil {
		return false, err
	}
	worktreePlan, err := os.ReadFile(planFile)
	if err != nil {
		return false, err
	}
	if !bytes.Equal(worktreePlan, planBefore) {
		return false, errors.New("commit admission: worktree plan differs from the accepted candidate tree")
	}
	document, err := plan.Parse(planBefore)
	if err != nil {
		return false, err
	}
	acceptance, err := completionAcceptanceEvidenceForPlan(document, g.planRef, g.completionAuthority)
	if err != nil {
		return false, fmt.Errorf("commit admission: plan changed after acceptance: %w", err)
	}
	if acceptance != g.stepEvidence["acceptance"] {
		return false, errors.New("commit admission: plan verifier changed after acceptance")
	}
	planHead, err := command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return false, err
	}
	planHead = strings.TrimSpace(planHead)
	if planHead != g.planHead {
		return false, fmt.Errorf("commit admission: plan authority moved from %.12s to %.12s", g.planHead, planHead)
	}
	completionStore, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return false, fmt.Errorf("commit admission: lock completion store: %w", err)
	}
	if err := requireSoleCurrentGatePreparation(
		context.Background(), completionStore, g.preparation, g.preparationCommit,
	); err != nil {
		_ = completionStore.Close()
		return false, fmt.Errorf("commit admission: recheck gate preparation authority: %w", err)
	}
	completionAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), g.repo, g.planHead, document, completionStore,
	)
	if err != nil {
		_ = completionStore.Close()
		return false, fmt.Errorf("commit admission: recheck completion authority: %w", err)
	}
	role, err := plan.AutomationRole("")
	if err != nil {
		_ = completionStore.Close()
		return false, err
	}
	item, step, open := plan.Current(document, role, completionAuthority)
	if !open || item.ID+"/"+step.ID != g.planRef {
		_ = completionStore.Close()
		return false, errors.New("commit admission: completion authority no longer selects the gated row")
	}
	planHead, err = command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		_ = completionStore.Close()
		return false, err
	}
	if planHead = strings.TrimSpace(planHead); planHead != g.planHead {
		_ = completionStore.Close()
		return false, fmt.Errorf("commit admission: plan authority moved from %.12s to %.12s", g.planHead, planHead)
	}
	g.completionAuthority = completionAuthority
	g.completionStore = completionStore
	planAfter, err := advancedPlanBytes(planBefore, g.planRef)
	if err != nil {
		return false, err
	}
	childDocument, err := plan.Parse(planAfter)
	if err != nil {
		return false, err
	}
	mergeAuthority, err := g.deriveProjectedMergeAuthority(document, childDocument, completionStore)
	if err != nil {
		return false, fmt.Errorf("commit admission: derive projected merge authority: %w", err)
	}
	g.mergeAuthority = mergeAuthority
	// Structured completion trailers make Git the completion record:
	// the row leaves the plan in this same commit, and the trailers
	// carry what completed and how it was verified. The gate record
	// published after commit binds the evidence by commit SHA.
	messageFile, err := g.completionMessageFile(document)
	if err != nil {
		return false, err
	}
	defer os.Remove(messageFile)
	planInfo, err := os.Stat(planFile)
	if err != nil {
		return false, err
	}
	headReference, err := currentHeadReference(g.repo)
	if err != nil {
		return false, err
	}
	if headReference == "HEAD" {
		return false, errors.New("commit admission: detached HEAD cannot provide branch-reference authority")
	}
	if err := g.requireGateStartState(); err != nil {
		return false, err
	}
	indexBefore, mergeState := g.indexBefore, g.mergeBefore
	indexAfter, err := buildAcceptedCompletionIndex(g.repo, g.acceptedTree, planAfter)
	if err != nil {
		return false, err
	}
	indexRestore, err := buildGateIndexForTree(g.repo, indexBefore.Tree)
	if err != nil {
		return false, err
	}
	intent := gateCommitIntent{
		Version: artifact.InitialDocumentVersion, Preparation: g.preparation.ID,
		PreparationCommit: g.preparationCommit, Recipe: recipeID,
		CandidateManifest: g.candidateManifest.ID,
		Parent:            g.planHead, HeadReference: headReference, Merge: mergeState,
		PlanProjection: g.planProjection,
		IndexTree:      indexBefore.Tree, Tree: indexAfter.Tree,
		IndexBefore: indexBefore.Data, IndexAfter: indexAfter.Data, IndexRestore: indexRestore.Data,
		IndexMode: uint32(indexBefore.Mode.Perm()),
		PlanRef:   g.planRef, Paths: slices.Clone(g.paths),
		Plan: planBefore, AdvancedPlan: planAfter, PlanMode: uint32(planInfo.Mode().Perm()),
	}
	if mergeAuthority != nil {
		content, contentErr := mergeAuthority.Content()
		if contentErr != nil {
			return false, contentErr
		}
		intent.MergeAuthority = content.Data
	}
	completionMessage, err := os.ReadFile(messageFile)
	if err != nil {
		return false, err
	}
	messageItem, messageStep, found := strings.Cut(g.planRef, "/")
	if !found {
		return false, errors.New("commit admission: invalid completion reference")
	}
	mergeAuthorityID := artifact.ID{}
	if mergeAuthority != nil {
		mergeAuthorityID = mergeAuthority.ID
	}
	if err := plan.VerifyCompletionCommitMessageWithMergeAuthority(
		string(completionMessage), document, messageItem, messageStep, recipeID,
		g.candidateManifest.ID, g.preparation.ID, g.preparationCommit,
		g.planProjection, mergeAuthorityID,
	); err != nil {
		return false, fmt.Errorf("commit admission: verify completion message: %w", err)
	}
	if err := verifyProspectiveGateCompletion(
		g.repo, intent, document, childDocument, string(completionMessage), g.completionStore,
	); err != nil {
		return false, fmt.Errorf("commit admission: prospective completion transition: %w", err)
	}
	// Construct the exact completion commit while every tree is still private
	// or referenced by the caller's captured state. The first durable intent can
	// then bind one keepalive commit that reaches both rollback trees and this
	// otherwise-unreferenced completion commit before any shared mutation.
	intent.Commit, err = createGateCommit(g.repo, intent, messageFile)
	if err != nil {
		return false, err
	}
	if err := initializeGateIntentKeepalive(g.repo, &intent); err != nil {
		return false, err
	}
	if err := writeGateCommitIntent(g.repo, intent); err != nil {
		return false, err
	}
	referenceAdvanced, planPublished := false, false
	var rollbackPlan func() error
	defer func() {
		if referenceAdvanced || g.commitInterrupted {
			return
		}
		// A failed pre-CAS attempt may have published the write-ahead plan or
		// installed only the exact final index recorded by the intent. Restore
		// both only while this worktree still names the captured parent and each
		// shared file remains in one of its exact write-ahead states. Any other
		// state belongs to a concurrent operation and must survive for recovery.
		if !exactGateHead(g.repo, intent.HeadReference, intent.Parent) {
			g.commitInterrupted = true
			return
		}
		if planPublished {
			if err := rollbackPlan(); err != nil {
				g.commitInterrupted = true
				return
			}
			planPublished = false
		}
		if err := restoreCapturedIndex(g.repo, intent); err != nil {
			g.commitInterrupted = true
			return
		}
		matches, err := planBytesMatch(g.repo, intent.Plan)
		if err != nil || !matches {
			g.commitInterrupted = true
			return
		}
		if err := removeGateCommitIntent(g.repo); err != nil {
			g.commitInterrupted = true
		}
	}()
	// Publish the plan before the prepared branch transaction. A process death
	// can therefore leave a durable intent with the parent and either exact plan
	// state, but never the completion branch pointing at a plan that publication
	// has not yet installed.
	rollbackPlan, err = publishPlanTransition(
		g.repo, intent.Preparation, planBefore, planAfter, planInfo.Mode().Perm(),
	)
	if err != nil {
		return false, err
	}
	planPublished = true
	// The final tree was built in an isolated index and made durable in the
	// intent before this shared-index transition. Install it only from the exact
	// captured index so a concurrent staged change is never silently replaced.
	if err := installCompletionIndexAndAdvanceGateReference(g.repo, intent); err != nil {
		if !gateReferenceEquals(g.repo, intent.HeadReference, intent.Parent) {
			g.commitInterrupted = true
		}
		return false, err
	}
	referenceAdvanced = true
	planPublished = false
	g.commitInterrupted = true
	writtenPlan, err := os.ReadFile(planFile)
	if err != nil {
		return false, err
	}
	if !bytes.Equal(writtenPlan, planAfter) {
		return false, errors.New("commit admission: advanced plan bytes differ from the write-ahead transition")
	}
	if err := clearCommittedMergeState(g.repo, intent); err != nil {
		return false, err
	}
	if err := validateInterruptedCommitAuthority(g.repo, intent, g.completionStore); err != nil {
		return false, fmt.Errorf("commit authority validation: %w", err)
	}
	if !exactGateHead(g.repo, intent.HeadReference, intent.Commit) {
		return false, errors.New("commit authority validation: captured branch moved after exact ref update")
	}
	g.committedHead = intent.Commit
	g.commitInterrupted = false
	return false, nil
}

func treePathMode(repo, tree, repositoryPath string) (string, error) {
	entry, err := command(repo, "git", "ls-tree", tree, "--", repositoryPath)
	if err != nil {
		return "", err
	}
	metadata, _, found := strings.Cut(entry, "\t")
	mode, objectMetadata, foundMode := strings.Cut(metadata, " ")
	objectType, objectID, foundObject := strings.Cut(objectMetadata, " ")
	if !found || !foundMode || !foundObject || mode == "" || objectType != "blob" ||
		!validGitObjectID(strings.TrimSpace(objectID)) {
		return "", fmt.Errorf("commit admission: %s is absent from accepted tree %s", repositoryPath, tree)
	}
	return mode, nil
}

func acceptedPlanBytes(repo, tree string) ([]byte, error) {
	if !validGitObjectID(tree) {
		return nil, errors.New("commit admission: accepted tree identity is invalid")
	}
	content, err := command(repo, "git", "show", tree+":"+plan.Path)
	if err != nil {
		return nil, err
	}
	return []byte(content), nil
}

func gitHashObject(repo string, content []byte) (string, error) {
	process := newGateGitWriterCommand(repo, nil, "hash-object", "-w", "--stdin")
	process.Stdin = bytes.NewReader(content)
	output, err := process.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf(
			"git hash-object -w --stdin: %v: %s",
			err,
			clioptions.Tail(string(output), clioptions.DiagnosticTailBytes),
		)
	}
	object := strings.TrimSpace(string(output))
	if !validGitObjectID(object) {
		return "", errors.New("git hash-object returned an invalid object identity")
	}
	return object, nil
}

type gateIndexSnapshot struct {
	Tree string
	Data []byte
	Mode fs.FileMode
}

// buildAcceptedCompletionIndex constructs the exact commit index in private.
// The caller's shared index is not touched until the completed index and its
// exact original bytes are both durable in the commit intent.
func buildAcceptedCompletionIndex(repo, acceptedTree string, planAfter []byte) (gateIndexSnapshot, error) {
	if !validGitObjectID(acceptedTree) {
		return gateIndexSnapshot{}, errors.New("commit admission: accepted tree identity is invalid")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-commit-index-*")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	defer os.RemoveAll(temporary)
	indexPath := filepath.Join(temporary, "index")
	environment := gitIndexEnvironment(indexPath)
	if _, err := gitWriterCommandEnvironment(repo, environment, "read-tree", acceptedTree); err != nil {
		return gateIndexSnapshot{}, err
	}
	planMode, err := treePathMode(repo, acceptedTree, plan.Path)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	planBlob, err := gitHashObject(repo, planAfter)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	if _, err := gitWriterCommandEnvironment(
		repo, environment, "update-index", "--add", "--cacheinfo", planMode+","+planBlob+","+plan.Path,
	); err != nil {
		return gateIndexSnapshot{}, err
	}
	tree, err := gitWriterCommandEnvironment(repo, environment, "write-tree")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	return gateIndexSnapshot{Tree: strings.TrimSpace(tree), Data: data}, nil
}

// buildGateIndexForTree precomputes the exact recovery target. Reusing captured
// index bytes after renaming them would retain stale entry stat caches under a
// new index-file timestamp and can make racily-clean worktree changes invisible.
// The gate rejects semantic special-entry flags, so a normalized index for the
// same tree preserves staging content while deliberately resetting volatile
// stat/cache metadata.
func buildGateIndexForTree(repo, tree string) (gateIndexSnapshot, error) {
	if !validGitObjectID(tree) {
		return gateIndexSnapshot{}, errors.New("gate: exact recovery index tree is invalid")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-recovery-index-*")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	defer os.RemoveAll(temporary)
	indexPath := filepath.Join(temporary, "index")
	environment := gitIndexEnvironment(indexPath)
	if _, err := gitWriterCommandEnvironment(repo, environment, "read-tree", tree); err != nil {
		return gateIndexSnapshot{}, err
	}
	writtenTree, err := gitWriterCommandEnvironment(repo, environment, "write-tree")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	writtenTree = strings.TrimSpace(writtenTree)
	if writtenTree != tree {
		return gateIndexSnapshot{}, errors.New("gate: normalized recovery index differs from its tree authority")
	}
	return gateIndexSnapshot{Tree: tree, Data: data}, nil
}

func captureGateIndex(repo string) (gateIndexSnapshot, error) {
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	info, err := os.Lstat(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	if !info.Mode().IsRegular() {
		return gateIndexSnapshot{}, errors.New("gate: Git index must be one regular file")
	}
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	tree, err := indexTreeForBytes(repo, data)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	return gateIndexSnapshot{Tree: tree, Data: data, Mode: info.Mode().Perm()}, nil
}

func requireSupportedGateIndex(repo string, environment []string) error {
	shared, err := commandEnvironment(repo, environment, "git", "rev-parse", "--shared-index-path")
	if err != nil {
		return err
	}
	if strings.TrimSpace(shared) != "" {
		return errors.New("commit admission: split Git indexes are not supported by exact index recovery")
	}
	entries, err := commandEnvironment(repo, environment, "git", "ls-files", "-v", "-z")
	if err != nil {
		return err
	}
	for entry := range strings.SplitSeq(entries, "\x00") {
		if entry == "" {
			continue
		}
		if !strings.HasPrefix(entry, "H ") {
			return errors.New("commit admission: special Git index entry flags are not supported by exact completion staging")
		}
	}
	resolveUndo, err := commandEnvironment(repo, environment, "git", "ls-files", "--resolve-undo", "-z")
	if err != nil {
		return err
	}
	if resolveUndo != "" {
		return errors.New("commit admission: resolve-undo Git index metadata is not supported by exact completion staging")
	}
	return nil
}

func gateIndexPath(repo string) (string, error) {
	indexPath, err := gitMetadataPath(repo, "index")
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(indexPath) == "" {
		return "", errors.New("gate: Git index path is absent")
	}
	return indexPath, nil
}

func indexTreeForBytes(repo string, data []byte) (string, error) {
	if len(data) == 0 {
		return "", errors.New("gate: Git index bytes are absent")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-index-verify-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(temporary)
	indexPath := filepath.Join(temporary, "index")
	if err := os.WriteFile(indexPath, data, gatePrivateFileMode); err != nil {
		return "", err
	}
	environment := gitIndexEnvironment(indexPath)
	if err := requireSupportedGateIndex(repo, environment); err != nil {
		return "", err
	}
	tree, err := gitWriterCommandEnvironment(repo, environment, "write-tree")
	if err != nil {
		return "", err
	}
	tree = strings.TrimSpace(tree)
	if !validGitObjectID(tree) {
		return "", errors.New("gate: Git index produced an invalid tree identity")
	}
	indexed, err := commandEnvironment(repo, environment, "git", "ls-files", "-z")
	if err != nil {
		return "", err
	}
	treePaths, err := commandEnvironment(repo, environment, "git", "ls-tree", "-r", "--name-only", "-z", tree)
	if err != nil {
		return "", err
	}
	if !sameNULTerminatedNames(indexed, treePaths) {
		return "", errors.New("commit admission: intent-to-add or non-tree Git index entries are not supported by exact completion staging")
	}
	return tree, nil
}

func sameNULTerminatedNames(left, right string) bool {
	toSet := func(raw string) map[string]struct{} {
		values := make(map[string]struct{})
		for value := range strings.SplitSeq(raw, "\x00") {
			if value != "" {
				values[value] = struct{}{}
			}
		}
		return values
	}
	return maps.Equal(toSet(left), toSet(right))
}

func validateGateIntentIndexes(repo string, intent gateCommitIntent) error {
	beforeTree, err := indexTreeForBytes(repo, intent.IndexBefore)
	if err != nil {
		return err
	}
	afterTree, err := indexTreeForBytes(repo, intent.IndexAfter)
	if err != nil {
		return err
	}
	restoreTree, err := indexTreeForBytes(repo, intent.IndexRestore)
	if err != nil {
		return err
	}
	if beforeTree != intent.IndexTree || afterTree != intent.Tree || restoreTree != intent.IndexTree {
		return errors.New("gate: interrupted commit index bytes differ from their tree authorities")
	}
	keepaliveTree, err := buildGateIntentKeepaliveTree(repo, intent)
	if err != nil {
		return err
	}
	if keepaliveTree != intent.KeepaliveTree {
		return errors.New("gate: interrupted commit keepalive differs from its object authorities")
	}
	keepaliveCommit, err := buildGateIntentKeepaliveCommit(repo, intent, keepaliveTree)
	if err != nil {
		return err
	}
	if keepaliveCommit != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive differs from its commit authority")
	}
	return nil
}

const gateIntentKeepaliveNamespace = "refs/overgo/gate-intents/"

func gateIntentKeepaliveReference(preparation artifact.ID) string {
	return gateIntentKeepaliveNamespace + preparation.DigestHex()
}

// buildGateIntentKeepaliveTree creates a deterministic synthetic tree whose
// subtrees are every otherwise-unreferenced Git object source needed for exact
// rollback. A deterministic synthetic commit points at this tree and names the
// completion commit as its parent, making the one intent-owned ref a complete
// reachability root without walking ordinary history during validation.
func buildGateIntentKeepaliveTree(repo string, intent gateCommitIntent) (string, error) {
	if !validGitObjectID(intent.IndexTree) {
		return "", errors.New("gate: interrupted commit keepalive has no exact index tree")
	}
	type entry struct {
		name string
		tree string
	}
	if !validGitObjectID(intent.Tree) {
		return "", errors.New("gate: interrupted commit keepalive has no exact accepted tree")
	}
	entries := []entry{
		{name: "accepted", tree: intent.Tree},
		{name: "index", tree: intent.IndexTree},
	}
	if intent.Merge != nil && intent.Merge.AutoMerge != nil {
		autoMerge := strings.TrimSpace(string(intent.Merge.AutoMerge))
		if !validGitObjectID(autoMerge) {
			return "", errors.New("gate: interrupted commit keepalive has invalid AUTO_MERGE authority")
		}
		entries = append(entries, entry{name: "auto-merge", tree: autoMerge})
	}
	slices.SortFunc(entries, func(left, right entry) int { return strings.Compare(left.name, right.name) })
	var input bytes.Buffer
	for _, candidate := range entries {
		fmt.Fprintf(&input, "040000 tree %s\t%s\x00", candidate.tree, candidate.name)
	}
	process := newGateGitWriterCommand(repo, nil, "mktree", "-z")
	process.Stdin = &input
	output, err := process.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf(
			"gate: build interrupted commit keepalive tree: %w: %s",
			err, clioptions.Tail(string(output), clioptions.DiagnosticTailBytes),
		)
	}
	tree := strings.TrimSpace(string(output))
	if !validGitObjectID(tree) {
		return "", errors.New("gate: Git produced an invalid interrupted commit keepalive tree")
	}
	return tree, nil
}

func buildGateIntentKeepaliveCommit(repo string, intent gateCommitIntent, tree string) (string, error) {
	if !validGitObjectID(tree) || !validGitObjectID(intent.Commit) {
		return "", errors.New("gate: interrupted commit keepalive requires exact tree and completion commit authorities")
	}
	process := newGateGitWriterCommand(repo, nil, "commit-tree", tree, "-p", intent.Commit)
	process.Env = append(
		process.Env,
		"GIT_AUTHOR_NAME=Overgo Gate",
		"GIT_AUTHOR_EMAIL=gate@overgo.invalid",
		"GIT_AUTHOR_DATE=1970-01-01T00:00:00 +0000",
		"GIT_COMMITTER_NAME=Overgo Gate",
		"GIT_COMMITTER_EMAIL=gate@overgo.invalid",
		"GIT_COMMITTER_DATE=1970-01-01T00:00:00 +0000",
	)
	process.Stdin = strings.NewReader("overgo interrupted commit keepalive\n")
	output, err := process.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf(
			"gate: build interrupted commit keepalive commit: %w: %s",
			err, clioptions.Tail(string(output), clioptions.DiagnosticTailBytes),
		)
	}
	commit := strings.TrimSpace(string(output))
	if !validGitObjectID(commit) {
		return "", errors.New("gate: Git produced an invalid interrupted commit keepalive identity")
	}
	return commit, nil
}

func initializeGateIntentKeepalive(repo string, intent *gateCommitIntent) error {
	if intent == nil || !intent.Preparation.Valid() {
		return errors.New("gate: interrupted commit keepalive has no preparation authority")
	}
	tree, err := buildGateIntentKeepaliveTree(repo, *intent)
	if err != nil {
		return err
	}
	commit, err := buildGateIntentKeepaliveCommit(repo, *intent, tree)
	if err != nil {
		return err
	}
	intent.KeepaliveRef = gateIntentKeepaliveReference(intent.Preparation)
	intent.KeepaliveTree = tree
	intent.KeepaliveCommit = commit
	return nil
}

var (
	errGateIndexMoved            = errors.New("Git index moved beyond the write-ahead states")
	gateIndexCASBeforeLockHook   func(string)
	gateIndexCASLockedHook       func(string)
	gateMergeCASBeforeLockHook   func(string)
	gateMergeCASLockedHook       func(string)
	gateMergeCASAfterStepHook    func(string, string, string)
	gateCommitIndexInstalledHook func(string, gateCommitIntent)
	gateKeepaliveBoundHook       func(string, gateCommitIntent)
	gateRecoveryBeforeLockHook   func(string)
	gateRecoveryLockedHook       func(string)
	gateRecoveryAfterStepHook    func(string, string)
)

func installCompletionIndexUnderLock(repo, indexPath string, intent gateCommitIntent) error {
	if err := requireGateIntentKeepalive(repo, intent); err != nil {
		return err
	}
	return replaceGateIndexStatesUnderLock(
		indexPath,
		[][]byte{intent.IndexBefore},
		[][]byte{intent.IndexAfter},
		intent.IndexAfter,
		fs.FileMode(intent.IndexMode),
	)
}

func restoreCapturedIndex(repo string, intent gateCommitIntent) error {
	err := withGateGitStateLock(
		repo, fs.FileMode(intent.IndexMode), intent,
		gateIndexCASBeforeLockHook, gateIndexCASLockedHook,
		func(indexPath string) error {
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if err := normalizeGateIndexToStates(
				repo, indexPath, fs.FileMode(intent.IndexMode),
				intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
			); err != nil {
				return err
			}
			return replaceGateIndexStatesUnderLock(
				indexPath,
				[][]byte{intent.IndexAfter},
				[][]byte{intent.IndexBefore, intent.IndexRestore},
				intent.IndexRestore,
				fs.FileMode(intent.IndexMode),
			)
		},
	)
	if err != nil {
		if errors.Is(err, errGateIndexMoved) {
			return errors.New("gate: Git index moved beyond the write-ahead states; automatic recovery refused")
		}
		return err
	}
	return nil
}

func createGateCommit(repo string, intent gateCommitIntent, messageFile string) (string, error) {
	if !validGitObjectID(intent.Tree) || !validGitObjectID(intent.Parent) || strings.TrimSpace(messageFile) == "" {
		return "", errors.New("commit admission: exact tree, parent, and message file are required")
	}
	arguments := []string{"commit-tree", intent.Tree, "-p", intent.Parent}
	if intent.Merge != nil {
		for _, parent := range intent.Merge.parents() {
			arguments = append(arguments, "-p", parent)
		}
	}
	arguments = append(arguments, "-F", messageFile)
	output, err := gitAuthorityWriterOutput(repo, arguments...)
	if err != nil {
		return "", err
	}
	commit := strings.TrimSpace(string(output))
	if !validGitObjectID(commit) {
		return "", errors.New("commit admission: git commit-tree returned an invalid object identity")
	}
	return commit, nil
}

type gatePreparedReferenceTransaction struct {
	process  *exec.Cmd
	input    io.WriteCloser
	output   *bufio.Reader
	stderr   *bytes.Buffer
	finished bool
}

func prepareGateReferenceTransaction(
	repo, reference, target, observed, reason string,
) (*gatePreparedReferenceTransaction, error) {
	if !strings.HasPrefix(reference, "refs/heads/") || strings.ContainsAny(reference, "\x00\r\n \t") ||
		!validGitObjectID(target) || !validGitObjectID(observed) {
		return nil, errors.New("gate: prepared ref transaction requires an attached branch and exact objects")
	}
	stderr := new(bytes.Buffer)
	process := newGateGitWriterCommand(repo, nil, "update-ref", "-m", reason, "--stdin")
	input, err := process.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := process.StdoutPipe()
	if err != nil {
		_ = input.Close()
		return nil, err
	}
	process.Stderr = stderr
	if err := process.Start(); err != nil {
		_ = input.Close()
		return nil, err
	}
	transaction := &gatePreparedReferenceTransaction{
		process: process, input: input, output: bufio.NewReader(stdout), stderr: stderr,
	}
	if err := transaction.exchange("start", "start: ok"); err != nil {
		return nil, transaction.fail("start", err)
	}
	// Because reference is the current symbolic HEAD referent, Git includes
	// HEAD in the prepared branch update and holds HEAD.lock with the branch
	// lock. Explicitly queueing symref-verify HEAD is both redundant and rejected
	// by Git as a duplicate referent update; the post-prepare exact check below
	// binds the locked symbolic target before any shared-state mutation.
	instruction := fmt.Sprintf("update %s %s %s\n", reference, target, observed)
	if target == observed {
		instruction = fmt.Sprintf("verify %s %s\n", reference, observed)
	}
	if _, err := io.WriteString(transaction.input, instruction); err != nil {
		return nil, transaction.fail("queue", err)
	}
	if err := transaction.exchange("prepare", "prepare: ok"); err != nil {
		return nil, transaction.fail("prepare", err)
	}
	return transaction, nil
}

func (transaction *gatePreparedReferenceTransaction) exchange(command, expected string) error {
	if transaction == nil || transaction.finished {
		return errors.New("gate: prepared ref transaction is not active")
	}
	if _, err := io.WriteString(transaction.input, command+"\n"); err != nil {
		return err
	}
	response, err := transaction.output.ReadString('\n')
	if err != nil {
		return err
	}
	if strings.TrimSpace(response) != expected {
		return fmt.Errorf("unexpected Git ref transaction response %q", strings.TrimSpace(response))
	}
	return nil
}

func (transaction *gatePreparedReferenceTransaction) fail(phase string, cause error) error {
	if transaction == nil {
		return cause
	}
	_ = transaction.input.Close()
	waitErr := transaction.process.Wait()
	transaction.finished = true
	return fmt.Errorf(
		"gate: prepared ref transaction %s: %w: %s",
		phase, errors.Join(cause, waitErr), strings.TrimSpace(transaction.stderr.String()),
	)
}

func (transaction *gatePreparedReferenceTransaction) commit() error {
	if transaction == nil || transaction.finished {
		return errors.New("gate: prepared ref transaction is not active")
	}
	if err := transaction.exchange("commit", "commit: ok"); err != nil {
		return transaction.fail("commit", err)
	}
	closeErr := transaction.input.Close()
	waitErr := transaction.process.Wait()
	transaction.finished = true
	if err := errors.Join(closeErr, waitErr); err != nil {
		return fmt.Errorf(
			"gate: finish prepared ref transaction: %w: %s",
			err, strings.TrimSpace(transaction.stderr.String()),
		)
	}
	return nil
}

func (transaction *gatePreparedReferenceTransaction) abort() error {
	if transaction == nil || transaction.finished {
		return nil
	}
	if err := transaction.exchange("abort", "abort: ok"); err != nil {
		return transaction.fail("abort", err)
	}
	closeErr := transaction.input.Close()
	waitErr := transaction.process.Wait()
	transaction.finished = true
	if err := errors.Join(closeErr, waitErr); err != nil {
		return fmt.Errorf(
			"gate: abort prepared ref transaction: %w: %s",
			err, strings.TrimSpace(transaction.stderr.String()),
		)
	}
	return nil
}

// installCompletionIndexAndAdvanceGateReference holds the actual index and
// captured branch locks across exact index installation and branch CAS. It
// never updates whichever branch HEAD might name later, and detached HEAD is
// refused because Git cannot atomically distinguish it from a symbolic HEAD at
// the same object identity.
func installCompletionIndexAndAdvanceGateReference(repo string, intent gateCommitIntent) error {
	err := withGateGitStateLock(
		repo, fs.FileMode(intent.IndexMode), intent,
		gateIndexCASBeforeLockHook, gateIndexCASLockedHook,
		func(indexPath string) (err error) {
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("commit admission: HEAD reference or parent moved before prepared commit")
			}
			if err := exactGateIndexState(
				indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore,
			); err != nil {
				return err
			}
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			transaction, err := prepareGateReferenceTransaction(
				repo, intent.HeadReference, intent.Commit, intent.Parent, "overgo gate completion",
			)
			if err != nil {
				return err
			}
			defer func() { err = errors.Join(err, transaction.abort()) }()
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("commit admission: HEAD moved while preparing exact ref update")
			}
			if err := installCompletionIndexUnderLock(repo, indexPath, intent); err != nil {
				return err
			}
			if gateCommitIndexInstalledHook != nil {
				gateCommitIndexInstalledHook(repo, intent)
			}
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("commit admission: prepared branch moved before publication")
			}
			if err := exactGateIndexState(
				indexPath, fs.FileMode(intent.IndexMode), intent.IndexAfter,
			); err != nil {
				return err
			}
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if err := transaction.commit(); err != nil {
				return err
			}
			if !exactGateHead(repo, intent.HeadReference, intent.Commit) {
				return errors.New("commit admission: prepared branch publication has inexact HEAD authority")
			}
			return nil
		},
	)
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("commit admission: Git index moved before exact completion publication")
	}
	return err
}

func exactGateHead(repo, reference, commit string) bool {
	current, err := currentHeadReference(repo)
	return err == nil && current == reference && gateReferenceEquals(repo, reference, commit)
}

func gateReferenceEquals(repo, reference, commit string) bool {
	if strings.TrimSpace(reference) == "" || !validGitObjectID(commit) {
		return false
	}
	value, err := gitAuthorityOutput(repo, "rev-parse", "--verify", reference)
	return err == nil && strings.TrimSpace(string(value)) == commit
}

func validateManifestCommitAdmission(manifest automationcheck.ManifestPlan, terminal map[string]automationcheck.Evidence) error {
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("commit admission: %w", err)
	}
	if len(manifest.Invocations) == 0 || manifest.Invocations[len(manifest.Invocations)-1].Check.Name != "commit" {
		return errors.New("commit admission: commit is not the final planned invocation")
	}
	for _, invocation := range manifest.Invocations[:len(manifest.Invocations)-1] {
		evidence, found := terminal[invocation.Check.Name]
		if !found || !evidence.ID.Valid() {
			return fmt.Errorf("commit admission: %s lacks terminal evidence", invocation.Check.Name)
		}
		if evidence.Outcome != runrecord.LanePassed || evidence.Authority == nil || evidence.Authority.Plan != manifest.ID ||
			evidence.Authority.Definition != invocation.ID {
			return fmt.Errorf("commit admission: %s evidence has wrong outcome or authority", invocation.Check.Name)
		}
		if evidence.Inapplicable && evidence.Reused {
			return fmt.Errorf("commit admission: %s evidence has contradictory outcomes", invocation.Check.Name)
		}
	}
	return nil
}

// completionMessageFile writes the operator's message plus the
// structured completion trailers to a temporary file: the plan item
// and step this commit completes, and the verify command that gated
// it. Git carries completion history; the plan keeps only open work.
func (g *gateContext) completionMessageFile(document plan.Plan) (string, error) {
	message, err := os.ReadFile(g.messageFile)
	if err != nil {
		return "", err
	}
	itemID, stepID, _ := strings.Cut(g.planRef, "/")
	if g.manifestPlan == nil || g.candidateManifest == nil {
		return "", errors.New("completion message: manifest authorities are absent")
	}
	mergeAuthority := artifact.ID{}
	if g.mergeAuthority != nil {
		mergeAuthority = g.mergeAuthority.ID
	}
	augmented, err := plan.CompletionCommitMessageWithMergeAuthority(
		message, document, itemID, stepID, g.manifestPlan.ID, g.candidateManifest.ID, g.preparation.ID,
		g.preparationCommit, g.planProjection, mergeAuthority,
	)
	if err != nil {
		return "", err
	}
	file, err := os.CreateTemp("", "gate-message-*.txt")
	if err != nil {
		return "", err
	}
	if _, err := file.Write(augmented); err != nil {
		file.Close()
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	return file.Name(), nil
}

func (g *gateContext) deriveProjectedMergeAuthority(
	preAdvance, child plan.Plan,
	targetStore *overgodb.Store,
) (*plan.FirstParentTargetMergeAuthority, error) {
	if g.planProjection != plan.MergeProjectionFirstParentTarget {
		return nil, nil
	}
	if targetStore == nil || g.mergeBefore == nil || g.mergeSourceStore == "" {
		return nil, errors.New("first-parent-target merge requires target and source store authority")
	}
	parents := g.mergeBefore.parents()
	if len(parents) != 1 || !validGitObjectID(parents[0]) || parents[0] == g.planHead {
		return nil, errors.New("first-parent-target merge requires one distinct incoming parent")
	}
	incomingRevision := parents[0]
	loadPlan := func(label, revision string) (plan.Plan, error) {
		data, err := gitCompletionFile(g.repo, revision, plan.Path)
		if err != nil {
			return plan.Plan{}, err
		}
		document, err := plan.ParseHistorical(data)
		if err != nil {
			return plan.Plan{}, fmt.Errorf("parse %s plan %.12s: %w", label, revision, err)
		}
		return document, nil
	}
	local, err := loadPlan("local parent", g.planHead)
	if err != nil {
		return nil, err
	}
	incoming, err := loadPlan("incoming parent", incomingRevision)
	if err != nil {
		return nil, err
	}
	baseOutput, err := gitAuthorityOutput(g.repo, "merge-base", "--all", g.planHead, incomingRevision)
	if err != nil {
		return nil, err
	}
	bases := strings.Fields(string(baseOutput))
	if len(bases) != 1 {
		return nil, fmt.Errorf("first-parent-target merge requires one merge base, found %d", len(bases))
	}
	mergeBase, err := loadPlan("merge-base", bases[0])
	if err != nil {
		return nil, err
	}
	sourceStore, err := overgodb.OpenReadOnly(g.mergeSourceStore)
	if err != nil {
		return nil, fmt.Errorf("open projected merge source store: %w", err)
	}
	defer sourceStore.Close()
	localAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), g.repo, g.planHead, local, targetStore,
	)
	if err != nil {
		return nil, fmt.Errorf("resolve local projected-merge authority: %w", err)
	}
	incomingAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), g.repo, incomingRevision, incoming, sourceStore,
	)
	if err != nil {
		return nil, fmt.Errorf("resolve incoming projected-merge authority: %w", err)
	}
	item, step, found := strings.Cut(g.planRef, "/")
	if !found {
		return nil, errors.New("first-parent-target merge has an invalid completion reference")
	}
	receipt, err := plan.NewFirstParentTargetMergeAuthority(
		context.Background(), g.repo, g.planHead, incomingRevision, bases[0],
		local, incoming, mergeBase, preAdvance, child, item, step,
		g.preparation.ID, g.preparationCommit, localAuthority, incomingAuthority,
		targetStore, sourceStore,
	)
	if err != nil {
		return nil, err
	}
	return &receipt, nil
}

func publishPlanTransition(
	repo string,
	preparation artifact.ID,
	before, after []byte,
	mode fs.FileMode,
) (func() error, error) {
	path := filepath.Join(repo, filepath.FromSlash(plan.Path))
	scratch, err := gatePlanScratchPath(repo, preparation)
	if err != nil {
		return nil, err
	}
	if err := atomicfile.CompareAndSwap(path, scratch, before, after, mode); err != nil {
		if errors.Is(err, atomicfile.ErrChanged) {
			return nil, fmt.Errorf("commit admission: plan changed before transition publication: %w", err)
		}
		return nil, err
	}
	return func() error {
		if err := atomicfile.CompareAndSwap(path, scratch, after, before, mode); err != nil {
			return fmt.Errorf("commit admission: plan changed before transition rollback: %w", err)
		}
		return nil
	}, nil
}

func gatePlanScratchPath(repo string, preparation artifact.ID) (string, error) {
	if preparation.Kind() != artifact.KindEvidence || preparation.DigestHex() == "" {
		return "", errors.New("gate: plan transaction scratch requires an exact preparation")
	}
	root := filepath.Join(repo, filepath.FromSlash(gatePlanScratchDir))
	path := filepath.Join(root, preparation.DigestHex())
	if err := os.MkdirAll(path, gatePrivateDirectoryMode); err != nil {
		return "", fmt.Errorf("gate: create private plan transaction scratch: %w", err)
	}
	for _, directory := range []string{root, path} {
		info, err := os.Lstat(directory)
		if err != nil {
			return "", err
		}
		if !info.IsDir() || info.Mode()&fs.ModeSymlink != 0 ||
			runtime.GOOS != "windows" && info.Mode().Perm() != gatePrivateDirectoryMode.Perm() {
			return "", errors.New("gate: plan transaction scratch is not an exact private directory")
		}
	}
	return path, nil
}

func advancedPlanBytes(before []byte, ref string) ([]byte, error) {
	document, err := plan.Parse(before)
	if err != nil {
		return nil, err
	}
	item, step, found := strings.Cut(ref, "/")
	if !found || item == "" || step == "" {
		return nil, fmt.Errorf("advance plan: invalid reference %q", ref)
	}
	advanced, err := plan.Advance(document, item, step)
	if err != nil {
		return nil, err
	}
	data, err := json.MarshalIndent(advanced, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

func (g *gateContext) prepare() error {
	if err := requireNoPendingGateState(g.repo, g.storePath); err != nil {
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
	store, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle before Git commit: %w", err)
	}
	defer store.Close()
	g.resolveAttemptStrategy(store)
	alias, err := gatePreparationAlias(context.Background(), store, g.preparation.ID)
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle authority: %w", err)
	}
	batch.Aliases = append(batch.Aliases, alias)
	// The locator is write-ahead: a process loss after the append-only
	// preparation commit must not leave debt that the next gate cannot name.
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		return fmt.Errorf("prepare gate lifecycle locator: %w", err)
	}
	preparationCommit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle before Git commit: %w", err)
	}
	introduction, found, err := store.ArtifactIntroduction(context.Background(), g.preparation.ID)
	if err != nil {
		return fmt.Errorf("prepare gate lifecycle introduction: %w", err)
	}
	if !found || introduction.Commit != preparationCommit {
		return errors.New("prepare gate lifecycle introduction differs from its durable commit")
	}
	g.preparationCommit = introduction.Commit
	return nil
}

func requireNoPendingGateState(repo, storePath string) error {
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err == nil {
		return errors.New("gate: unresolved tmp/gate_debt.json; run `go run ./cmd/gate -reconcile` before another gate")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))); err == nil {
		return errors.New("gate: interrupted commit intent remains; run `go run ./cmd/gate -recover-interrupted`")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return fmt.Errorf("gate: inspect lifecycle authority: %w", err)
	}
	defer store.Close()
	if err := requireNoOutstandingGateLifecycle(context.Background(), store); err != nil {
		return err
	}
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return fmt.Errorf("gate: inspect lifecycle locator: %w", err)
	}
	if err := heartbeat.Validate(); err != nil {
		return fmt.Errorf("gate: invalid lifecycle locator: %w", err)
	}
	if heartbeat.State == runrecord.HeartbeatRunning || heartbeat.State == runrecord.HeartbeatRecordDebt {
		return errors.New("gate: unresolved gate lifecycle locator; run `go run ./cmd/gate -record-failure` before another gate")
	}
	return nil
}

func requireNoOutstandingGateLifecycle(ctx context.Context, store *overgodb.Store) error {
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return fmt.Errorf("gate: resolve lifecycle authority: %w", err)
	}
	lifecycles, err := runrecord.GateLifecyclesInStore(ctx, store)
	if err != nil {
		return fmt.Errorf("gate: inspect lifecycle debt: %w", err)
	}
	complete := make(map[artifact.ID]bool)
	var unresolved []artifact.ID
	for _, preparation := range lifecycles {
		if preparation.State != runrecord.GatePrepared {
			continue
		}
		finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
		if err != nil {
			return fmt.Errorf("gate: validate lifecycle %s: %w", preparation.ID, err)
		}
		if !finalized {
			unresolved = append(unresolved, preparation.ID)
			continue
		}
		if err := validateCompleteGateFinalization(ctx, store, preparation, finalization); err != nil {
			return fmt.Errorf("gate: validate lifecycle %s: %w", preparation.ID, err)
		}
		complete[finalization.ID] = true
	}
	if found {
		lifecycle, err := runrecord.RequireGateLifecycle(ctx, store, current)
		if err != nil {
			return fmt.Errorf("gate: load lifecycle authority: %w", err)
		}
		if lifecycle.State != runrecord.GateFinalized {
			return errors.New("gate: OvergoDB records an unresolved prepared lifecycle; run `go run ./cmd/gate -record-failure`")
		}
		if !complete[lifecycle.ID] {
			return errors.New("gate: lifecycle authority is not one complete typed gate finalization")
		}
	}
	if len(unresolved) == 1 {
		return errors.New("gate: OvergoDB records 1 unresolved prepared lifecycle; run `go run ./cmd/gate -record-failure`")
	}
	if len(unresolved) != 0 {
		// The refusal carries its own deterministic remediation: one exact
		// argument-bound recovery command per stale preparation.
		commands := make([]string, 0, len(unresolved))
		for _, preparation := range unresolved {
			commands = append(commands, "go run ./cmd/gate -record-failure -preparation "+preparation.String())
		}
		return fmt.Errorf(
			"gate: OvergoDB records %d unresolved prepared lifecycle(s); remediate each with: %s",
			len(unresolved), strings.Join(commands, " ; "),
		)
	}
	return nil
}

func requireCompleteGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	finalization runrecord.GateLifecycle,
) error {
	if finalization.State != runrecord.GateFinalized || finalization.Preparation == nil || finalization.Result == nil {
		return errors.New("current lifecycle alias does not target a finalization")
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, *finalization.Preparation)
	if err != nil {
		return err
	}
	canonical, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if !found || canonical.ID != finalization.ID {
		return errors.New("preparation does not have the current alias target as its sole canonical finalization")
	}
	return validateCompleteGateFinalization(ctx, store, preparation, finalization)
}

func validateCompleteGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	preparation, finalization runrecord.GateLifecycle,
) error {
	if preparation.State != runrecord.GatePrepared || finalization.State != runrecord.GateFinalized ||
		finalization.Preparation == nil || *finalization.Preparation != preparation.ID || finalization.Result == nil ||
		finalization.TreeKey != preparation.TreeKey || finalization.Environment != preparation.Environment ||
		finalization.Started != preparation.Started {
		return errors.New("gate finalization contradicts its preparation")
	}
	result, err := runrecord.RequireGateResult(ctx, store, *finalization.Result)
	if err != nil {
		return fmt.Errorf("load typed gate result: %w", err)
	}
	if result.Environment != finalization.Environment || result.CodeCommit != finalization.CodeCommit ||
		result.Outcome != finalization.Outcome {
		return errors.New("gate result contradicts its finalization")
	}
	if _, err := runrecord.RequireEnvironment(ctx, store, preparation.Environment); err != nil {
		return fmt.Errorf("load typed gate environment: %w", err)
	}
	introduction := func(id artifact.ID, label string) (overgodb.ArtifactIntroduction, error) {
		value, found, err := store.ArtifactIntroduction(ctx, id)
		if err != nil {
			return overgodb.ArtifactIntroduction{}, err
		}
		if !found {
			return overgodb.ArtifactIntroduction{}, fmt.Errorf("%s introduction is absent", label)
		}
		return value, nil
	}
	environmentIntroduction, err := introduction(preparation.Environment, "gate environment")
	if err != nil {
		return err
	}
	preparationIntroduction, err := introduction(preparation.ID, "gate preparation")
	if err != nil {
		return err
	}
	resultIntroduction, err := introduction(result.ID, "gate result")
	if err != nil {
		return err
	}
	finalizationIntroduction, err := introduction(finalization.ID, "gate finalization")
	if err != nil {
		return err
	}
	if environmentIntroduction.Sequence > preparationIntroduction.Sequence {
		return errors.New("gate environment was published after its preparation")
	}
	if preparationIntroduction.Sequence > finalizationIntroduction.Sequence {
		return errors.New("gate preparation was published after its finalization")
	}
	if resultIntroduction.Commit != finalizationIntroduction.Commit ||
		resultIntroduction.Sequence != finalizationIntroduction.Sequence {
		return errors.New("gate result and finalization were not published atomically")
	}
	edges, err := store.Children(ctx, result.ID)
	if err != nil {
		return err
	}
	for _, edge := range edges {
		if edge.Child == finalization.ID && edge.Parent == result.ID && edge.Relation == artifact.RelationDependsOn {
			return nil
		}
	}
	return errors.New("finalization lacks exact gate-result lineage")
}

func gatePreparationAlias(
	ctx context.Context,
	store *overgodb.Store,
	preparation artifact.ID,
) (artifact.AliasBinding, error) {
	// The writable store handle owns a stable view until this preparation is
	// committed. Repeat the complete census here so a preparation appended
	// after the gate-start precheck cannot hide behind a finalized alias.
	if err := requireNoOutstandingGateLifecycle(ctx, store); err != nil {
		return artifact.AliasBinding{}, err
	}
	binding := artifact.AliasBinding{Name: runrecord.GateLifecycleCurrentAlias, Target: preparation}
	current, found, err := artifact.ResolveAlias(ctx, store, binding.Name)
	if err != nil {
		return artifact.AliasBinding{}, err
	}
	if !found {
		return binding, nil
	}
	lifecycle, err := runrecord.RequireGateLifecycle(ctx, store, current)
	if err != nil {
		return artifact.AliasBinding{}, err
	}
	if lifecycle.State != runrecord.GateFinalized {
		return artifact.AliasBinding{}, errors.New("current gate lifecycle is not finalized")
	}
	binding.Previous = &current
	return binding, nil
}

// requireSoleCurrentGatePreparation is the locked-store admission boundary for
// a gate's final Git and record transaction. The current alias must still name
// this exact preparation, its durable introduction must be the receipt carried
// by Git, it must have no finalization yet, and every older preparation must
// already have one complete typed finalization. Callers retain the writable
// store handle through the final record append so no second writer can enter
// after this proof.
func requireSoleCurrentGatePreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparation runrecord.GateLifecycle,
	preparationCommit artifact.CommitID,
) error {
	if ctx == nil || store == nil || preparation.State != runrecord.GatePrepared ||
		!preparationCommit.Valid() {
		return errors.New("gate: exact current preparation authority is absent")
	}
	if err := preparation.ValidateIdentity(); err != nil {
		return err
	}
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return err
	}
	if !found || current != preparation.ID {
		return errors.New("gate: current lifecycle alias no longer names this preparation")
	}
	stored, err := runrecord.RequireGateLifecycle(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if stored.ID != preparation.ID || stored.State != runrecord.GatePrepared {
		return errors.New("gate: current preparation differs from its typed store authority")
	}
	introduction, found, err := store.ArtifactIntroduction(ctx, preparation.ID)
	if err != nil {
		return err
	}
	if !found || introduction.Commit != preparationCommit {
		return errors.New("gate: current preparation differs from its durable introduction receipt")
	}
	lifecycles, err := runrecord.GateLifecyclesInStore(ctx, store)
	if err != nil {
		return err
	}
	seenCurrent := false
	for _, candidate := range lifecycles {
		if candidate.State != runrecord.GatePrepared {
			continue
		}
		finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, candidate.ID)
		if err != nil {
			return err
		}
		if candidate.ID == preparation.ID {
			seenCurrent = true
			if finalized {
				return errors.New("gate: current preparation already has a finalization")
			}
			continue
		}
		if !finalized {
			return fmt.Errorf("gate: prior lifecycle %s has no finalization", candidate.ID)
		}
		if err := validateCompleteGateFinalization(ctx, store, candidate, finalization); err != nil {
			return fmt.Errorf("gate: validate prior lifecycle %s: %w", candidate.ID, err)
		}
	}
	if !seenCurrent {
		return errors.New("gate: current preparation is absent from the lifecycle census")
	}
	return nil
}

// requireGateDebtAdmission revalidates the append-only lifecycle immediately
// before a saved final batch is reconciled. A fresh debt may advance only the
// exact still-current preparation with no finalization. One matching terminal
// lifecycle is accepted only as an idempotent replay candidate: its alias must
// already be terminal, and Store.Commit below proves the same batch key and
// payload before returning success. An unaliased or different finalization can
// therefore never be followed by a second finalization from stale debt.
func requireGateDebtAdmission(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
	debtFinalization runrecord.GateLifecycle,
) error {
	if ctx == nil || store == nil || preparationID.Kind() != artifact.KindEvidence ||
		debtFinalization.State != runrecord.GateFinalized || debtFinalization.Preparation == nil ||
		*debtFinalization.Preparation != preparationID {
		return errors.New("gate: record debt lacks exact lifecycle authority")
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, preparationID)
	if err != nil {
		return fmt.Errorf("gate: debt preparation is unavailable: %w", err)
	}
	if preparation.State != runrecord.GatePrepared ||
		debtFinalization.TreeKey != preparation.TreeKey ||
		debtFinalization.Environment != preparation.Environment ||
		debtFinalization.Started != preparation.Started {
		return errors.New("gate: debt finalization contradicts its preparation")
	}
	finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if !finalized {
		introduction, found, err := store.ArtifactIntroduction(ctx, preparation.ID)
		if err != nil {
			return err
		}
		if !found {
			return errors.New("gate: debt preparation has no durable introduction authority")
		}
		return requireSoleCurrentGatePreparation(ctx, store, preparation, introduction.Commit)
	}
	if finalization.ID != debtFinalization.ID {
		return errors.New("gate: record debt preparation already has another finalization")
	}
	current, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return err
	}
	if !found || current != debtFinalization.ID {
		return errors.New("gate: matching debt finalization is not terminal lifecycle authority")
	}
	if err := validateCompleteGateFinalization(ctx, store, preparation, finalization); err != nil {
		return fmt.Errorf("gate: matching debt finalization is incomplete: %w", err)
	}
	return nil
}

func appendGateFinalizationAlias(
	batch *artifact.Batch,
	preparation, finalization artifact.ID,
) {
	previous := preparation
	batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
		Name: runrecord.GateLifecycleCurrentAlias, Target: finalization, Previous: &previous,
	})
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
	return writeJSON(g.repo, gateHeartbeatFile, heartbeat, clioptions.OutputFileMode)
}

func (g *gateContext) startHeartbeat() (func(), error) {
	if err := g.writeHeartbeat(runrecord.HeartbeatRunning); err != nil {
		return nil, err
	}
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.Tick(5 * time.Second)
		for {
			select {
			case <-ticker:
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

// gateCommitIntent is the write-ahead boundary between the Git ref update and
// the final OvergoDB batch. If the process dies after git commit, the exact
// parent, original index, final tree, preparation, and pre-prune plan bytes
// make rollback bounded and deterministic instead of leaving a pruned row with
// no successful authority or overwriting staged-only caller state.
type gateCommitIntent struct {
	Version           uint16               `json:"version"`
	Preparation       artifact.ID          `json:"preparation"`
	PreparationCommit artifact.CommitID    `json:"preparation_commit"`
	Recipe            artifact.ID          `json:"recipe"`
	CandidateManifest artifact.ID          `json:"candidate_manifest"`
	Parent            string               `json:"parent"`
	HeadReference     string               `json:"head_reference"`
	Merge             *gateMergeIntent     `json:"merge,omitempty"`
	PlanProjection    plan.MergeProjection `json:"plan_projection,omitzero"`
	MergeAuthority    []byte               `json:"merge_authority,omitempty"`
	Commit            string               `json:"commit,omitzero"`
	IndexTree         string               `json:"index_tree"`
	Tree              string               `json:"tree"`
	KeepaliveRef      string               `json:"keepalive_ref"`
	KeepaliveTree     string               `json:"keepalive_tree"`
	KeepaliveCommit   string               `json:"keepalive_commit"`
	IndexBefore       []byte               `json:"index_before"`
	IndexAfter        []byte               `json:"index_after"`
	IndexRestore      []byte               `json:"index_restore"`
	IndexMode         uint32               `json:"index_mode"`
	PlanRef           string               `json:"plan_ref"`
	Paths             []string             `json:"paths"`
	Plan              []byte               `json:"plan"`
	AdvancedPlan      []byte               `json:"advanced_plan"`
	PlanMode          uint32               `json:"plan_mode"`
}

var gatePlanRecoveryBeforeSwapHook func(string)

type gateMergeIntent struct {
	IndexTree         string `json:"index_tree"`
	Head              []byte `json:"head"`
	HeadFileMode      uint32 `json:"head_file_mode"`
	Mode              []byte `json:"mode"`
	ModeFileMode      uint32 `json:"mode_file_mode"`
	Message           []byte `json:"message"`
	MessageFileMode   uint32 `json:"message_file_mode"`
	AutoMerge         []byte `json:"auto_merge,omitempty"`
	AutoMergeFileMode uint32 `json:"auto_merge_file_mode,omitzero"`
}

func (intent gateCommitIntent) validate() error {
	projection, projectionErr := plan.ParseMergeProjection(string(intent.PlanProjection))
	if projectionErr != nil || projection != intent.PlanProjection ||
		projection == plan.MergeProjectionFirstParentTarget && intent.Merge == nil {
		return errors.New("gate: invalid interrupted commit plan projection")
	}
	if projection == plan.MergeProjectionFirstParentTarget {
		receipt, err := plan.ParseFirstParentTargetMergeAuthority(intent.MergeAuthority)
		if err != nil || receipt.ID.Kind() != artifact.KindEvidence {
			return errors.New("gate: invalid interrupted projected-merge authority")
		}
	} else if len(intent.MergeAuthority) != 0 {
		return errors.New("gate: semantic-union intent carries projected-merge authority")
	}
	if intent.Version != artifact.InitialDocumentVersion ||
		intent.Preparation.Kind() != artifact.KindEvidence || !intent.PreparationCommit.Valid() ||
		intent.Recipe.Kind() != artifact.KindRecipe ||
		intent.CandidateManifest.Kind() != artifact.KindProfile ||
		!validGitObjectID(intent.Parent) || !validGitObjectID(intent.Commit) ||
		!validGitObjectID(intent.IndexTree) || !validGitObjectID(intent.Tree) ||
		!validGitObjectID(intent.KeepaliveTree) || !validGitObjectID(intent.KeepaliveCommit) ||
		len(intent.IndexBefore) == 0 || len(intent.IndexAfter) == 0 || len(intent.IndexRestore) == 0 ||
		intent.IndexMode == 0 || intent.IndexMode > uint32(fs.ModePerm) ||
		len(intent.Plan) == 0 || len(intent.AdvancedPlan) == 0 ||
		strings.TrimSpace(intent.HeadReference) != intent.HeadReference || intent.HeadReference == "" ||
		strings.ContainsAny(intent.HeadReference, "\x00\r\n") || intent.PlanMode > uint32(fs.ModePerm) ||
		intent.PlanMode == 0 {
		return errors.New("gate: invalid interrupted commit intent")
	}
	if intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) {
		return errors.New("gate: interrupted commit intent has invalid keepalive reference authority")
	}
	if intent.Merge != nil {
		if err := intent.Merge.validate(intent.Parent); err != nil {
			return err
		}
		if intent.Merge.IndexTree != intent.IndexTree {
			return errors.New("gate: interrupted merge and shared-index authorities differ")
		}
	}
	if len(intent.Paths) == 0 {
		return errors.New("gate: interrupted commit intent has no scoped paths")
	}
	containsPlan := false
	seenPaths := make(map[string]bool, len(intent.Paths))
	for _, candidate := range intent.Paths {
		clean := filepath.Clean(filepath.FromSlash(candidate))
		if candidate == "" || clean == "." || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("gate: interrupted commit intent has an invalid scoped path")
		}
		canonical := filepath.ToSlash(clean)
		if seenPaths[canonical] {
			return errors.New("gate: interrupted commit intent repeats a scoped path")
		}
		seenPaths[canonical] = true
		containsPlan = containsPlan || canonical == plan.Path
	}
	if !containsPlan {
		return errors.New("gate: interrupted commit intent does not scope the plan")
	}
	item, step, found := strings.Cut(intent.PlanRef, "/")
	if !found || item == "" || step == "" {
		return errors.New("gate: interrupted commit intent has invalid plan reference")
	}
	before, err := plan.Parse(intent.Plan)
	if err != nil {
		return fmt.Errorf("gate: interrupted commit intent has invalid plan bytes: %w", err)
	}
	after, err := plan.Parse(intent.AdvancedPlan)
	if err != nil {
		return fmt.Errorf("gate: interrupted commit intent has invalid advanced plan bytes: %w", err)
	}
	expected, err := plan.Advance(before, item, step)
	if err != nil {
		return fmt.Errorf("gate: interrupted commit intent cannot derive its plan transition: %w", err)
	}
	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		return err
	}
	afterJSON, err := json.Marshal(after)
	if err != nil {
		return err
	}
	if !bytes.Equal(expectedJSON, afterJSON) {
		return errors.New("gate: interrupted commit intent has the wrong advanced plan")
	}
	return nil
}

func (merge gateMergeIntent) validate(parent string) error {
	if !validGitObjectID(merge.IndexTree) || len(merge.Head) == 0 || len(merge.Message) == 0 {
		return errors.New("gate: interrupted commit intent has invalid merge state")
	}
	for _, mode := range []uint32{merge.HeadFileMode, merge.ModeFileMode, merge.MessageFileMode} {
		if mode == 0 || mode > uint32(fs.ModePerm) {
			return errors.New("gate: interrupted commit intent has invalid merge metadata permissions")
		}
	}
	parents := strings.Fields(string(merge.Head))
	if len(parents) != 1 || !validGitObjectID(parents[0]) || parents[0] == parent {
		return errors.New("gate: interrupted commit intent requires one distinct merge parent")
	}
	if merge.AutoMerge != nil {
		autoMerge := strings.TrimSpace(string(merge.AutoMerge))
		if merge.AutoMergeFileMode == 0 || merge.AutoMergeFileMode > uint32(fs.ModePerm) ||
			!validGitObjectID(autoMerge) ||
			!bytes.Equal(merge.AutoMerge, []byte(autoMerge)) &&
				!bytes.Equal(merge.AutoMerge, []byte(autoMerge+"\n")) {
			return errors.New("gate: interrupted commit intent has invalid AUTO_MERGE authority")
		}
	} else if merge.AutoMergeFileMode != 0 {
		return errors.New("gate: interrupted commit intent has AUTO_MERGE permissions without content")
	}
	return nil
}

func (merge gateMergeIntent) parents() []string {
	return strings.Fields(string(merge.Head))
}

func validGitObjectID(value string) bool {
	return gitauthority.ValidObjectID(value)
}

func currentHeadReference(repo string) (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitReaderCommand(repo, "symbolic-ref", "--quiet", "HEAD")
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err == nil {
		reference := strings.TrimSpace(stdout.String())
		if reference == "" {
			return "", errors.New("gate: cannot resolve the current HEAD reference")
		}
		return reference, nil
	}
	exitError, exitFailure := errors.AsType[*exec.ExitError](err)
	if !exitFailure || exitError.ExitCode() != 1 {
		return "", fmt.Errorf("gate: resolve current HEAD reference: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if _, verifyErr := gitAuthorityOutput(repo, "rev-parse", "--verify", "HEAD"); verifyErr != nil {
		return "", verifyErr
	}
	return "HEAD", nil
}

func gitMetadataPath(repo, name string) (string, error) {
	value, err := command(repo, "git", "rev-parse", "--git-path", name)
	if err != nil {
		return "", err
	}
	value = strings.TrimSpace(value)
	if !filepath.IsAbs(value) {
		value = filepath.Join(repo, value)
	}
	return filepath.Clean(value), nil
}

var gateBeforeCommitStateHook func(string)

func captureGateStartState(repo string) (*gateMergeIntent, gateIndexSnapshot, error) {
	if err := requireFilesGateRefFormat(repo); err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	for _, name := range gitOperationMarkers {
		path, err := gitMetadataPath(repo, name)
		if err != nil {
			return nil, gateIndexSnapshot{}, err
		}
		if _, err := os.Stat(path); err == nil {
			return nil, gateIndexSnapshot{}, fmt.Errorf(
				"commit admission: Git operation %s is not supported by gate",
				name,
			)
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, gateIndexSnapshot{}, err
		}
	}
	if err := requireSupportedMergeRR(repo); err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	merge, err := capturePendingMerge(repo)
	if err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	index, err := captureGateIndex(repo)
	if err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	if merge != nil && merge.IndexTree != index.Tree {
		return nil, gateIndexSnapshot{}, errors.New(
			"commit admission: pending merge index changed while capturing the gate-start state",
		)
	}
	return merge, index, nil
}

func (g *gateContext) requireGateStartState() error {
	if g == nil || g.repo == "" || !validGitObjectID(g.indexBefore.Tree) || len(g.indexBefore.Data) == 0 ||
		g.indexBefore.Mode.Perm() == 0 {
		return errors.New("commit admission: exact gate-start Git state is absent")
	}
	if gateBeforeCommitStateHook != nil {
		gateBeforeCommitStateHook(g.repo)
	}
	merge, index, err := captureGateStartState(g.repo)
	if err != nil {
		return err
	}
	if index.Tree != g.indexBefore.Tree || index.Mode.Perm() != g.indexBefore.Mode.Perm() ||
		!bytes.Equal(index.Data, g.indexBefore.Data) {
		return errors.New("commit admission: Git index moved after the gate-start snapshot")
	}
	if !sameGateMergeState(merge, g.mergeBefore) {
		return errors.New("commit admission: pending merge moved after the gate-start snapshot")
	}
	return nil
}

func sameGateMergeState(left, right *gateMergeIntent) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.IndexTree == right.IndexTree && bytes.Equal(left.Head, right.Head) &&
		bytes.Equal(left.Mode, right.Mode) && bytes.Equal(left.Message, right.Message) &&
		bytes.Equal(left.AutoMerge, right.AutoMerge) &&
		left.HeadFileMode == right.HeadFileMode && left.ModeFileMode == right.ModeFileMode &&
		left.MessageFileMode == right.MessageFileMode && left.AutoMergeFileMode == right.AutoMergeFileMode
}

func capturePendingMerge(repo string) (*gateMergeIntent, error) {
	if err := requireFilesGateRefFormat(repo); err != nil {
		return nil, err
	}
	if err := requireSupportedMergeRR(repo); err != nil {
		return nil, err
	}
	headPath, err := gitMetadataPath(repo, "MERGE_HEAD")
	if err != nil {
		return nil, err
	}
	headInfo, err := os.Lstat(headPath)
	if errors.Is(err, os.ErrNotExist) {
		autoMergePath, pathErr := gitMetadataPath(repo, "AUTO_MERGE")
		if pathErr != nil {
			return nil, pathErr
		}
		if _, autoMergeErr := os.Lstat(autoMergePath); autoMergeErr == nil {
			return nil, errors.New("gate: orphan AUTO_MERGE exists without MERGE_HEAD")
		} else if !errors.Is(autoMergeErr, os.ErrNotExist) {
			return nil, autoMergeErr
		}
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !headInfo.Mode().IsRegular() {
		return nil, errors.New("gate: pending MERGE_HEAD authority is not a regular file")
	}
	head, err := os.ReadFile(headPath)
	if err != nil {
		return nil, err
	}
	autostashPath, err := gitMetadataPath(repo, "MERGE_AUTOSTASH")
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(autostashPath); err == nil {
		return nil, errors.New("gate: pending merge uses MERGE_AUTOSTASH; abort it and stage the merge without autostash")
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	read := func(name string) ([]byte, uint32, error) {
		path, pathErr := gitMetadataPath(repo, name)
		if pathErr != nil {
			return nil, 0, pathErr
		}
		info, statErr := os.Lstat(path)
		if statErr != nil {
			return nil, 0, statErr
		}
		if !info.Mode().IsRegular() {
			return nil, 0, fmt.Errorf("gate: pending merge metadata %s is not a regular file", name)
		}
		data, readErr := os.ReadFile(path)
		return data, uint32(info.Mode().Perm()), readErr
	}
	mode, modeFileMode, err := read("MERGE_MODE")
	if err != nil {
		return nil, err
	}
	message, messageFileMode, err := read("MERGE_MSG")
	if err != nil {
		return nil, err
	}
	autoMergePath, err := gitMetadataPath(repo, "AUTO_MERGE")
	if err != nil {
		return nil, err
	}
	autoMergeInfo, err := os.Lstat(autoMergePath)
	var autoMerge []byte
	var autoMergeFileMode uint32
	if errors.Is(err, os.ErrNotExist) {
		autoMerge = nil
	} else if err != nil {
		return nil, err
	} else if !autoMergeInfo.Mode().IsRegular() {
		return nil, errors.New("gate: pending AUTO_MERGE authority is not a regular file")
	} else if autoMerge, err = os.ReadFile(autoMergePath); err != nil {
		return nil, err
	} else {
		autoMergeFileMode = uint32(autoMergeInfo.Mode().Perm())
	}
	index, err := captureGateIndex(repo)
	if err != nil {
		return nil, err
	}
	merge := &gateMergeIntent{
		IndexTree: index.Tree,
		Head:      head, HeadFileMode: uint32(headInfo.Mode().Perm()),
		Mode: mode, ModeFileMode: modeFileMode,
		Message: message, MessageFileMode: messageFileMode,
		AutoMerge: autoMerge, AutoMergeFileMode: autoMergeFileMode,
	}
	if err := merge.validate(""); err != nil {
		return nil, err
	}
	return merge, nil
}

func requireSupportedMergeRR(repo string) error {
	mergeRRPath, err := gitMetadataPath(repo, "MERGE_RR")
	if err != nil {
		return err
	}
	info, err := os.Lstat(mergeRRPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("gate: MERGE_RR is not a regular file")
	}
	mergeRR, err := os.ReadFile(mergeRRPath)
	if err != nil {
		return err
	}
	if len(mergeRR) != 0 {
		return errors.New(
			"gate: pending merge has nonempty MERGE_RR; finish or clear rerere state before gate",
		)
	}
	return nil
}

func requireFilesGateRefFormat(repo string) error {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitReaderCommand(repo, "config", "--local", "--get", "extensions.refStorage")
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	format := strings.TrimSpace(stdout.String())
	if err != nil {
		exitError, exitFailure := errors.AsType[*exec.ExitError](err)
		if !exitFailure || exitError.ExitCode() != 1 || format != "" {
			return fmt.Errorf(
				"gate: resolve Git reference format: %w: %s", err, strings.TrimSpace(stderr.String()),
			)
		}
		// Older files-backend repositories predate extensions.refStorage and
		// report a missing config key with exit 1. Reftable repositories name
		// that extension explicitly, so absence is a positive files proof.
		format = "files"
	}
	if strings.ContainsAny(format, "\r\n") || format != "files" {
		return fmt.Errorf(
			"gate: Git reference format %q cannot provide exact AUTO_MERGE file authority; files is required",
			format,
		)
	}
	return nil
}

var errGateMergeMetadataMoved = errors.New("Git merge metadata moved beyond the write-ahead states")

type gateMergeMetadataFile struct {
	name       string
	path       string
	want       []byte
	mode       fs.FileMode
	managed    bool
	allowEmpty bool
}

type gateMergeMetadataState struct {
	present bool
	data    []byte
	mode    fs.FileMode
}

func gateMergeMetadataFiles(repo string, merge *gateMergeIntent) ([]gateMergeMetadataFile, error) {
	files := []gateMergeMetadataFile{
		{name: "MERGE_MODE"},
		{name: "MERGE_MSG"},
		{name: "AUTO_MERGE"},
		// MERGE_RR is unsupported rerere state. An absent or empty file is
		// preserved exactly as Git left it; nonempty content is never consumed,
		// removed, or restored by the gate transaction.
		{name: "MERGE_RR", allowEmpty: true},
		// MERGE_HEAD is the operation-presence marker. It is always the last
		// cleanup removal and the last recovery publication.
		{name: "MERGE_HEAD"},
	}
	if merge != nil {
		files[0].want, files[0].mode, files[0].managed = merge.Mode, fs.FileMode(merge.ModeFileMode), true
		files[1].want, files[1].mode, files[1].managed = merge.Message, fs.FileMode(merge.MessageFileMode), true
		if merge.AutoMerge != nil {
			files[2].want, files[2].mode, files[2].managed = merge.AutoMerge, fs.FileMode(merge.AutoMergeFileMode), true
		}
		files[4].want, files[4].mode, files[4].managed = merge.Head, fs.FileMode(merge.HeadFileMode), true
	}
	for index := range files {
		metadataPath, err := gitMetadataPath(repo, files[index].name)
		if err != nil {
			return nil, err
		}
		files[index].path = metadataPath
	}
	return files, nil
}

func readGateMergeMetadata(files []gateMergeMetadataFile) ([]gateMergeMetadataState, error) {
	states := make([]gateMergeMetadataState, len(files))
	for index, file := range files {
		info, err := os.Lstat(file.path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("%w: %s is not a regular file", errGateMergeMetadataMoved, file.name)
		}
		data, err := os.ReadFile(file.path)
		if err != nil {
			return nil, err
		}
		states[index] = gateMergeMetadataState{present: true, data: data, mode: info.Mode().Perm()}
	}
	return states, nil
}

// gateMergeMetadataProgress recognizes the only crash-recoverable states for
// the ordered MODE, MSG, optional AUTO_MERGE, HEAD transaction. A present
// prefix is restoration progress; an absent prefix is cleanup progress.
// Unmanaged metadata must remain absent, except that MERGE_RR may remain an
// untouched empty file.
func gateMergeMetadataProgress(
	files []gateMergeMetadataFile,
	states []gateMergeMetadataState,
	presentPrefix bool,
) (int, bool) {
	var progress int
	if len(files) != len(states) {
		return progress, false
	}
	var managedStates []gateMergeMetadataState
	for index, state := range states {
		file := files[index]
		if !file.managed {
			if state.present && !(file.allowEmpty && len(state.data) == 0) {
				return progress, false
			}
			continue
		}
		if state.present && (!bytes.Equal(state.data, file.want) || state.mode != file.mode.Perm()) {
			return progress, false
		}
		managedStates = append(managedStates, state)
	}
	for progress < len(managedStates) && managedStates[progress].present == presentPrefix {
		progress++
	}
	for index := progress; index < len(managedStates); index++ {
		if managedStates[index].present == presentPrefix {
			return progress, false
		}
	}
	return progress, true
}

func gateMergeManagedMetadata(files []gateMergeMetadataFile) []gateMergeMetadataFile {
	managed := make([]gateMergeMetadataFile, 0, len(files))
	for _, file := range files {
		if file.managed {
			managed = append(managed, file)
		}
	}
	return managed
}

type gateGitLockPath struct {
	name string
	path string
	mode fs.FileMode
}

type gateGitLockMarker struct {
	Version            uint16 `json:"version"`
	Authority          string `json:"authority"`
	Preparation        string `json:"preparation"`
	KeepaliveReference string `json:"keepalive_reference"`
	KeepaliveCommit    string `json:"keepalive_commit"`
	WorktreeGitDir     string `json:"worktree_git_dir"`
	IndexPath          string `json:"index_path"`
	IntentSHA256       string `json:"intent_sha256"`
}

type gateGitLockIntentAuthority struct {
	Version           uint16               `json:"version"`
	Preparation       string               `json:"preparation"`
	PreparationCommit string               `json:"preparation_commit"`
	Recipe            string               `json:"recipe"`
	CandidateManifest string               `json:"candidate_manifest"`
	Parent            string               `json:"parent"`
	HeadReference     string               `json:"head_reference"`
	Merge             *gateMergeIntent     `json:"merge,omitempty"`
	PlanProjection    plan.MergeProjection `json:"plan_projection,omitzero"`
	MergeAuthority    []byte               `json:"merge_authority,omitempty"`
	Commit            string               `json:"commit,omitzero"`
	IndexTree         string               `json:"index_tree"`
	Tree              string               `json:"tree"`
	KeepaliveRef      string               `json:"keepalive_ref"`
	KeepaliveTree     string               `json:"keepalive_tree"`
	KeepaliveCommit   string               `json:"keepalive_commit"`
	IndexBefore       []byte               `json:"index_before"`
	IndexAfter        []byte               `json:"index_after"`
	IndexRestore      []byte               `json:"index_restore"`
	IndexMode         uint32               `json:"index_mode"`
	PlanRef           string               `json:"plan_ref"`
	Paths             []string             `json:"paths"`
	Plan              []byte               `json:"plan"`
	AdvancedPlan      []byte               `json:"advanced_plan"`
	PlanMode          uint32               `json:"plan_mode"`
}

func gateGitManualLockPaths(
	repo string,
	mode fs.FileMode,
	intent gateCommitIntent,
) (string, []gateGitLockPath, error) {
	if mode.Perm() == 0 || mode.Perm() != mode {
		return "", nil, errors.New("gate: exact Git index mode is required for merge metadata")
	}
	if !strings.HasPrefix(intent.KeepaliveRef, gateIntentKeepaliveNamespace) ||
		strings.ContainsAny(intent.KeepaliveRef, "\x00\r\n") {
		return "", nil, fmt.Errorf(
			"gate: invalid interrupted commit keepalive lock authority %q", intent.KeepaliveRef,
		)
	}
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		return "", nil, err
	}
	locks := []gateGitLockPath{{name: "index", path: indexPath + ".lock", mode: mode.Perm()}}
	rootRefNames := []string{
		"AUTO_MERGE", "MERGE_HEAD", "MERGE_AUTOSTASH", "CHERRY_PICK_HEAD", "REVERT_HEAD",
		intent.KeepaliveRef,
	}
	for _, name := range rootRefNames {
		path, err := gitMetadataPath(repo, name)
		if err != nil {
			return "", nil, err
		}
		if name == intent.KeepaliveRef {
			if err := os.MkdirAll(filepath.Dir(path), gatePrivateDirectoryMode); err != nil {
				return "", nil, fmt.Errorf("gate: prepare interrupted commit keepalive lock: %w", err)
			}
		}
		locks = append(locks, gateGitLockPath{
			name: name, path: path + ".lock", mode: gatePrivateFileMode,
		})
	}
	return indexPath, locks, nil
}

func gateGitLockMarkerBytes(repo, indexPath string, intent gateCommitIntent) ([]byte, error) {
	if !intent.Preparation.Valid() || intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) ||
		!validGitObjectID(intent.KeepaliveCommit) {
		return nil, errors.New("gate: Git lock marker has invalid interrupted-intent authority")
	}
	if err := requireGateIntentKeepalive(repo, intent); err != nil {
		return nil, err
	}
	canonicalIntent, err := json.Marshal(gateGitLockIntentAuthority{
		Version: intent.Version, Preparation: intent.Preparation.String(),
		PreparationCommit: intent.PreparationCommit.String(), Recipe: intent.Recipe.String(),
		CandidateManifest: intent.CandidateManifest.String(), Parent: intent.Parent,
		HeadReference: intent.HeadReference, Merge: intent.Merge, PlanProjection: intent.PlanProjection,
		MergeAuthority: intent.MergeAuthority,
		Commit:         intent.Commit,
		IndexTree:      intent.IndexTree, Tree: intent.Tree, KeepaliveRef: intent.KeepaliveRef,
		KeepaliveTree: intent.KeepaliveTree, KeepaliveCommit: intent.KeepaliveCommit,
		IndexBefore: intent.IndexBefore, IndexAfter: intent.IndexAfter, IndexRestore: intent.IndexRestore,
		IndexMode: intent.IndexMode, PlanRef: intent.PlanRef, Paths: intent.Paths,
		Plan: intent.Plan, AdvancedPlan: intent.AdvancedPlan, PlanMode: intent.PlanMode,
	})
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(canonicalIntent)
	absoluteIndex, err := filepath.Abs(indexPath)
	if err != nil {
		return nil, err
	}
	absoluteIndex = filepath.Clean(absoluteIndex)
	marker := gateGitLockMarker{
		Version: artifact.InitialDocumentVersion, Authority: "overgo-gate-git-lock/v1",
		Preparation: intent.Preparation.String(), KeepaliveReference: intent.KeepaliveRef,
		KeepaliveCommit: intent.KeepaliveCommit, WorktreeGitDir: filepath.Dir(absoluteIndex),
		IndexPath: absoluteIndex, IntentSHA256: hex.EncodeToString(digest[:]),
	}
	return json.Marshal(marker)
}

func gateGitStateProcessLockPath(repo string) (string, error) {
	output, err := gitAuthorityOutput(repo, "rev-parse", "--git-common-dir")
	if err != nil {
		return "", err
	}
	commonDirectory := strings.TrimSpace(string(output))
	if commonDirectory == "" || strings.ContainsAny(commonDirectory, "\x00\r\n") {
		return "", errors.New("gate: Git common directory is unavailable for state locking")
	}
	if !filepath.IsAbs(commonDirectory) {
		commonDirectory = filepath.Join(repo, commonDirectory)
	}
	commonDirectory = filepath.Clean(commonDirectory)
	if err := os.MkdirAll(commonDirectory, gatePrivateDirectoryMode); err != nil {
		return "", err
	}
	return filepath.Join(commonDirectory, gateGitStateLockFile), nil
}

func inspectGateGitLockMarker(path string, marker []byte, mode fs.FileMode) (bool, bool, error) {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, false, nil
	}
	if err != nil {
		return false, false, err
	}
	if !info.Mode().IsRegular() || runtime.GOOS != "windows" && info.Mode().Perm() != mode.Perm() ||
		info.Size() != int64(len(marker)) {
		return true, false, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return true, false, err
	}
	return true, bytes.Equal(data, marker), nil
}

func removeExactGateGitLockMarkers(
	locks []gateGitLockPath,
	marker []byte,
	requirePresent bool,
) error {
	toRemove := make([]gateGitLockPath, 0, len(locks))
	for _, lock := range locks {
		present, exact, err := inspectGateGitLockMarker(lock.path, marker, lock.mode)
		if err != nil {
			return err
		}
		if !present {
			if requirePresent {
				return fmt.Errorf("gate: owned Git lock %s disappeared before release", lock.name)
			}
			// fsatomic.Remove is also the idempotent cleanup for a Windows
			// write-through tombstone left after the target name disappeared.
			toRemove = append(toRemove, lock)
			continue
		}
		if !exact {
			return fmt.Errorf(
				"gate: Git lock %s has partial or foreign ownership; automatic recovery refused",
				lock.name,
			)
		}
		toRemove = append(toRemove, lock)
	}
	var err error
	for index := len(toRemove) - 1; index >= 0; index-- {
		if removeErr := fsatomic.Remove(toRemove[index].path); removeErr != nil {
			err = errors.Join(err, fmt.Errorf("remove owned Git lock %s: %w", toRemove[index].name, removeErr))
		}
	}
	return err
}

func publishGateGitLockMarker(lock gateGitLockPath, marker []byte) (err error) {
	directory := filepath.Dir(lock.path)
	prepared, err := os.CreateTemp(directory, ".overgo-gate-lock-marker-*")
	if err != nil {
		return err
	}
	preparedPath := prepared.Name()
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, prepared.Close())
		}
		_ = fsatomic.Remove(preparedPath)
	}()
	if err := prepared.Chmod(lock.mode); err != nil {
		return err
	}
	if _, err := prepared.Write(marker); err != nil {
		return err
	}
	if err := prepared.Sync(); err != nil {
		return err
	}
	if err := prepared.Close(); err != nil {
		return err
	}
	closed = true
	if err := os.Link(preparedPath, lock.path); err != nil {
		return err
	}
	if err := fsatomic.SyncFile(lock.path); err != nil {
		return err
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return err
	}
	present, exact, err := inspectGateGitLockMarker(lock.path, marker, lock.mode)
	if err != nil {
		return err
	}
	if !present || !exact {
		return fmt.Errorf("gate: Git lock %s marker publication is inexact", lock.name)
	}
	return nil
}

func acquireGateGitLockMarkers(
	locks []gateGitLockPath,
	marker []byte,
) ([]gateGitLockPath, error) {
	owned := make([]gateGitLockPath, 0, len(locks))
	for _, lock := range locks {
		if err := publishGateGitLockMarker(lock, marker); err != nil {
			present, exact, inspectErr := inspectGateGitLockMarker(lock.path, marker, lock.mode)
			if inspectErr == nil && present && exact {
				owned = append(owned, lock)
			}
			cleanupErr := removeExactGateGitLockMarkers(owned, marker, true)
			return nil, errors.Join(
				fmt.Errorf("gate: acquire exact %s lock: %w", lock.name, err), inspectErr, cleanupErr,
			)
		}
		owned = append(owned, lock)
	}
	return owned, nil
}

func withGateMergeMetadataLock(
	repo string,
	mode fs.FileMode,
	intent gateCommitIntent,
	transaction func(string) error,
) error {
	return withGateGitStateLock(
		repo, mode, intent, gateMergeCASBeforeLockHook, gateMergeCASLockedHook, transaction,
	)
}

// withGateGitStateLock serializes the shared index, merge root refs, and the
// durable intent's keepalive ref. A process-lifetime guard makes exact marker
// residues reclaimable after abrupt process death; marker census refuses any
// partial or foreign contents before deleting even one pathname.
func withGateGitStateLock(
	repo string,
	mode fs.FileMode,
	intent gateCommitIntent,
	beforeLock, locked func(string),
	transaction func(string) error,
) (err error) {
	if err := requireFilesGateRefFormat(repo); err != nil {
		return err
	}
	indexPath, manualLocks, err := gateGitManualLockPaths(repo, mode, intent)
	if err != nil {
		return err
	}
	marker, err := gateGitLockMarkerBytes(repo, indexPath, intent)
	if err != nil {
		return err
	}
	processLockPath, err := gateGitStateProcessLockPath(repo)
	if err != nil {
		return err
	}
	processGuard, err := processlock.Acquire(processLockPath, gatePrivateFileMode)
	if err != nil {
		return fmt.Errorf("gate: another Git-state transaction is active: %w", err)
	}
	defer func() { err = errors.Join(err, processGuard.Close()) }()
	if err := removeExactGateGitLockMarkers(manualLocks, marker, false); err != nil {
		return err
	}
	if beforeLock != nil {
		beforeLock(repo)
	}
	ownedLocks, err := acquireGateGitLockMarkers(manualLocks, marker)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, removeExactGateGitLockMarkers(ownedLocks, marker, true))
	}()
	if locked != nil {
		locked(repo)
	}
	return transaction(indexPath)
}

func gateIndexState(indexPath string, mode fs.FileMode) ([]byte, error) {
	info, err := os.Lstat(indexPath)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Mode().Perm() != mode.Perm() {
		return nil, errGateIndexMoved
	}
	return os.ReadFile(indexPath)
}

func exactGateIndexState(indexPath string, mode fs.FileMode, accepted ...[]byte) error {
	current, err := gateIndexState(indexPath, mode)
	if err != nil {
		return err
	}
	if slices.ContainsFunc(accepted, func(state []byte) bool { return bytes.Equal(current, state) }) {
		return nil
	}
	return errGateIndexMoved
}

func replaceGateIndexStatesUnderLock(
	indexPath string,
	sources, accepted [][]byte,
	target []byte,
	mode fs.FileMode,
) error {
	if len(sources) == 0 || len(accepted) == 0 || len(target) == 0 ||
		mode.Perm() == 0 || mode.Perm() != mode {
		return errors.New("gate: exact locked Git index states and mode are required")
	}
	current, err := gateIndexState(indexPath, mode)
	if err != nil {
		return err
	}
	if slices.ContainsFunc(accepted, func(state []byte) bool { return bytes.Equal(current, state) }) {
		return nil
	}
	if !slices.ContainsFunc(sources, func(state []byte) bool { return bytes.Equal(current, state) }) {
		return errGateIndexMoved
	}
	if err := atomicfile.Write(indexPath, target, mode.Perm()); err != nil {
		return err
	}
	return exactGateIndexState(indexPath, mode, target)
}

func removeGateMergeMetadata(
	repo, operation string,
	files []gateMergeMetadataFile,
	progress int,
) error {
	managed := gateMergeManagedMetadata(files)
	for progress < len(managed) {
		states, err := readGateMergeMetadata(files)
		if err != nil {
			return err
		}
		current, exact := gateMergeMetadataProgress(files, states, false)
		if !exact || current != progress {
			return errGateMergeMetadataMoved
		}
		file := managed[progress]
		if err := fsatomic.Remove(file.path); err != nil {
			return fmt.Errorf("remove %s: %w", file.name, err)
		}
		if gateMergeCASAfterStepHook != nil {
			gateMergeCASAfterStepHook(repo, operation, file.name)
		}
		progress++
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if current, exact := gateMergeMetadataProgress(files, states, false); !exact || current != len(managed) {
		return errGateMergeMetadataMoved
	}
	return nil
}

func restoreGateMergeMetadata(
	repo string,
	files []gateMergeMetadataFile,
	progress int,
) error {
	managed := gateMergeManagedMetadata(files)
	for progress < len(managed) {
		states, err := readGateMergeMetadata(files)
		if err != nil {
			return err
		}
		current, exact := gateMergeMetadataProgress(files, states, true)
		if !exact || current != progress {
			return errGateMergeMetadataMoved
		}
		file := managed[progress]
		if err := atomicfile.Write(file.path, file.want, file.mode.Perm()); err != nil {
			return fmt.Errorf("restore %s: %w", file.name, err)
		}
		if gateMergeCASAfterStepHook != nil {
			gateMergeCASAfterStepHook(repo, "restore", file.name)
		}
		progress++
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if current, exact := gateMergeMetadataProgress(files, states, true); !exact || current != len(managed) {
		return errGateMergeMetadataMoved
	}
	return nil
}

func clearCommittedMergeState(repo string, intent gateCommitIntent) error {
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	err = withGateMergeMetadataLock(repo, fs.FileMode(intent.IndexMode), intent, func(indexPath string) error {
		if err := requireGateIntentKeepalive(repo, intent); err != nil {
			return err
		}
		if err := exactGateIndexState(indexPath, fs.FileMode(intent.IndexMode), intent.IndexAfter); err != nil {
			return err
		}
		for _, name := range gitOperationMarkers {
			path, err := gitMetadataPath(repo, name)
			if err != nil {
				return err
			}
			if _, err := os.Stat(path); err == nil {
				return fmt.Errorf("commit admission: Git operation %s appeared before final state cleanup", name)
			} else if !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
		states, err := readGateMergeMetadata(files)
		if err != nil {
			return err
		}
		progress, exact := gateMergeMetadataProgress(files, states, false)
		if !exact {
			return errGateMergeMetadataMoved
		}
		return removeGateMergeMetadata(repo, "cleanup", files, progress)
	})
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("commit admission: Git index moved before merge metadata cleanup")
	}
	if errors.Is(err, errGateMergeMetadataMoved) {
		return fmt.Errorf("commit admission: pending merge metadata moved before cleanup: %w", err)
	}
	return err
}

func gateIntentKeepaliveRefValue(repo, reference string) (string, bool, error) {
	var stdout, stderr bytes.Buffer
	process := newGateGitReaderCommand(repo, "rev-parse", "--verify", "--quiet", reference)
	process.Stdout, process.Stderr = &stdout, &stderr
	err := process.Run()
	if err == nil {
		object := strings.TrimSpace(stdout.String())
		if !validGitObjectID(object) {
			return "", false, errors.New("gate: keepalive reference resolved to an invalid object identity")
		}
		return object, true, nil
	}
	if exitError, exitFailure := errors.AsType[*exec.ExitError](err); exitFailure && exitError.ExitCode() == 1 && stdout.Len() == 0 {
		return "", false, nil
	}
	return "", false, fmt.Errorf(
		"gate: resolve interrupted commit keepalive reference: %w: %s",
		err, strings.TrimSpace(stderr.String()),
	)
}

func requireGateIntentKeepaliveObjects(repo string, intent gateCommitIntent) error {
	if _, err := gitAuthorityOutput(repo, "cat-file", "-e", intent.KeepaliveCommit+"^{commit}"); err != nil {
		return fmt.Errorf("gate: interrupted commit keepalive commit is absent: %w", err)
	}
	if _, err := gitAuthorityOutput(repo, "cat-file", "-e", intent.Commit+"^{commit}"); err != nil {
		return fmt.Errorf("gate: interrupted completion commit is absent: %w", err)
	}
	output, err := gitAuthorityOutput(
		repo, "rev-list", "--objects", "--missing=print", intent.KeepaliveTree,
	)
	if err != nil {
		return fmt.Errorf("gate: inspect interrupted commit keepalive objects: %w", err)
	}
	for line := range strings.SplitSeq(string(output), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "?") {
			return errors.New("gate: interrupted commit keepalive object closure is incomplete")
		}
	}
	return nil
}

func requireGateIntentKeepalive(repo string, intent gateCommitIntent) error {
	if intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) ||
		!validGitObjectID(intent.KeepaliveTree) || !validGitObjectID(intent.KeepaliveCommit) {
		return errors.New("gate: interrupted commit keepalive authority is invalid")
	}
	object, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
	if err != nil {
		return err
	}
	if !found || object != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive reference moved from its exact intent")
	}
	return requireGateIntentKeepaliveObjects(repo, intent)
}

func bindGateIntentKeepalive(repo string, intent gateCommitIntent) (bool, error) {
	expectedTree, err := buildGateIntentKeepaliveTree(repo, intent)
	if err != nil {
		return false, err
	}
	expectedCommit, err := buildGateIntentKeepaliveCommit(repo, intent, expectedTree)
	if err != nil {
		return false, err
	}
	if intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) ||
		intent.KeepaliveTree != expectedTree || intent.KeepaliveCommit != expectedCommit {
		return false, errors.New("gate: interrupted commit keepalive differs from its exact intent")
	}
	if err := requireGateIntentKeepaliveObjects(repo, intent); err != nil {
		return false, err
	}
	current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
	if err != nil {
		return false, err
	}
	if found {
		if current != intent.KeepaliveCommit {
			return false, errors.New("gate: interrupted commit keepalive reference is owned by another object")
		}
		return false, requireGateIntentKeepalive(repo, intent)
	}
	zero := strings.Repeat("0", len(intent.KeepaliveCommit))
	if _, err := gitAuthorityWriterOutput(
		repo, "update-ref", intent.KeepaliveRef, intent.KeepaliveCommit, zero,
	); err != nil {
		current, found, inspectErr := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
		if inspectErr == nil && found && current == intent.KeepaliveCommit {
			return false, requireGateIntentKeepalive(repo, intent)
		}
		return false, errors.Join(err, inspectErr)
	}
	return true, requireGateIntentKeepalive(repo, intent)
}

func ensureGateIntentKeepalive(repo string, intent gateCommitIntent) error {
	_, err := bindGateIntentKeepalive(repo, intent)
	if err != nil {
		return err
	}
	return nil
}

func deleteGateIntentKeepalive(repo string, intent gateCommitIntent) error {
	current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
	if err != nil || !found {
		return err
	}
	if current != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive reference moved before intent resolution")
	}
	if _, err := gitAuthorityWriterOutput(
		repo, "update-ref", "-d", intent.KeepaliveRef, intent.KeepaliveCommit,
	); err != nil {
		return err
	}
	if current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef); err != nil {
		return err
	} else if found {
		return fmt.Errorf(
			"gate: interrupted commit keepalive remains at %s after intent resolution", current,
		)
	}
	return nil
}

func writeGateCommitIntent(repo string, intent gateCommitIntent) error {
	if intent.KeepaliveRef == "" && intent.KeepaliveTree == "" && intent.KeepaliveCommit == "" {
		if err := initializeGateIntentKeepalive(repo, &intent); err != nil {
			return err
		}
	}
	if err := intent.validate(); err != nil {
		return err
	}
	if err := validateGateIntentIndexes(repo, intent); err != nil {
		return err
	}
	intentPath := filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))
	if _, err := os.Stat(intentPath); err == nil {
		var previous gateCommitIntent
		if err := readJSON(repo, gateCommitIntentFile, &previous); err != nil {
			return err
		}
		if err := previous.validate(); err != nil {
			return err
		}
		if previous.KeepaliveRef != intent.KeepaliveRef ||
			previous.KeepaliveTree != intent.KeepaliveTree ||
			previous.KeepaliveCommit != intent.KeepaliveCommit {
			return errors.New("gate: interrupted commit rewrite changed keepalive authority")
		}
		if err := ensureGateIntentKeepalive(repo, previous); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	_, err := bindGateIntentKeepalive(repo, intent)
	if err != nil {
		return err
	}
	if gateKeepaliveBoundHook != nil {
		gateKeepaliveBoundHook(repo, intent)
	}
	if err := writeJSON(repo, gateCommitIntentFile, intent, gatePrivateFileMode); err != nil {
		// Never delete the ref on an ambiguous publication failure: a post-rename
		// error or racing direct writer may have made the intent durable. A stale
		// exact ref is a bounded leak; a durable intent without it is unrecoverable.
		return err
	}
	return ensureGateIntentKeepalive(repo, intent)
}

func removeGateCommitIntent(repo string) error {
	path := filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return fsatomic.Remove(path)
	} else if err != nil {
		return err
	}
	var intent gateCommitIntent
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		return err
	}
	if err := intent.validate(); err != nil {
		return err
	}
	if current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef); err != nil {
		return err
	} else if !found || current != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive reference moved before intent resolution")
	}
	if err := fsatomic.Remove(path); err != nil {
		return err
	}
	return deleteGateIntentKeepalive(repo, intent)
}

func recoverInterruptedCommit(repo, storePath string) (recovered artifact.ID, err error) {
	recoveryStarted := time.Now()
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err == nil {
		return artifact.ID{}, errors.New("gate: record debt exists for the interrupted commit; run `go run ./cmd/gate -reconcile`")
	} else if !errors.Is(err, os.ErrNotExist) {
		return artifact.ID{}, err
	}
	if err := gitauthority.RequireCompleteHistory(context.Background(), repo); err != nil {
		return artifact.ID{}, fmt.Errorf("gate: %w", err)
	}
	var intent gateCommitIntent
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		return artifact.ID{}, fmt.Errorf("read interrupted commit intent: %w", err)
	}
	if err := intent.validate(); err != nil {
		return artifact.ID{}, err
	}
	if err := validateGateIntentIndexes(repo, intent); err != nil {
		return artifact.ID{}, err
	}
	// The reachability ref is published before the intent, so a crash can leave
	// only a harmless orphan ref, never a durable intent whose rollback objects
	// were exposed to pruning. Requiring it here rejects any later ref drift;
	// every destructive transition below repeats that check under lock.
	if err := ensureGateIntentKeepalive(repo, intent); err != nil {
		return artifact.ID{}, err
	}
	defer func() {
		if err == nil && recovered.Valid() {
			err = finalizeMatchingHeartbeat(repo, recovered)
		}
	}()
	headReference, err := currentHeadReference(repo)
	if err != nil {
		return artifact.ID{}, err
	}
	if headReference != intent.HeadReference {
		return artifact.ID{}, fmt.Errorf(
			"gate: HEAD reference moved from %s to %s; automatic recovery refused",
			intent.HeadReference, headReference,
		)
	}
	head, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return artifact.ID{}, err
	}
	head = strings.TrimSpace(head)
	if intent.Commit == "" && head != intent.Parent {
		return artifact.ID{}, errors.New(
			"gate: HEAD moved before the interrupted intent durably named its commit; automatic recovery refused",
		)
	}
	if intent.Commit != "" {
		if err := validateInterruptedCommitIdentity(repo, intent); err != nil {
			return artifact.ID{}, err
		}
	}
	if head != intent.Parent && head != intent.Commit {
		return artifact.ID{}, fmt.Errorf(
			"gate: HEAD %.12s moved beyond interrupted commit %.12s; automatic recovery refused",
			head, intent.Commit,
		)
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	ctx := context.Background()
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, intent.Preparation)
	if err != nil || preparation.State != runrecord.GatePrepared {
		return artifact.ID{}, errors.Join(errors.New("gate: interrupted preparation is unavailable"), err)
	}
	finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return artifact.ID{}, err
	}
	if finalized {
		switch finalization.Outcome {
		case runrecord.OutcomeSucceeded:
			if intent.Commit == "" || finalization.CodeCommit != intent.Commit {
				return artifact.ID{}, errors.New("gate: successful interrupted finalization contradicts Git")
			}
			if head == intent.Parent {
				planRestored, planErr := planBytesMatch(repo, intent.Plan)
				if planErr != nil {
					return artifact.ID{}, planErr
				}
				if planRestored {
					if err := removeGateCommitIntent(repo); err != nil {
						return artifact.ID{}, err
					}
					return preparation.ID, nil
				}
			}
			if head == intent.Commit {
				if err := validateInterruptedCommitAuthority(repo, intent, store); err != nil {
					return artifact.ID{}, err
				}
				complete, completionErr := interruptedCompletionRecorded(ctx, store, intent)
				if completionErr != nil {
					return artifact.ID{}, completionErr
				}
				if complete {
					if err := clearCommittedMergeState(repo, intent); err != nil {
						return artifact.ID{}, err
					}
					if err := removeGateCommitIntent(repo); err != nil {
						return artifact.ID{}, err
					}
					return preparation.ID, nil
				}
			}
			// The store is append-only, so an incomplete success record cannot
			// be erased. Keep it unreachable by rolling Git and the plan back;
			// no cancellation is added because that would make the preparation's
			// finalization lineage ambiguous.
		case runrecord.OutcomeCancelled:
			planRestored, planErr := planBytesMatch(repo, intent.Plan)
			if planErr != nil {
				return artifact.ID{}, planErr
			}
			if head != intent.Parent || !planRestored {
				return artifact.ID{}, errors.New("gate: interrupted cancellation contradicts Git or the restored plan")
			}
			complete, completionErr := interruptedCancellationRecorded(ctx, store, intent, preparation, finalization)
			if completionErr != nil {
				return artifact.ID{}, completionErr
			}
			if !complete {
				return artifact.ID{}, errors.New("gate: interrupted cancellation is not the exact recovery record")
			}
			if err := removeGateCommitIntent(repo); err != nil {
				return artifact.ID{}, err
			}
			return preparation.ID, nil
		default:
			return artifact.ID{}, errors.New("gate: interrupted preparation already has a non-recoverable finalization")
		}
	}
	if err := restoreInterruptedState(repo, intent, head); err != nil {
		return artifact.ID{}, err
	}
	if !finalized {
		codeCommit := intent.Parent
		if intent.Commit != "" {
			codeCommit = intent.Commit
		}
		durationNS, err := completedGateMeasurement(recoveryStarted, "interrupted recovery")
		if err != nil {
			return artifact.ID{}, err
		}
		record, err := runrecord.NewGateRecord(
			intent.Recipe, preparation.Environment, codeCommit, runrecord.OutcomeCancelled, "", durationNS,
			[]runrecord.GateStep{{
				Name: "recovery", Phase: runrecord.PhaseValidate,
				Outcome: runrecord.StepCancelled, DurationNS: durationNS,
			}},
		)
		if err != nil {
			return artifact.ID{}, err
		}
		batch, err := record.Batch("gate/recovered/" + preparation.ID.String())
		if err != nil {
			return artifact.ID{}, err
		}
		finalized, err := runrecord.NewGateFinalization(preparation, codeCommit, record.Result.ID, runrecord.OutcomeCancelled)
		if err != nil {
			return artifact.ID{}, err
		}
		finalizedContent, err := finalized.Content()
		if err != nil {
			return artifact.ID{}, err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: intent.Recipe})
		batch.Contents = append(batch.Contents, finalizedContent)
		batch.Lineage = append(batch.Lineage, finalized.Lineage()...)
		appendGateFinalizationAlias(&batch, preparation.ID, finalized.ID)
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			return artifact.ID{}, err
		}
	}
	if err := removeGateCommitIntent(repo); err != nil {
		return artifact.ID{}, err
	}
	return preparation.ID, nil
}

func interruptedCompletionRecorded(
	ctx context.Context,
	store *overgodb.Store,
	intent gateCommitIntent,
) (bool, error) {
	introduction, found, err := store.ArtifactIntroduction(ctx, intent.Preparation)
	if err != nil {
		return false, err
	}
	if !found || introduction.Commit != intent.PreparationCommit {
		return false, nil
	}
	attempts, err := runrecord.AttemptsForPreparation(ctx, store, intent.Preparation)
	if err != nil {
		return false, err
	}
	item, step, _ := strings.Cut(intent.PlanRef, "/")
	var matches []runrecord.AttemptRecord
	for _, attempt := range attempts {
		if attempt.PlanItem == item && attempt.PlanStep == step && attempt.Recipe == intent.Recipe &&
			attempt.CandidateManifest == intent.CandidateManifest && attempt.CodeCommit == intent.Commit &&
			attempt.Outcome == runrecord.OutcomeSucceeded {
			matches = append(matches, attempt)
		}
	}
	if len(matches) > 1 {
		return false, errors.New("gate: interrupted completion has ambiguous successful attempts")
	}
	if len(matches) == 0 {
		return false, nil
	}
	verification, err := runrecord.VerifyAttemptGate(ctx, store, matches[0])
	if err != nil || verification.Preparation.ID != intent.Preparation {
		return false, nil
	}
	document, err := plan.Parse(intent.Plan)
	if err != nil {
		return false, err
	}
	verify := ""
	for _, candidateItem := range document.Items {
		if candidateItem.ID != item {
			continue
		}
		for _, candidateStep := range candidateItem.Steps {
			if candidateStep.ID == step && candidateStep.Status == plan.StatusOpen {
				verify = candidateStep.Verify
				break
			}
		}
	}
	if verify == "" {
		return false, errors.New("gate: interrupted completion row is absent from its recorded plan")
	}
	acceptance := []runrecord.GateStep{}
	for _, gateStep := range verification.Gate.Steps {
		if gateStep.Name == "acceptance" &&
			(gateStep.Outcome == runrecord.StepSucceeded || gateStep.Outcome == runrecord.StepReused) {
			acceptance = append(acceptance, gateStep)
		}
	}
	if len(acceptance) != 1 {
		return false, nil
	}
	if err := runrecord.VerifyCompletionAcceptanceEvidence(
		acceptance[0].Evidence, intent.PlanRef, verify,
	); err != nil {
		return false, nil
	}
	return true, nil
}

func planBytesMatch(repo string, expected []byte) (bool, error) {
	current, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(plan.Path)))
	if err != nil {
		return false, err
	}
	return bytes.Equal(current, expected), nil
}

func interruptedCancellationRecorded(
	ctx context.Context,
	store *overgodb.Store,
	intent gateCommitIntent,
	preparation, finalization runrecord.GateLifecycle,
) (bool, error) {
	if finalization.Result == nil || finalization.Preparation == nil ||
		*finalization.Preparation != preparation.ID {
		return false, nil
	}
	codeCommit := intent.Parent
	if intent.Commit != "" {
		codeCommit = intent.Commit
	}
	if finalization.CodeCommit != codeCommit {
		return false, nil
	}
	gate, err := runrecord.RequireGateResult(ctx, store, *finalization.Result)
	if err != nil {
		return false, nil
	}
	if gate.Recipe != intent.Recipe || gate.Environment != preparation.Environment ||
		gate.CodeCommit != codeCommit || gate.Outcome != runrecord.OutcomeCancelled || gate.Failure != "" ||
		len(gate.Steps) != 1 || gate.Steps[0].Name != "recovery" ||
		gate.Steps[0].Phase != runrecord.PhaseValidate || gate.Steps[0].Outcome != runrecord.StepCancelled {
		return false, nil
	}
	gateIntroduction, gateFound, err := store.ArtifactIntroduction(ctx, gate.ID)
	if err != nil || !gateFound {
		return false, err
	}
	finalIntroduction, finalFound, err := store.ArtifactIntroduction(ctx, finalization.ID)
	if err != nil || !finalFound {
		return false, err
	}
	return gateIntroduction.Commit == finalIntroduction.Commit &&
		gateIntroduction.Sequence == finalIntroduction.Sequence, nil
}

func validateInterruptedCommitAuthority(repo string, intent gateCommitIntent, store *overgodb.Store) error {
	if err := gitauthority.RequireCompleteHistory(context.Background(), repo); err != nil {
		return fmt.Errorf("gate: %w", err)
	}
	if err := validateInterruptedCommitIdentity(repo, intent); err != nil {
		return err
	}
	messageBytes, err := gitAuthorityOutput(repo, "show", "-s", "--format=%B", intent.Commit)
	if err != nil {
		return err
	}
	message := string(messageBytes)
	preAdvance, err := plan.Parse(intent.Plan)
	if err != nil {
		return err
	}
	child, err := plan.Parse(intent.AdvancedPlan)
	if err != nil {
		return err
	}
	item, step, found := strings.Cut(intent.PlanRef, "/")
	if !found {
		return errors.New("gate: interrupted commit has an invalid completion reference")
	}
	mergeAuthority, err := projectedMergeAuthorityFromIntent(intent)
	if err != nil {
		return err
	}
	mergeAuthorityID := artifact.ID{}
	if mergeAuthority != nil {
		mergeAuthorityID = mergeAuthority.ID
	}
	if err := plan.VerifyCompletionCommitMessageWithMergeAuthority(
		message, preAdvance, item, step, intent.Recipe, intent.CandidateManifest,
		intent.Preparation, intent.PreparationCommit, intent.PlanProjection, mergeAuthorityID,
	); err != nil {
		return fmt.Errorf("gate: interrupted commit has the wrong completion authority: %w", err)
	}
	if err := verifyProspectiveGateCompletion(repo, intent, preAdvance, child, message, store); err != nil {
		return fmt.Errorf("gate: interrupted commit has the wrong completion authority: %w", err)
	}
	return nil
}

func validateInterruptedCommitIdentity(repo string, intent gateCommitIntent) error {
	parentsOutput, err := gitAuthorityOutput(repo, "show", "-s", "--format=%P", intent.Commit)
	if err != nil {
		return err
	}
	parentFields := strings.Fields(string(parentsOutput))
	wantParents := []string{intent.Parent}
	if intent.Merge != nil {
		wantParents = append(wantParents, intent.Merge.parents()...)
	}
	if !slices.Equal(parentFields, wantParents) {
		return errors.New("gate: interrupted commit does not descend from its recorded parent")
	}
	treeOutput, err := gitAuthorityOutput(repo, "rev-parse", intent.Commit+"^{tree}")
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(treeOutput)) != intent.Tree {
		return errors.New("gate: interrupted commit tree differs from its write-ahead intent")
	}
	planBytes, err := gitCompletionFile(repo, intent.Commit, plan.Path)
	if err != nil {
		return err
	}
	if !bytes.Equal(planBytes, intent.AdvancedPlan) {
		return errors.New("gate: interrupted commit plan differs from its write-ahead transition")
	}
	return nil
}

func gitCompletionFile(repo, revision, path string) ([]byte, error) {
	return gitAuthorityOutput(repo, "show", revision+":"+path)
}

func restoreInterruptedParentIndexLocked(repo string, intent gateCommitIntent, indexPath string) error {
	if err := normalizeGateIndexToStates(
		repo, indexPath, fs.FileMode(intent.IndexMode),
		intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
	); err != nil {
		return err
	}
	if err := replaceGateIndexStatesUnderLock(
		indexPath,
		[][]byte{intent.IndexAfter},
		[][]byte{intent.IndexBefore, intent.IndexRestore},
		intent.IndexRestore,
		fs.FileMode(intent.IndexMode),
	); err != nil {
		return fmt.Errorf("gate: restore interrupted parent index: %w", err)
	}
	return nil
}

// restoreInterruptedState holds Git's actual index lock and the AUTO_MERGE
// pseudoref lock across the complete branch/index/plan/merge-state recovery.
// The admission check is repeated only after both locks are held, and no
// cancellation authority is recorded until the final exact-state check passes.
// The seven manually owned lock paths carry exact intent markers and are
// reclaimed under the process guard after abrupt death. Git's update-ref child
// separately owns branch and HEAD locks: ordinary parent death closes its pipe
// and makes Git abort them, while machine failure can still leave those
// indistinguishable Git locks requiring conventional manual cleanup.
func restoreInterruptedState(repo string, intent gateCommitIntent, observedHead string) error {
	err := withGateGitStateLock(
		repo,
		fs.FileMode(intent.IndexMode),
		intent,
		gateRecoveryBeforeLockHook,
		gateRecoveryLockedHook,
		func(indexPath string) (err error) {
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if err := requireRecoverableGitState(repo, intent, observedHead); err != nil {
				return err
			}
			if !exactGateHead(repo, intent.HeadReference, observedHead) {
				return errors.New("gate: interrupted HEAD changed before prepared state recovery")
			}
			transaction, err := prepareGateReferenceTransaction(
				repo, intent.HeadReference, intent.Parent, observedHead, "overgo gate recovery",
			)
			if err != nil {
				return fmt.Errorf("gate: prepare interrupted state rollback: %w", err)
			}
			defer func() { err = errors.Join(err, transaction.abort()) }()
			if !exactGateHead(repo, intent.HeadReference, observedHead) {
				return errors.New("gate: interrupted HEAD moved while preparing state recovery")
			}
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "admission")
			}
			// RecoverSwap may restore a detached plan pathname and retire its
			// exact phase evidence. Do that only after the prepared Git transaction
			// has locked the captured branch and HEAD referent.
			observedPlan, err := requireRecoverablePlan(repo, intent, observedHead)
			if err != nil {
				return err
			}
			// Publish the parent before changing the index, plan, or merge
			// metadata. Each of those exact write-ahead states is recoverable
			// with HEAD at the parent, while publishing the parent last would
			// leave a crash window where the restored index appears as foreign
			// staged work against the completion commit. Reacquire a verify-only
			// transaction before the first state mutation so a competing branch
			// writer either wins this mutation-free handoff or remains locked out
			// through the rest of recovery.
			if observedHead != intent.Parent {
				if err := transaction.commit(); err != nil {
					return err
				}
				if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
					return errors.New("gate: prepared recovery published inexact parent authority")
				}
				if gateRecoveryAfterStepHook != nil {
					gateRecoveryAfterStepHook(repo, "publication")
				}
				parentGuard, err := prepareGateReferenceTransaction(
					repo, intent.HeadReference, intent.Parent, intent.Parent, "overgo gate recovery guard",
				)
				if err != nil {
					return fmt.Errorf("gate: guard restored parent authority: %w", err)
				}
				transaction = parentGuard
				if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
					return errors.New("gate: restored parent moved while preparing state recovery")
				}
			}
			if err := restoreInterruptedParentIndexLocked(repo, intent, indexPath); err != nil {
				return err
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "parent")
			}
			planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
			if gatePlanRecoveryBeforeSwapHook != nil {
				gatePlanRecoveryBeforeSwapHook(planPath)
			}
			if err := restoreInterruptedPlan(repo, planPath, observedPlan, intent); err != nil {
				return fmt.Errorf("gate: restore interrupted plan: %w", err)
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "plan")
			}
			if err := restoreInterruptedMergeLocked(repo, intent, indexPath); err != nil {
				return err
			}
			if gateRecoveryAfterStepHook != nil {
				gateRecoveryAfterStepHook(repo, "merge")
			}
			if err := requireExactRecoveredGitStateExceptHead(repo, intent, indexPath); err != nil {
				return err
			}
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("gate: prepared recovery branch moved before state recovery completed")
			}
			if err := transaction.commit(); err != nil {
				return err
			}
			return requireExactRecoveredGitState(repo, intent, indexPath)
		},
	)
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("gate: Git index moved before exact interrupted-state recovery")
	}
	if errors.Is(err, errGateMergeMetadataMoved) {
		return fmt.Errorf("gate: merge metadata moved before exact interrupted-state recovery: %w", err)
	}
	return err
}

func requireExactRecoveredGitState(repo string, intent gateCommitIntent, indexPath string) error {
	if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
		return errors.New("gate: HEAD moved before interrupted-state recovery completed")
	}
	return requireExactRecoveredGitStateExceptHead(repo, intent, indexPath)
}

func requireExactRecoveredGitStateExceptHead(repo string, intent gateCommitIntent, indexPath string) error {
	if err := normalizeGateIndexToStates(
		repo, indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore, intent.IndexRestore,
	); err != nil {
		return err
	}
	if err := exactGateIndexState(
		indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore, intent.IndexRestore,
	); err != nil {
		return err
	}
	if matches, err := planBytesMatch(repo, intent.Plan); err != nil {
		return err
	} else if !matches {
		return errors.New("gate: plan moved before interrupted-state recovery completed")
	}
	for _, name := range gitOperationMarkers {
		path, err := gitMetadataPath(repo, name)
		if err != nil {
			return err
		}
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("gate: Git operation %s appeared during interrupted-state recovery", name)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	present := intent.Merge != nil
	progress, exact := gateMergeMetadataProgress(files, states, present)
	if !exact || progress != len(gateMergeManagedMetadata(files)) {
		return errors.New("gate: merge metadata is not in its exact recovered state")
	}
	return nil
}

func gitAuthorityOutput(repo string, arguments ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitReaderCommand(repo, arguments...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gate: git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func gitAuthorityWriterOutput(repo string, arguments ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitWriterCommand(repo, nil, arguments...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gate: git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func projectedMergeAuthorityFromIntent(
	intent gateCommitIntent,
) (*plan.FirstParentTargetMergeAuthority, error) {
	if intent.PlanProjection != plan.MergeProjectionFirstParentTarget {
		if len(intent.MergeAuthority) != 0 {
			return nil, errors.New("gate: semantic-union intent carries projected merge authority")
		}
		return nil, nil
	}
	receipt, err := plan.ParseFirstParentTargetMergeAuthority(intent.MergeAuthority)
	if err != nil {
		return nil, fmt.Errorf("gate: parse projected merge authority: %w", err)
	}
	return &receipt, nil
}

func verifyProspectiveGateCompletion(
	repo string,
	intent gateCommitIntent,
	preAdvance, child plan.Plan,
	message string,
	store *overgodb.Store,
) error {
	mergeAuthority, err := projectedMergeAuthorityFromIntent(intent)
	if err != nil {
		return err
	}
	parentIDs := []string{intent.Parent}
	mergeParent := ""
	if intent.Merge != nil {
		for _, parentID := range intent.Merge.parents() {
			if mergeParent != "" {
				return errors.New("gate: completion merge requires one merge parent")
			}
			mergeParent = parentID
		}
		if !validGitObjectID(mergeParent) || mergeParent == intent.Parent {
			return errors.New("gate: completion merge requires one distinct merge parent")
		}
		parentIDs = append(parentIDs, mergeParent)
	}
	parents := make([]plan.Plan, 0, len(parentIDs))
	for _, parentID := range parentIDs {
		data, err := gitCompletionFile(repo, parentID, plan.Path)
		if err != nil {
			return err
		}
		document, err := plan.ParseHistorical(data)
		if err != nil {
			return fmt.Errorf("gate: parse parent plan %.12s: %w", parentID, err)
		}
		parents = append(parents, document)
	}
	var mergeBase *plan.Plan
	if intent.Merge != nil {
		baseOutput, err := gitAuthorityOutput(repo, "merge-base", "--all", intent.Parent, mergeParent)
		if err != nil {
			return err
		}
		bases := strings.Fields(string(baseOutput))
		if len(bases) != 1 {
			return fmt.Errorf("gate: completion merge requires one merge base, found %d", len(bases))
		}
		data, err := gitCompletionFile(repo, bases[0], plan.Path)
		if err != nil {
			return err
		}
		base, err := plan.ParseHistorical(data)
		if err != nil {
			return fmt.Errorf("gate: parse merge-base plan %.12s: %w", bases[0], err)
		}
		mergeBase = &base
		if store == nil {
			return errors.New("gate: completion merge requires the locked authority store")
		}
		if intent.PlanProjection == plan.MergeProjectionFirstParentTarget {
			if mergeAuthority == nil {
				return errors.New("gate: first-parent-target completion lacks projected merge authority")
			}
			item, step, found := strings.Cut(intent.PlanRef, "/")
			if !found {
				return errors.New("gate: projected completion has an invalid plan reference")
			}
			if err := plan.VerifyFirstParentTargetMergeAuthorityTransition(
				intent.Parent, mergeParent, bases[0], parents[0], parents[1], base,
				preAdvance, child, item, step, intent.Preparation, intent.PreparationCommit,
				*mergeAuthority,
			); err != nil {
				return fmt.Errorf("gate: audit projected merge receipt: %w", err)
			}
			localAuthority, err := plan.ResolveCompletionAuthority(
				context.Background(), repo, intent.Parent, parents[0], store,
			)
			if err != nil {
				return fmt.Errorf("gate: audit local completion merge parent %.12s: %w", intent.Parent, err)
			}
			if !localAuthority.ProtectsRevision() {
				return fmt.Errorf("gate: local completion merge parent %.12s is outside the protected epoch", intent.Parent)
			}
			if err := plan.VerifyFirstParentTargetLocalAuthority(
				repo, intent.Parent, parents[0], preAdvance, child, localAuthority,
			); err != nil {
				return fmt.Errorf("gate: audit first-parent target authority: %w", err)
			}
			return plan.VerifyProspectiveCompletionTransitionWithProjection(
				parents, mergeBase, preAdvance, child, message, intent.PlanProjection,
			)
		}
		parentAuthorities := make([]plan.CompletionAuthority, len(parentIDs))
		for index, parentID := range parentIDs {
			authority, err := plan.ResolveCompletionAuthority(
				context.Background(), repo, parentID, parents[index], store,
			)
			if err != nil {
				return fmt.Errorf("gate: audit completion merge parent %.12s: %w", parentID, err)
			}
			if !authority.ProtectsRevision() {
				return fmt.Errorf(
					"gate: completion merge parent %.12s is outside the protected epoch; rebase the merge source",
					parentID,
				)
			}
			parentAuthorities[index] = authority
		}
		baseAuthority, err := plan.ResolveCompletionAuthority(
			context.Background(), repo, bases[0], base, store,
		)
		if err != nil {
			return fmt.Errorf("gate: audit completion merge base %.12s: %w", bases[0], err)
		}
		if !baseAuthority.ProtectsRevision() {
			return errors.New("gate: completion merge parents do not share a protected epoch; rebase the merge source")
		}
		if err := plan.VerifyProspectiveMergeAuthorityWithProjection(
			repo, parentIDs[0], parentIDs[1],
			parents[0], parents[1], preAdvance,
			parentAuthorities[0], parentAuthorities[1],
			intent.PlanProjection,
		); err != nil {
			return fmt.Errorf("gate: audit prospective completion merge: %w", err)
		}
	}
	return plan.VerifyProspectiveCompletionTransitionWithProjection(
		parents, mergeBase, preAdvance, child, message, intent.PlanProjection,
	)
}

func requireRecoverablePlan(repo string, intent gateCommitIntent, head string) ([]byte, error) {
	planPath := filepath.Join(repo, filepath.FromSlash(plan.Path))
	scratch, err := gatePlanScratchPath(repo, intent.Preparation)
	if err != nil {
		return nil, err
	}
	current, err := atomicfile.RecoverSwap(
		planPath, scratch, intent.Plan, intent.AdvancedPlan, fs.FileMode(intent.PlanMode),
	)
	if err != nil {
		return nil, fmt.Errorf("gate: resolve interrupted plan publication: %w", err)
	}
	if !bytes.Equal(intent.Plan, current) && !bytes.Equal(intent.AdvancedPlan, current) {
		return nil, errors.New("gate: plan moved after the interrupted commit; automatic recovery refused")
	}
	if head == intent.Commit {
		status, err := gitAuthorityOutput(
			repo, "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=no",
		)
		if err != nil {
			return nil, err
		}
		dirty, err := repoanalysis.ParseDirtyStatus(status)
		if err != nil {
			return nil, err
		}
		prePublicationPlan := bytes.Equal(intent.Plan, current)
		exactPrePublicationDelta := prePublicationPlan && len(dirty) == 1 &&
			dirty[0].Path == plan.Path && dirty[0].OriginalPath == "" &&
			dirty[0].IndexStatus == " " && dirty[0].WorktreeStatus == "M"
		if len(dirty) != 0 && !exactPrePublicationDelta {
			return nil, errors.New("gate: tracked worktree changes appeared after the interrupted commit")
		}
	}
	return current, nil
}

func restoreInterruptedPlan(repo, path string, observed []byte, intent gateCommitIntent) error {
	if bytes.Equal(observed, intent.Plan) {
		return nil
	}
	scratch, err := gatePlanScratchPath(repo, intent.Preparation)
	if err != nil {
		return err
	}
	return atomicfile.CompareAndSwap(
		path, scratch, observed, intent.Plan, fs.FileMode(intent.PlanMode),
	)
}

var gitOperationMarkers = [...]string{
	"MERGE_AUTOSTASH", "SQUASH_MSG", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-apply", "rebase-merge", "sequencer",
}

// normalizeGateIndexToStates restores the exact write-ahead index bytes when
// a concurrent read-only observer — a `git status` from another session —
// has opportunistically refreshed the stat cache. A refresh rewrites entry
// timestamps but never the tree the index resolves to, so when the live
// bytes differ from every captured state yet write-tree to the same tree
// authority as one of them, that state's exact bytes are restored under the
// held gate lock; an index whose tree matches no captured state is genuine
// drift and stays untouched for the byte comparison to refuse.
func normalizeGateIndexToStates(repo, indexPath string, mode fs.FileMode, states ...[]byte) error {
	current, err := gateIndexState(indexPath, mode)
	if err != nil {
		return err
	}
	for _, state := range states {
		if bytes.Equal(current, state) {
			return nil
		}
	}
	currentTree, err := indexTreeForBytes(repo, current)
	if err != nil {
		return err
	}
	for _, state := range states {
		stateTree, err := indexTreeForBytes(repo, state)
		if err != nil {
			return err
		}
		if stateTree == currentTree {
			if err := atomicfile.Write(indexPath, state, mode.Perm()); err != nil {
				return err
			}
			return exactGateIndexState(indexPath, mode, state)
		}
	}
	return nil
}

func requireRecoverableGitState(repo string, intent gateCommitIntent, head string) error {
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		return err
	}
	if err := normalizeGateIndexToStates(
		repo, indexPath, fs.FileMode(intent.IndexMode),
		intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
	); err != nil {
		return err
	}
	indexBytes, err := gateIndexState(indexPath, fs.FileMode(intent.IndexMode))
	if err != nil {
		return err
	}
	if !bytes.Equal(indexBytes, intent.IndexBefore) && !bytes.Equal(indexBytes, intent.IndexAfter) &&
		!bytes.Equal(indexBytes, intent.IndexRestore) {
		return errors.New("gate: Git index moved beyond the write-ahead states; automatic recovery refused")
	}
	for _, name := range gitOperationMarkers {
		path, err := gitMetadataPath(repo, name)
		if err != nil {
			return err
		}
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("gate: Git operation %s appeared after the interrupted commit", name)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if intent.Merge == nil {
		if _, exact := gateMergeMetadataProgress(files, states, false); !exact {
			return errors.New("gate: unrelated merge metadata appeared after the interrupted commit")
		}
		return nil
	}
	if _, exact := gateMergeMetadataProgress(files, states, false); exact {
		return nil
	}
	if _, exact := gateMergeMetadataProgress(files, states, true); exact {
		return nil
	}
	return errors.New("gate: current merge metadata differs from the interrupted merge")
}

func restoreInterruptedMergeLocked(repo string, intent gateCommitIntent, indexPath string) error {
	if intent.Merge == nil {
		return nil
	}
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	if err := exactGateIndexState(
		indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore, intent.IndexRestore,
	); err != nil {
		return err
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if progress, exact := gateMergeMetadataProgress(files, states, true); exact {
		return restoreGateMergeMetadata(repo, files, progress)
	}
	// A crash during committed-state cleanup leaves an exact absent prefix.
	// Finish that cleanup (including removing MERGE_HEAD last), then restore
	// MODE, MSG, and AUTO_MERGE before publishing MERGE_HEAD.
	progress, exact := gateMergeMetadataProgress(files, states, false)
	if !exact {
		return errGateMergeMetadataMoved
	}
	if err := removeGateMergeMetadata(repo, "recovery-cleanup", files, progress); err != nil {
		return err
	}
	var restoreProgress int
	return restoreGateMergeMetadata(repo, files, restoreProgress)
}

func (g *gateContext) oweRecord(batch artifact.Batch, cause error) error {
	debt := gateDebtEnvelope{Version: artifact.InitialDocumentVersion, Preparation: g.preparation.ID, Batch: batch}
	if err := validateGateDebt(debt); err != nil {
		return fmt.Errorf("%w; invalid record debt: %v", cause, err)
	}
	if err := writeJSON(g.repo, gateDebtFile, debt, gatePrivateFileMode); err != nil {
		return fmt.Errorf("%w; persist record debt: %v", cause, err)
	}
	return cause
}

func reconcileGateDebt(repo, storePath string) (artifact.ID, error) {
	var debt gateDebtEnvelope
	if err := readJSON(repo, gateDebtFile, &debt); err != nil {
		return artifact.ID{}, fmt.Errorf("read record debt: %w", err)
	}
	finalization, err := gateDebtFinalization(debt)
	if err != nil {
		return artifact.ID{}, err
	}
	var successIntent *gateCommitIntent
	if finalization.Outcome == runrecord.OutcomeSucceeded {
		intent, found, intentErr := readGateCommitIntentForPreparation(repo, debt.Preparation)
		if intentErr != nil {
			return artifact.ID{}, intentErr
		}
		if !found || intent.Commit == "" {
			return artifact.ID{}, errors.New("gate: successful record debt requires its completed Git intent")
		}
		headReference, headErr := currentHeadReference(repo)
		if headErr != nil {
			return artifact.ID{}, headErr
		}
		head, headErr := command(repo, "git", "rev-parse", "HEAD")
		if headErr != nil {
			return artifact.ID{}, headErr
		}
		if headReference != intent.HeadReference || strings.TrimSpace(head) != intent.Commit {
			return artifact.ID{}, errors.New("gate: successful record debt does not name the current Git commit")
		}
		successIntent = &intent
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	ctx := context.Background()
	if err := requireGateDebtAdmission(ctx, store, debt.Preparation, finalization); err != nil {
		return artifact.ID{}, errors.Join(err, store.Close())
	}
	if successIntent != nil {
		if err := validateInterruptedCommitAuthority(repo, *successIntent, store); err != nil {
			return artifact.ID{}, errors.Join(err, store.Close())
		}
	}
	if _, err := store.Commit(ctx, debt.Batch); err != nil {
		return artifact.ID{}, errors.Join(err, store.Close())
	}
	if err := store.Close(); err != nil {
		return artifact.ID{}, err
	}
	if err := fsatomic.Remove(filepath.Join(repo, filepath.FromSlash(gateDebtFile))); err != nil {
		return artifact.ID{}, err
	}
	if err := finalizeMatchingHeartbeat(repo, debt.Preparation); err != nil {
		return artifact.ID{}, err
	}
	if successIntent != nil {
		if _, err := recoverInterruptedCommit(repo, storePath); err != nil {
			return debt.Preparation, err
		}
		head, err := command(repo, "git", "rev-parse", "HEAD")
		if err != nil {
			return debt.Preparation, err
		}
		switch strings.TrimSpace(head) {
		case successIntent.Commit:
			return debt.Preparation, nil
		case successIntent.Parent:
			return debt.Preparation, errors.New("gate: reconciled success lacked exact completion authority; commit was rolled back")
		default:
			return debt.Preparation, errors.New("gate: debt recovery left HEAD at an unknown commit")
		}
	}
	if err := removeGateCommitIntentForPreparation(repo, debt.Preparation); err != nil {
		return artifact.ID{}, err
	}
	return debt.Preparation, nil
}

func removeGateCommitIntentForPreparation(repo string, preparation artifact.ID) error {
	_, found, err := readGateCommitIntentForPreparation(repo, preparation)
	if err != nil || !found {
		return err
	}
	return removeGateCommitIntent(repo)
}

func readGateCommitIntentForPreparation(
	repo string,
	preparation artifact.ID,
) (gateCommitIntent, bool, error) {
	path := filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return gateCommitIntent{}, false, nil
	} else if err != nil {
		return gateCommitIntent{}, false, err
	}
	var intent gateCommitIntent
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		return gateCommitIntent{}, false, err
	}
	if err := intent.validate(); err != nil {
		return gateCommitIntent{}, false, err
	}
	if err := validateGateIntentIndexes(repo, intent); err != nil {
		return gateCommitIntent{}, false, err
	}
	if intent.Preparation != preparation {
		return gateCommitIntent{}, false, errors.New("gate: record debt and interrupted commit name different preparations")
	}
	return intent, true, nil
}

type gateLifecycleRecoveryCensus struct {
	Head               artifact.CommitID
	Outstanding        runrecord.GateLifecycle
	OutstandingCount   int
	OutstandingDebt    []runrecord.GateLifecycle
	NewestFinalization runrecord.GateLifecycle
	HasFinalization    bool
}

func inspectGateLifecycleRecoveryCensus(
	ctx context.Context,
	store *overgodb.Store,
) (gateLifecycleRecoveryCensus, error) {
	if ctx == nil || store == nil {
		return gateLifecycleRecoveryCensus{}, errors.New("gate: lifecycle recovery requires the canonical store")
	}
	census := gateLifecycleRecoveryCensus{}
	census.Head, _ = store.Head()
	lifecycles, err := runrecord.GateLifecyclesInStore(ctx, store)
	if err != nil {
		return gateLifecycleRecoveryCensus{}, err
	}
	preparations := make(map[artifact.ID]runrecord.GateLifecycle)
	var finalizations []runrecord.GateLifecycle
	for _, lifecycle := range lifecycles {
		if err := lifecycle.ValidateIdentity(); err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: validate lifecycle %s during recovery: %w", lifecycle.ID, err,
			)
		}
		if lifecycle.State == runrecord.GatePrepared {
			preparations[lifecycle.ID] = lifecycle
			continue
		}
		finalizations = append(finalizations, lifecycle)
	}
	validatedFinalizations := make(map[artifact.ID]artifact.ID)
	for _, finalization := range finalizations {
		preparation, found := preparations[*finalization.Preparation]
		if !found {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: finalization %s lacks its exact preparation during recovery", finalization.ID,
			)
		}
		canonical, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
		if err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: validate lifecycle %s during recovery: %w", preparation.ID, err,
			)
		}
		if !found || canonical.ID != finalization.ID {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: finalization %s is not the sole canonical finalization for preparation %s",
				finalization.ID, preparation.ID,
			)
		}
		if err := validateCompleteGateFinalization(ctx, store, preparation, finalization); err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: validate lifecycle %s during recovery: %w", preparation.ID, err,
			)
		}
		validatedFinalizations[preparation.ID] = finalization.ID
		// GateLifecyclesInStore owns the deterministic durable order, including
		// its artifact-ID tie-break for documents co-introduced by legacy import.
		census.NewestFinalization = finalization
		census.HasFinalization = true
	}
	var outstanding []runrecord.GateLifecycle
	for _, preparation := range lifecycles {
		if preparation.State != runrecord.GatePrepared {
			continue
		}
		finalization, finalized, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
		if err != nil {
			return gateLifecycleRecoveryCensus{}, err
		}
		if !finalized {
			outstanding = append(outstanding, preparation)
			continue
		}
		if validated, found := validatedFinalizations[preparation.ID]; !found || validated != finalization.ID {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: preparation %s lacks one validated canonical finalization during recovery",
				preparation.ID,
			)
		}
	}
	census.OutstandingCount = len(outstanding)
	census.OutstandingDebt = outstanding
	if len(outstanding) != 1 {
		return census, nil
	}
	if _, err := runrecord.RequireEnvironment(ctx, store, outstanding[0].Environment); err != nil {
		return gateLifecycleRecoveryCensus{}, fmt.Errorf(
			"gate: load outstanding lifecycle environment during recovery: %w", err,
		)
	}
	census.Outstanding = outstanding[0]
	return census, nil
}

func requireGateLifecycleRecoveryCensus(
	ctx context.Context,
	store *overgodb.Store,
) (gateLifecycleRecoveryCensus, error) {
	census, err := inspectGateLifecycleRecoveryCensus(ctx, store)
	if err != nil {
		return gateLifecycleRecoveryCensus{}, err
	}
	if census.OutstandingCount != 1 {
		return gateLifecycleRecoveryCensus{}, fmt.Errorf(
			"gate: lifecycle recovery found %d unresolved preparations, want 1",
			census.OutstandingCount,
		)
	}
	return census, nil
}

// requireSelectedGateLifecycleRecovery resolves ambiguous recovery debt by an
// exact operator-named preparation instead of the sole-debt cardinality rule.
// The selection is argument-bound: an identity that is not currently
// outstanding, complete recovery debt refuses rather than degrading to any
// nearest candidate.
func requireSelectedGateLifecycleRecovery(
	ctx context.Context,
	store *overgodb.Store,
	selected string,
) (gateLifecycleRecoveryCensus, error) {
	id, err := artifact.ParseID(selected)
	if err != nil || id.Kind() != artifact.KindEvidence {
		return gateLifecycleRecoveryCensus{}, fmt.Errorf(
			"gate: -preparation %q is not an exact evidence artifact ID", selected,
		)
	}
	census, err := inspectGateLifecycleRecoveryCensus(ctx, store)
	if err != nil {
		return gateLifecycleRecoveryCensus{}, err
	}
	for _, preparation := range census.OutstandingDebt {
		if preparation.ID != id {
			continue
		}
		if _, err := runrecord.RequireEnvironment(ctx, store, preparation.Environment); err != nil {
			return gateLifecycleRecoveryCensus{}, fmt.Errorf(
				"gate: load selected lifecycle environment during recovery: %w", err,
			)
		}
		census.Outstanding = preparation
		return census, nil
	}
	return gateLifecycleRecoveryCensus{}, fmt.Errorf(
		"gate: preparation %s is not outstanding recovery debt", selected,
	)
}

func validateLegacyFinalizedHeartbeat(
	ctx context.Context,
	store *overgodb.Store,
	heartbeat runrecord.GateHeartbeat,
) error {
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, heartbeat.Preparation)
	if err != nil {
		return fmt.Errorf("gate: load finalized lifecycle locator preparation: %w", err)
	}
	if preparation.State != runrecord.GatePrepared || preparation.TreeKey != heartbeat.TreeKey ||
		preparation.Environment != heartbeat.Environment {
		return errors.New("gate: finalized lifecycle locator contradicts its exact preparation")
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("gate: finalized lifecycle locator preparation lacks a finalization")
	}
	if err := validateHeartbeatFinalization(ctx, store, preparation, finalization); err != nil {
		return err
	}
	return nil
}

func writeFinalizedHeartbeatForAuthority(
	repo string,
	store *overgodb.Store,
	finalization runrecord.GateLifecycle,
) error {
	ctx := context.Background()
	if err := requireCompleteGateFinalization(ctx, store, finalization); err != nil {
		return fmt.Errorf("gate: validate bootstrapped lifecycle authority: %w", err)
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, *finalization.Preparation)
	if err != nil {
		return err
	}
	heartbeat := runrecord.GateHeartbeat{
		Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatFinalized,
		Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
		PID: os.Getpid(), Updated: time.Now().UTC(),
	}
	return writeJSON(repo, gateHeartbeatFile, heartbeat, clioptions.OutputFileMode)
}

func repairCompletedLifecycleHeartbeat(
	repo string,
	ctx context.Context,
	store *overgodb.Store,
	current runrecord.GateLifecycle,
	heartbeat runrecord.GateHeartbeat,
) (artifact.ID, bool, error) {
	if current.State != runrecord.GateFinalized || current.Preparation == nil ||
		*current.Preparation == heartbeat.Preparation {
		return artifact.ID{}, false, nil
	}
	census, err := inspectGateLifecycleRecoveryCensus(ctx, store)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if census.OutstandingCount != 0 {
		return artifact.ID{}, false, nil
	}
	if err := requireCompleteGateFinalization(ctx, store, current); err != nil {
		return artifact.ID{}, false, fmt.Errorf("gate: validate completed retry authority: %w", err)
	}
	preparation, err := runrecord.RequireGateLifecycle(ctx, store, heartbeat.Preparation)
	if err != nil {
		return artifact.ID{}, false, fmt.Errorf("gate: load completed retry locator preparation: %w", err)
	}
	if preparation.State != runrecord.GatePrepared || preparation.TreeKey != heartbeat.TreeKey ||
		preparation.Environment != heartbeat.Environment {
		return artifact.ID{}, false, errors.New("gate: completed retry locator contradicts its exact preparation")
	}
	finalization, found, err := runrecord.GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if !found {
		return artifact.ID{}, false, errors.New("gate: completed retry locator preparation lacks a finalization")
	}
	if err := validateHeartbeatFinalization(ctx, store, preparation, finalization); err != nil {
		return artifact.ID{}, false, err
	}
	currentID, found, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if !found || currentID != current.ID {
		return artifact.ID{}, false, errors.New("gate: completed retry lifecycle authority moved")
	}
	if err := writeFinalizedHeartbeatForAuthority(repo, store, current); err != nil {
		return artifact.ID{}, false, err
	}
	return preparation.ID, true, nil
}

var gateRecordFailureBeforeStoreCommitHook func(*overgodb.Store)
var gateRecordFailureAfterStoreCommitHook func(*overgodb.Store) error

func recordUnbatchableFailure(repo, storePath string) (artifact.ID, error) {
	return recordSelectedUnbatchableFailure(repo, storePath, "")
}

// recordSelectedUnbatchableFailure additionally accepts an exact preparation
// selector so an operator can close one named stale debt when the census finds
// more than one. The selector is honored only behind a complete finalized
// current authority whose locator matches it; every other recovery shape keeps
// the census cardinality rule.
func recordSelectedUnbatchableFailure(repo, storePath, selected string) (artifact.ID, error) {
	recordStarted := time.Now()
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))); err == nil {
		return artifact.ID{}, errors.New("gate: interrupted commit intent exists; run `go run ./cmd/gate -recover-interrupted` instead of recording failure")
	} else if !errors.Is(err, os.ErrNotExist) {
		return artifact.ID{}, err
	}
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	ctx := context.Background()
	currentID, aliasFound, err := artifact.ResolveAlias(ctx, store, runrecord.GateLifecycleCurrentAlias)
	if err != nil {
		return artifact.ID{}, err
	}
	var current runrecord.GateLifecycle
	if aliasFound {
		current, err = runrecord.RequireGateLifecycle(ctx, store, currentID)
		if err != nil {
			return artifact.ID{}, err
		}
	}
	var heartbeat runrecord.GateHeartbeat
	heartbeatFound := true
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); errors.Is(err, os.ErrNotExist) {
		heartbeatFound = false
	} else if err != nil {
		return artifact.ID{}, err
	} else if err := heartbeat.Validate(); err != nil {
		return artifact.ID{}, errors.New("gate: no valid unresolved lifecycle locator to finalize")
	}
	if selected != "" && (!heartbeatFound || heartbeat.State != runrecord.HeartbeatFinalized || !aliasFound) {
		return artifact.ID{}, errors.New(
			"gate: -preparation recovery requires a finalized lifecycle locator and current terminal authority",
		)
	}
	preserveCurrentAuthority := false
	preserveHeartbeat := false
	legacyBootstrap := false
	censusBoundRecovery := false
	var recoveryCensus gateLifecycleRecoveryCensus
	if !heartbeatFound {
		var preparation runrecord.GateLifecycle
		if aliasFound && current.State == runrecord.GatePrepared {
			preparation = current
		} else {
			if aliasFound && current.State != runrecord.GateFinalized {
				return artifact.ID{}, errors.New("gate: current lifecycle authority cannot identify recovery debt")
			}
			if aliasFound {
				if err := requireCompleteGateFinalization(ctx, store, current); err != nil {
					return artifact.ID{}, fmt.Errorf("gate: validate current lifecycle authority: %w", err)
				}
				recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
				if err != nil {
					return artifact.ID{}, err
				}
				preparation = recoveryCensus.Outstanding
				censusBoundRecovery = true
				preserveCurrentAuthority = true
				preserveHeartbeat = true
			} else {
				recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
				if err != nil {
					return artifact.ID{}, err
				}
				preparation = recoveryCensus.Outstanding
				censusBoundRecovery = true
				legacyBootstrap = true
			}
		}
		heartbeat = runrecord.GateHeartbeat{
			Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
			Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
			PID: os.Getpid(), Updated: time.Now().UTC(),
		}
	} else if heartbeat.State == runrecord.HeartbeatFinalized && !aliasFound {
		recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		if err := validateLegacyFinalizedHeartbeat(ctx, store, heartbeat); err != nil {
			return artifact.ID{}, err
		}
		preparation := recoveryCensus.Outstanding
		heartbeat = runrecord.GateHeartbeat{
			Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
			Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
			PID: os.Getpid(), Updated: time.Now().UTC(),
		}
		censusBoundRecovery = true
		legacyBootstrap = true
		preserveHeartbeat = true
	} else if heartbeat.State == runrecord.HeartbeatFinalized {
		if current.State != runrecord.GateFinalized || current.Preparation == nil {
			return artifact.ID{}, errors.New("gate: finalized lifecycle locator lacks current terminal authority")
		}
		currentPreparation, err := runrecord.RequireGateLifecycle(ctx, store, *current.Preparation)
		if err != nil {
			return artifact.ID{}, err
		}
		if heartbeat.Preparation != currentPreparation.ID || heartbeat.TreeKey != currentPreparation.TreeKey ||
			heartbeat.Environment != currentPreparation.Environment {
			if selected != "" {
				return artifact.ID{}, errors.New(
					"gate: -preparation recovery requires the lifecycle locator to match current terminal authority",
				)
			}
			preparation, repaired, repairErr := repairCompletedLifecycleHeartbeat(
				repo, ctx, store, current, heartbeat,
			)
			if repairErr != nil {
				return artifact.ID{}, repairErr
			}
			if repaired {
				return preparation, nil
			}
			return artifact.ID{}, errors.New("gate: finalized lifecycle locator differs from current terminal authority")
		}
		if err := requireCompleteGateFinalization(ctx, store, current); err != nil {
			return artifact.ID{}, fmt.Errorf("gate: validate current lifecycle authority: %w", err)
		}
		if selected != "" {
			recoveryCensus, err = requireSelectedGateLifecycleRecovery(ctx, store, selected)
		} else {
			recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		}
		if err != nil {
			return artifact.ID{}, err
		}
		preparation := recoveryCensus.Outstanding
		heartbeat = runrecord.GateHeartbeat{
			Version: artifact.InitialDocumentVersion, State: runrecord.HeartbeatRunning,
			Preparation: preparation.ID, TreeKey: preparation.TreeKey, Environment: preparation.Environment,
			PID: os.Getpid(), Updated: time.Now().UTC(),
		}
		censusBoundRecovery = true
		preserveCurrentAuthority = true
		preserveHeartbeat = true
	} else if heartbeat.State != runrecord.HeartbeatRunning && heartbeat.State != runrecord.HeartbeatRecordDebt {
		return artifact.ID{}, errors.New("gate: no valid unresolved lifecycle locator to finalize")
	}
	content, ok, err := artifact.ReadContent(context.Background(), store, heartbeat.Preparation)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("gate: prepared lifecycle unavailable: %w", err)
	}
	if !ok && !aliasFound && heartbeatFound &&
		(heartbeat.State == runrecord.HeartbeatRunning || heartbeat.State == runrecord.HeartbeatRecordDebt) {
		recoveryCensus, err = inspectGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		switch recoveryCensus.OutstandingCount {
		case 0:
		case 1:
			preparation := recoveryCensus.Outstanding
			heartbeat.Preparation = preparation.ID
			heartbeat.TreeKey = preparation.TreeKey
			heartbeat.Environment = preparation.Environment
			censusBoundRecovery = true
			legacyBootstrap = true
			content, ok, err = artifact.ReadContent(ctx, store, preparation.ID)
			if err != nil {
				return artifact.ID{}, fmt.Errorf("gate: prepared lifecycle unavailable: %w", err)
			}
		default:
			return artifact.ID{}, fmt.Errorf(
				"gate: lifecycle recovery found %d unresolved preparations, want 1",
				recoveryCensus.OutstandingCount,
			)
		}
	}
	if !ok {
		priorFinalizedAuthority := aliasFound && current.State == runrecord.GateFinalized &&
			current.Preparation != nil && *current.Preparation != heartbeat.Preparation
		if !heartbeatFound || heartbeat.State != runrecord.HeartbeatRunning || aliasFound && !priorFinalizedAuthority {
			return artifact.ID{}, errors.New("gate: prepared lifecycle unavailable")
		}
		if err := fsatomic.Remove(filepath.Join(repo, filepath.FromSlash(gateHeartbeatFile))); err != nil {
			return artifact.ID{}, fmt.Errorf("gate: remove uncommitted lifecycle locator: %w", err)
		}
		return heartbeat.Preparation, nil
	}
	if aliasFound && !preserveCurrentAuthority {
		switch current.State {
		case runrecord.GatePrepared:
			if current.ID != heartbeat.Preparation {
				return artifact.ID{}, errors.New("gate: lifecycle locator differs from current store authority")
			}
		case runrecord.GateFinalized:
			if current.Preparation == nil || *current.Preparation != heartbeat.Preparation {
				preparation, repaired, repairErr := repairCompletedLifecycleHeartbeat(
					repo, ctx, store, current, heartbeat,
				)
				if repairErr != nil {
					return artifact.ID{}, repairErr
				}
				if repaired {
					return preparation, nil
				}
				return artifact.ID{}, errors.New("gate: finalized lifecycle authority differs from its locator")
			}
		}
	}
	preparation, err := runrecord.ParseGateLifecycle(content.Data)
	if err != nil || preparation.State != runrecord.GatePrepared || preparation.Environment != heartbeat.Environment {
		return artifact.ID{}, errors.New("gate: lifecycle locator contradicts its preparation")
	}
	if !aliasFound && !legacyBootstrap {
		recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		if recoveryCensus.Outstanding.ID != preparation.ID ||
			recoveryCensus.Outstanding.TreeKey != preparation.TreeKey ||
			recoveryCensus.Outstanding.Environment != preparation.Environment {
			return artifact.ID{}, errors.New("gate: lifecycle locator differs from the sole legacy recovery debt")
		}
		censusBoundRecovery = true
		legacyBootstrap = true
	}
	finalization, finalized, err := runrecord.GateFinalizationForPreparation(
		context.Background(), store, preparation.ID,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if finalized {
		if preserveCurrentAuthority || legacyBootstrap {
			return artifact.ID{}, errors.New("gate: recovered stale preparation unexpectedly has a finalization")
		}
		if aliasFound && current.State == runrecord.GateFinalized && current.ID != finalization.ID {
			return artifact.ID{}, errors.New("gate: current lifecycle authority names another finalization")
		}
		if err := validateHeartbeatFinalization(context.Background(), store, preparation, finalization); err != nil {
			return artifact.ID{}, err
		}
		if err := bindRecoveredGateFinalization(
			context.Background(), store, preparation, finalization, currentID, aliasFound,
		); err != nil {
			return artifact.ID{}, err
		}
		return preparation.ID, finalizeHeartbeat(repo, heartbeat)
	}
	if !censusBoundRecovery {
		recoveryCensus, err = requireGateLifecycleRecoveryCensus(ctx, store)
		if err != nil {
			return artifact.ID{}, err
		}
		if recoveryCensus.Outstanding.ID != preparation.ID ||
			recoveryCensus.Outstanding.TreeKey != preparation.TreeKey ||
			recoveryCensus.Outstanding.Environment != preparation.Environment {
			return artifact.ID{}, errors.New("gate: lifecycle locator differs from the sole recovery debt")
		}
		censusBoundRecovery = true
		legacyBootstrap = !aliasFound
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
	outcome, failure, stepName, stepOutcome := runrecord.OutcomeFailed, "record", "record", runrecord.StepFailed
	if heartbeat.State == runrecord.HeartbeatRunning {
		outcome, failure, stepName, stepOutcome = runrecord.OutcomeCancelled, "", "recovery", runrecord.StepCancelled
	}
	durationNS, err := completedGateMeasurement(recordStarted, "record recovery")
	if err != nil {
		return artifact.ID{}, err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, preparation.Environment, codeCommit, outcome, failure, durationNS,
		[]runrecord.GateStep{{
			Name: stepName, Phase: runrecord.PhaseValidate, Outcome: stepOutcome, DurationNS: durationNS,
		}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := record.Batch("gate/final/" + preparation.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	newFinalization, err := runrecord.NewGateFinalization(preparation, codeCommit, record.Result.ID, outcome)
	if err != nil {
		return artifact.ID{}, err
	}
	finalizedContent, err := newFinalization.Content()
	if err != nil {
		return artifact.ID{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipeID})
	batch.Contents = append(batch.Contents, finalizedContent)
	batch.Lineage = append(batch.Lineage, newFinalization.Lineage()...)
	bootstrapFinalization := newFinalization
	if censusBoundRecovery {
		batch.ExpectedHead = &recoveryCensus.Head
	}
	if legacyBootstrap {
		if recoveryCensus.HasFinalization {
			bootstrapFinalization = recoveryCensus.NewestFinalization
		}
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
			Name: runrecord.GateLifecycleCurrentAlias, Target: bootstrapFinalization.ID,
		})
	} else if !preserveCurrentAuthority {
		if aliasFound {
			appendGateFinalizationAlias(&batch, preparation.ID, newFinalization.ID)
		} else {
			batch.Aliases = append(batch.Aliases, artifact.AliasBinding{
				Name: runrecord.GateLifecycleCurrentAlias, Target: newFinalization.ID,
			})
		}
	}
	if gateRecordFailureBeforeStoreCommitHook != nil {
		gateRecordFailureBeforeStoreCommitHook(store)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return artifact.ID{}, err
	}
	if gateRecordFailureAfterStoreCommitHook != nil {
		if err := gateRecordFailureAfterStoreCommitHook(store); err != nil {
			return preparation.ID, err
		}
	}
	if legacyBootstrap {
		return preparation.ID, writeFinalizedHeartbeatForAuthority(repo, store, bootstrapFinalization)
	}
	if preserveHeartbeat {
		return preparation.ID, nil
	}
	if err := finalizeHeartbeat(repo, heartbeat); err != nil {
		return artifact.ID{}, err
	}
	return preparation.ID, nil
}

func completedGateMeasurement(started time.Time, operation string) (uint64, error) {
	duration := time.Since(started).Nanoseconds()
	if duration <= 0 {
		return 0, fmt.Errorf("gate: %s duration is unavailable", operation)
	}
	return uint64(duration), nil
}

func bindRecoveredGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	preparation, finalization runrecord.GateLifecycle,
	current artifact.ID,
	aliasFound bool,
) error {
	if ctx == nil || store == nil || preparation.State != runrecord.GatePrepared ||
		finalization.State != runrecord.GateFinalized || finalization.Preparation == nil ||
		*finalization.Preparation != preparation.ID {
		return errors.New("gate: recovered lifecycle alias authority is invalid")
	}
	if aliasFound && current == finalization.ID {
		return nil
	}
	binding := artifact.AliasBinding{
		Name: runrecord.GateLifecycleCurrentAlias, Target: finalization.ID,
	}
	if aliasFound {
		if current != preparation.ID {
			return errors.New("gate: recovered lifecycle alias moved beyond its preparation")
		}
		previous := current
		binding.Previous = &previous
	}
	_, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:     "gate/recovered-alias/" + preparation.ID.String(),
		Aliases: []artifact.AliasBinding{binding},
	})
	if err != nil {
		return fmt.Errorf("gate: bind recovered lifecycle finalization: %w", err)
	}
	return nil
}

func validateHeartbeatFinalization(
	ctx context.Context,
	store *overgodb.Store,
	preparation, finalization runrecord.GateLifecycle,
) error {
	if finalization.Preparation == nil || *finalization.Preparation != preparation.ID {
		return errors.New("gate: lifecycle locator has a contradictory finalization")
	}
	if err := requireCompleteGateFinalization(ctx, store, finalization); err != nil {
		return fmt.Errorf("gate: lifecycle locator finalization: %w", err)
	}
	return nil
}

func finalizeHeartbeat(repo string, heartbeat runrecord.GateHeartbeat) error {
	heartbeat.State, heartbeat.PID, heartbeat.Updated = runrecord.HeartbeatFinalized, os.Getpid(), time.Now().UTC()
	return writeJSON(repo, gateHeartbeatFile, heartbeat, clioptions.OutputFileMode)
}

func finalizeMatchingHeartbeat(repo string, preparation artifact.ID) error {
	var heartbeat runrecord.GateHeartbeat
	if err := readJSON(repo, gateHeartbeatFile, &heartbeat); errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	if err := heartbeat.Validate(); err != nil {
		return fmt.Errorf("gate: invalid lifecycle locator after recovery: %w", err)
	}
	if heartbeat.Preparation != preparation {
		return errors.New("gate: lifecycle locator differs from the recovered preparation")
	}
	if heartbeat.State == runrecord.HeartbeatFinalized {
		return nil
	}
	return finalizeHeartbeat(repo, heartbeat)
}

func (g *gateContext) manifestAnalysisContent() (artifact.Content, error) {
	if g.manifestPlan == nil || g.manifestDelta == nil || g.manifestImpact == nil {
		return artifact.Content{}, nil
	}
	analysis, err := automationcheck.NewManifestAnalysis(
		*g.manifestDelta, *g.manifestImpact, *g.manifestPlan, g.selection, g.manifestMetrics,
	)
	if err != nil {
		return artifact.Content{}, err
	}
	return analysis.Content()
}

func validateGateDebt(debt gateDebtEnvelope) error {
	_, err := gateDebtFinalization(debt)
	return err
}

func gateDebtFinalization(debt gateDebtEnvelope) (runrecord.GateLifecycle, error) {
	if debt.Version != artifact.InitialDocumentVersion || debt.Preparation.Kind() != artifact.KindEvidence {
		return runrecord.GateLifecycle{}, errors.New("gate: invalid record debt envelope")
	}
	if err := debt.Batch.Validate(); err != nil {
		return runrecord.GateLifecycle{}, err
	}
	if debt.Batch.Key != "gate/final/"+debt.Preparation.String() {
		return runrecord.GateLifecycle{}, errors.New("gate: record debt is not bound to its preparation")
	}
	var matching []runrecord.GateLifecycle
	for _, content := range debt.Batch.Contents {
		if content.Descriptor.MediaType == automationcheck.ManifestAnalysisMediaType &&
			content.Descriptor.Schema == automationcheck.ManifestAnalysisSchema {
			if _, err := automationcheck.ParseManifestAnalysis(content.Data); err != nil {
				return runrecord.GateLifecycle{}, err
			}
		}
		if content.Descriptor.MediaType != runrecord.GateLifecycleMediaType || content.Descriptor.Schema != runrecord.GateLifecycleSchema {
			continue
		}
		lifecycle, err := runrecord.ParseGateLifecycle(content.Data)
		if err != nil {
			return runrecord.GateLifecycle{}, err
		}
		if lifecycle.State == runrecord.GateFinalized && lifecycle.Preparation != nil && *lifecycle.Preparation == debt.Preparation {
			matching = append(matching, lifecycle)
		}
	}
	if len(matching) != 1 {
		return runrecord.GateLifecycle{}, fmt.Errorf("gate: record debt has %d matching finalizations, want 1", len(matching))
	}
	return matching[0], nil
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
	codeCommit := g.planHead
	if outcome == runrecord.OutcomeSucceeded {
		codeCommit = g.committedHead
	}
	if !validGitObjectID(codeCommit) {
		return errors.New("gate: exact code commit is unavailable for the final record")
	}

	recipeID, err := g.gateRecipeID()
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
		switch g.planProjection {
		case plan.MergeProjectionFirstParentTarget:
			if g.mergeAuthority == nil {
				return errors.New("gate: successful first-parent-target merge lacks authority receipt")
			}
			receiptContent, err := g.mergeAuthority.Content()
			if err != nil {
				return err
			}
			batch.Contents = append(batch.Contents, receiptContent)
			batch.Lineage = append(batch.Lineage, g.mergeAuthority.Lineage()...)
			batch.Lineage = append(batch.Lineage, artifact.Lineage{
				Child: record.Result.ID, Parent: g.mergeAuthority.ID, Relation: artifact.RelationDependsOn,
			})
		case plan.MergeProjectionSemanticUnion:
			if g.mergeAuthority != nil {
				return errors.New("gate: semantic-union completion carries projected merge authority")
			}
		default:
			return errors.New("gate: successful completion has invalid plan projection")
		}
	}
	appendGateFinalizationAlias(&batch, g.preparation.ID, finalized.ID)
	for _, manifest := range []*codemanifest.Manifest{g.baseManifest, g.candidateManifest} {
		if manifest == nil {
			continue
		}
		// Register the manifest identity in this batch so the record's
		// lineage stays resolvable even when the batch lands as owed
		// debt before the digest publication ran.
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: manifest.ID})
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: record.Result.ID, Parent: manifest.ID, Relation: artifact.RelationDependsOn,
		})
	}
	if analysis, analysisErr := g.manifestAnalysisContent(); analysisErr != nil {
		return g.oweRecord(batch, analysisErr)
	} else if analysis.Descriptor.ID.Valid() {
		batch.Contents = append(batch.Contents, analysis)
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: record.Result.ID, Parent: analysis.Descriptor.ID, Relation: artifact.RelationDependsOn,
		})
	}
	if outcome == runrecord.OutcomeSucceeded {
		if err := g.appendProfileEvidence(&batch, codeCommit, record.Result.ID); err != nil {
			return err
		}
	}
	if err := g.appendAttemptRecord(&batch, codeCommit, recipeID, record.Result.ID, outcome, failure); err != nil {
		return err
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

	store := g.completionStore
	closeStore := false
	if store == nil {
		store, err = overgodb.Open(filepath.Join(g.repo, g.storePath))
		if err != nil {
			return g.oweRecord(batch, err)
		}
		closeStore = true
	}
	if closeStore {
		defer store.Close()
	}
	if err := requireSoleCurrentGatePreparation(
		context.Background(), store, g.preparation, g.preparationCommit,
	); err != nil {
		// This batch would create a second finalization or advance an alias
		// the gate no longer owns. Do not persist it as debt: retain the
		// existing lifecycle and heartbeat for explicit recovery instead.
		return fmt.Errorf("gate: final record lost preparation authority: %w", err)
	}
	// Digests only: the full manifest is derivable from git at this
	// commit, and persisting it per run was the store's growth curve.
	for _, manifest := range []*codemanifest.Manifest{g.baseManifest, g.candidateManifest} {
		if manifest == nil {
			continue
		}
		if _, err := codemanifest.PublishDigest(context.Background(), store, *manifest); err != nil {
			return g.oweRecord(batch, err)
		}
	}
	if err := appendGateAdvisoryFinding(context.Background(), store, &batch, g.paths, g.honesty); err != nil {
		return g.oweRecord(batch, err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return g.oweRecord(batch, err)
	}
	_ = fsatomic.Remove(filepath.Join(g.repo, filepath.FromSlash(gateDebtFile)))
	return nil
}

func (g *gateContext) gateRecipeID() (artifact.ID, error) {
	if g.manifestPlan != nil {
		if err := g.manifestPlan.Validate(); err != nil {
			return artifact.ID{}, err
		}
		return g.manifestPlan.ID, nil
	}
	return artifact.IdentifyBytes(artifact.KindRecipe, []byte(gateRecipeSeed))
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

// appendAttemptRecord rides the gate's own record batch with the
// typed attempt document: the plan step this run served, the manifest
// selection it observed, and the diff it carried, on success and
// failure alike -- the measurement unit for automation effectiveness.
func (g *gateContext) appendAttemptRecord(
	batch *artifact.Batch,
	codeCommit string,
	recipeID, resultID artifact.ID,
	outcome runrecord.Outcome,
	failure string,
) error {
	item, step, bound := strings.Cut(g.planRef, "/")
	if !bound {
		return fmt.Errorf("gate: attempt record requires an item/step plan reference, got %q", g.planRef)
	}
	attempt := runrecord.AttemptRecord{
		PlanItem: item, PlanStep: step, Result: resultID, Recipe: recipeID,
		// The driver exports the declared strategy; an interactive
		// session leaves it empty and the record stays honest.
		Strategy:   os.Getenv(loop.StrategyEnvironment),
		CodeCommit: codeCommit, Outcome: outcome, Failure: failure,
		WallNS: uint64(time.Since(g.start).Nanoseconds()),
		Selection: runrecord.AttemptSelection{
			Defined: g.manifestMetrics.Defined, Selected: g.manifestMetrics.Selected,
			Excluded: g.manifestMetrics.Excluded, Uncertainty: g.manifestMetrics.Uncertainty,
			CacheEligible: g.manifestMetrics.CacheEligible, CacheHits: g.manifestMetrics.CacheHits,
			PlanningNS: g.manifestMetrics.PlanningNS,
		},
		Diff: g.diff,
	}
	if g.baseManifest != nil {
		attempt.BaseManifest = g.baseManifest.ID
	}
	if g.candidateManifest != nil {
		attempt.CandidateManifest = g.candidateManifest.ID
	}
	var published runrecord.AttemptRecord
	var err error
	if g.strategy == nil {
		published, err = runrecord.NewAttemptRecord(attempt)
	} else {
		published, err = g.strategy.StampAttempt(attempt)
	}
	if err != nil {
		return err
	}
	content, err := published.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, published.Lineage()...)
	return nil
}

func (g *gateContext) resolveAttemptStrategy(reader artifact.Reader) {
	declared := strings.TrimSpace(os.Getenv(loop.StrategyIDEnvironment))
	if declared == "" {
		return
	}
	id, err := artifact.ParseID(declared)
	if err != nil || id.Kind() != artifact.KindProfile || reader == nil {
		g.honesty = append(g.honesty, "attempt strategy profile was declared but not resolvable; attempt remains comparison-ineligible")
		return
	}
	strategy, err := loop.RequireStrategy(context.Background(), reader, id)
	if err != nil {
		g.honesty = append(g.honesty, "attempt strategy profile was declared but not resolvable; attempt remains comparison-ineligible")
		return
	}
	g.strategy = &strategy
}

// observeDiff reads the worktree's change size against HEAD; binary
// rows count as files with no line observation, and a failed read
// reports zero counts rather than inventing any.
func observeDiff(repo string) runrecord.AttemptDiff {
	output, err := command(repo, "git", "diff", "--numstat", "HEAD")
	if err != nil {
		return runrecord.AttemptDiff{}
	}
	var diff runrecord.AttemptDiff
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		insertionsText, remainder, hasDeletions := strings.Cut(line, "\t")
		deletionsText, _, hasPath := strings.Cut(remainder, "\t")
		if !hasDeletions || !hasPath {
			continue
		}
		diff.Files++
		if insertions, err := strconv.Atoi(insertionsText); err == nil {
			diff.Insertions += insertions
		}
		if deletions, err := strconv.Atoi(deletionsText); err == nil {
			diff.Deletions += deletions
		}
	}
	return diff
}

func (g *gateContext) printSummary(output io.Writer, outcome runrecord.Outcome, failure string) {
	var run, reused, skipped, inapplicable []string
	for _, step := range g.steps {
		switch step.Outcome {
		case runrecord.StepSkipped:
			skipped = append(skipped, step.Name)
		case runrecord.StepReused:
			reused = append(reused, step.Name)
		case runrecord.StepInapplicable:
			inapplicable = append(inapplicable, step.Name)
		default:
			run = append(run, step.Name)
		}
	}
	fmt.Fprintf(output, "GATE %s %.1fs | ran=%s | reused=%s | skipped=%s | inapplicable=%s\n", strings.ToUpper(string(outcome)), time.Since(g.start).Seconds(), strings.Join(run, ","), strings.Join(reused, ","), strings.Join(skipped, ","), strings.Join(inapplicable, ","))
	if failure != "" {
		fmt.Fprintf(output, "blocker: %s\n", failure)
	}
	for _, line := range compactHonesty(g.honesty) {
		fmt.Fprintln(output, line)
	}
}

func appendGateAdvisoryFinding(ctx context.Context, store *overgodb.Store, batch *artifact.Batch, owners, honesty []string) error {
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
	return commandEnvironment(dir, nil, name, args...)
}

func gitWriterCommand(dir string, args ...string) (string, error) {
	return gitWriterCommandEnvironment(dir, nil, args...)
}

func gitWriterCommandEnvironment(dir string, environment []string, args ...string) (string, error) {
	cmd := newGateGitWriterCommand(dir, environment, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf(
			"git %s: %v: %s",
			strings.Join(args, " "),
			err,
			clioptions.Tail(string(out), clioptions.DiagnosticTailBytes),
		)
	}
	return string(out), nil
}

func newGateGitWriterCommand(dir string, environment []string, args ...string) *exec.Cmd {
	cmd := exec.Command("git", gitauthority.WriterArguments(args...)...)
	cmd.Dir = dir
	cmd.Env = gateGitEnvironment(gitauthority.RepositoryEnvironment(), environment)
	return cmd
}

func newGateGitReaderCommand(dir string, args ...string) *exec.Cmd {
	cmd := exec.Command("git", append([]string{"--no-replace-objects"}, args...)...)
	cmd.Dir = dir
	cmd.Env = gitauthority.ReaderEnvironment()
	return cmd
}

func commandEnvironment(dir string, environment []string, name string, args ...string) (string, error) {
	commandArgs := args
	if name == "git" {
		commandArgs = append([]string{"--no-replace-objects"}, args...)
	}
	cmd := exec.Command(name, commandArgs...)
	cmd.Dir = dir
	if name == "git" {
		cmd.Env = gateGitEnvironment(gitauthority.ReaderEnvironment(), environment)
	} else if environment != nil {
		cmd.Env = environment
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf(
			"%s %s: %v: %s",
			name,
			strings.Join(args, " "),
			err,
			clioptions.Tail(string(out), clioptions.DiagnosticTailBytes),
		)
	}
	return string(out), nil
}

func gateGitEnvironment(base, environment []string) []string {
	gitEnvironment := slices.Clone(base)
	for _, entry := range environment {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, "GIT_INDEX_FILE") {
			gitEnvironment = append(gitEnvironment, entry)
		}
	}
	return gitEnvironment
}

func gitIndexEnvironment(value string) []string {
	return []string{"GIT_INDEX_FILE=" + value}
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
