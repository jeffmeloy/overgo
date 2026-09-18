package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/authoritylock"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

// The lanes a gate may run after its commit: the device test group, the
// device lane and the browser lane. Their reach is proven by the same
// selection as before; only their place in the pipeline moves, and the
// obligation they leave is recorded before the commit is visible.
var deferredLaneChecks = []string{testDeviceCheckName, "device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName}

const (
	gateLanesLocatorFile = "tmp/gate_lanes.json"
	gateLanesLogFile     = "tmp/gate_lanes.log"
	gateLanesLockFile    = "tmp/gate_lanes.lock"
)

// gateLanesLocator is the advisory pointer to the lane runner: which
// obligation it serves and the process that holds it.
type gateLanesLocator struct {
	Version    uint16                        `json:"version"`
	State      runrecord.LaneObligationState `json:"state"`
	Obligation artifact.ID                   `json:"obligation"`
	CodeCommit string                        `json:"code_commit"`
	PID        int                           `json:"pid"`
	Updated    time.Time                     `json:"updated"`
}

// laneDeferral lists the lane checks this gate runs after its commit.
func (g *gateContext) laneDeferral() map[string]bool {
	if !g.deferLanes {
		return nil
	}
	deferred := make(map[string]bool, len(deferredLaneChecks))
	for _, name := range deferredLaneChecks {
		deferred[name] = true
	}
	return deferred
}

// Owed checks survive a new diff, both in the runner and in an inline retry.
func (g *gateContext) retainLaneObligations(impact automationcheck.Impact, definitions []automationcheck.Check) (automationcheck.Impact, error) {
	if g.laneDebt == nil {
		return impact, nil
	}
	for _, name := range g.laneDebt.Checks {
		if !slices.Contains(deferredLaneChecks, name) || !slices.ContainsFunc(definitions, func(check automationcheck.Check) bool { return check.Descriptor.Name == name }) {
			return automationcheck.Impact{}, fmt.Errorf("gate: required lane %q has no current implementation", name)
		}
	}
	impact.Exclusions = slices.DeleteFunc(slices.Clone(impact.Exclusions), func(exclusion automationcheck.Exclusion) bool {
		return slices.Contains(g.laneDebt.Checks, exclusion.Check)
	})
	return impact, nil
}

// rewireDeferredLanes lets the commit follow the host tests alone when the
// lanes run afterwards; the declared graph otherwise stands.
func (g *gateContext) rewireDeferredLanes(definitions []automationcheck.Check) []automationcheck.Check {
	if !g.deferLanes {
		return definitions
	}
	for index := range definitions {
		switch definitions[index].Descriptor.Name {
		case "commit":
			definitions[index].Descriptor.Dependencies = []string{testRestCheckName}
		case testRestCheckName:
			definitions[index].Descriptor.Dependencies = []string{testOwnersCheckName}
		}
	}
	return definitions
}

// requireLaneObligationsResolved admits a gate against the store's current
// lane obligation: none or a resolved one admits; a failed one admits with
// the debt returned so the gate runs its lanes inline; an open one refuses
// until its runner finishes or is resumed.
func requireLaneObligationsResolved(repo string, store *overgodb.Store) (*runrecord.GateLaneObligation, error) {
	current, found, err := runrecord.CurrentGateLaneObligation(context.Background(), store)
	if err != nil {
		return nil, fmt.Errorf("gate: resolve lane obligation: %w", err)
	}
	if !found || current.Resolved() {
		return nil, nil
	}
	if current.State == runrecord.LaneObligationFailed {
		return &current, nil
	}
	guard, err := acquireLaneRunner(repo)
	if errors.Is(err, processlock.ErrBusy) {
		return nil, fmt.Errorf("gate: deferred lanes of %.12s are still running (log %s); wait for them before another gate", current.CodeCommit, gateLanesLogFile)
	}
	if err != nil {
		return nil, fmt.Errorf("gate: inspect lane ownership: %w", err)
	}
	if err := guard.Close(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("gate: deferred lanes of %.12s did not finish; run `go run ./cmd/gate -lanes` before another gate", current.CodeCommit)
}

// appendLaneObligation records the lanes a successful commit still owes in
// the same batch as its result, and lets an inline lane pass supersede the
// failed obligation that forced it.
func (g *gateContext) appendLaneObligation(batch *artifact.Batch, codeCommit string, result artifact.ID) error {
	now := time.Now()
	if g.laneDebt != nil && len(g.deferredLanes) == 0 {
		superseded, err := g.laneDebt.Transition(runrecord.LaneObligationSuperseded, result, now)
		if err != nil {
			return err
		}
		content, err := superseded.Content()
		if err != nil {
			return err
		}
		previous := g.laneDebt.ID
		batch.Contents = append(batch.Contents, content)
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: runrecord.GateLaneObligationAlias, Target: superseded.ID, Previous: &previous})
		batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: superseded.ID, Parent: previous, Relation: artifact.RelationDerivedFrom})
		g.note("lane debt superseded: the lanes ran inline on " + codeCommit[:12] + " and passed")
		return nil
	}
	if len(g.deferredLanes) == 0 {
		return nil
	}
	obligation, err := runrecord.NewGateLaneObligation(codeCommit, g.preparation.ID, result, g.deferredLanes, g.paths, now)
	if err != nil {
		return err
	}
	content, err := obligation.Content()
	if err != nil {
		return err
	}
	var previous *artifact.ID
	if current, found, err := runrecord.CurrentGateLaneObligation(context.Background(), g.store); err != nil {
		return err
	} else if found {
		previous = &current.ID
	}
	batch.Contents = append(batch.Contents, content)
	batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: runrecord.GateLaneObligationAlias, Target: obligation.ID, Previous: previous})
	g.laneObligation = &obligation
	return nil
}

// spawnLaneRunner starts the detached runner for the obligation this gate
// recorded and leaves its locator beside the lifecycle locator.
func (g *gateContext) spawnLaneRunner(storePath string) error {
	if g.laneObligation == nil {
		return nil
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	executable, err = laneExecutable(g.repo, executable)
	if err != nil {
		return fmt.Errorf("gate: retain lane executable: %w", err)
	}
	log, err := os.OpenFile(filepath.Join(g.repo, filepath.FromSlash(gateLanesLogFile)), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, clioptions.OutputFileMode)
	if err != nil {
		return err
	}
	defer log.Close()
	pid, err := processcontrol.StartDetached(processcontrol.Command{
		Path: executable, Args: []string{"-lanes", "-store", storePath}, Dir: g.repo,
	}, log)
	if err != nil {
		return fmt.Errorf("gate: start lane runner: %w", err)
	}
	locator := gateLanesLocator{
		Version: artifact.InitialDocumentVersion, State: runrecord.LaneObligationPending,
		Obligation: g.laneObligation.ID, CodeCommit: g.laneObligation.CodeCommit, PID: pid, Updated: time.Now().UTC(),
	}
	if err := writeJSON(g.repo, gateLanesLocatorFile, locator, clioptions.OutputFileMode); err != nil {
		return err
	}
	fmt.Printf("gate: lanes deferred: %s run on %.12s as pid %d; log %s; the next gate waits for them\n",
		strings.Join(g.laneObligation.Checks, ","), g.laneObligation.CodeCommit, pid, gateLanesLogFile)
	return nil
}

// laneExecutable retains the running image outside go run's temporary tree.
// Admission serializes lane obligations; one file replaces per-run snapshots.
func laneExecutable(repo, executable string) (string, error) {
	source, err := os.Stat(executable)
	if err != nil {
		return "", err
	}
	path := filepath.Join(repo, "bin", "gate-lanes"+filepath.Ext(executable))
	if target, err := os.Stat(path); err == nil {
		if os.SameFile(source, target) {
			return path, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), clioptions.OutputDirectoryMode); err != nil {
		return "", err
	}
	if err := atomicfile.Write(path, data, source.Mode().Perm()); err != nil {
		return "", err
	}
	return path, nil
}

// runDeferredLanes is `gate -lanes`: it takes the store's open obligation,
// runs its lanes against the exact landed commit in a candidate worktree,
// and records the outcome on the obligation chain.
func runDeferredLanes(repo, storePath string, execute func(*gateContext, runrecord.GateLaneObligation) (runrecord.Outcome, string, []runrecord.GateStep, error)) (runErr error) {
	if execute == nil {
		return errors.New("gate: deferred execution is absent")
	}
	guard, err := acquireLaneRunner(repo)
	if err != nil {
		return fmt.Errorf("gate: acquire lane runner: %w", err)
	}
	defer func() { runErr = errors.Join(runErr, guard.Close()) }()
	// The gate that spawned this runner still holds the authority lock for a
	// moment; the runner waits on the lock itself under its own lifetime.
	lock, err := authoritylock.AcquireContext(context.Background(), repo)
	if err != nil {
		return err
	}
	defer func() { runErr = errors.Join(runErr, lock.Close()) }()
	store, err := overgodb.OpenContext(context.Background(), filepath.Join(repo, storePath))
	if err != nil {
		return fmt.Errorf("gate: open store for the lane runner: %w", err)
	}
	g := &gateContext{
		repo: repo, storePath: storePath, store: store, start: time.Now(), clock: processmeasure.NewStopwatch(),
		stepEvidence: map[string]string{}, terminal: map[string]automationcheck.Evidence{},
	}
	defer func() { runErr = errors.Join(runErr, g.closeStore()) }()
	current, found, err := runrecord.CurrentGateLaneObligation(context.Background(), store)
	if err != nil {
		return err
	}
	if !found || current.Resolved() {
		return errors.New("gate: no deferred lanes are outstanding")
	}
	g.environment, err = discoverEnvironment(repo)
	if err != nil {
		return err
	}
	running, err := g.resumeLaneObligation(current)
	if err != nil {
		return err
	}
	// The locator is advisory; the OS guard owns the runner's lifetime.
	if err := g.writeLaneLocator(running); err != nil {
		return err
	}
	g.paths = slices.Clone(running.Paths)
	tree, err := command(repo, "git", "rev-parse", running.CodeCommit+"^{tree}")
	if err != nil {
		return fmt.Errorf("gate: resolve the landed commit's tree: %w", err)
	}
	g.fixedTree = strings.TrimSpace(tree)
	g.sourceEnv, err = g.sourceEnvironment()
	if err != nil {
		return err
	}
	var outcome runrecord.Outcome
	var failure string
	var steps []runrecord.GateStep
	err = g.withCandidateWorktree(g.fixedTree, func(string) error {
		releaseErr := lock.Close()
		lock = nil
		if releaseErr != nil {
			return releaseErr
		}
		outcome, failure, steps, runErr = execute(g, running)
		return nil
	})
	if err != nil {
		outcome, failure, steps, runErr = deferredResolutionFailure(g.clock, steps, errors.Join(runErr, err))
	}
	if lock == nil {
		lock, err = authoritylock.AcquireContext(context.Background(), repo)
		if err != nil {
			return errors.Join(runErr, err)
		}
	}
	if err := store.Refresh(context.Background()); err != nil {
		return errors.Join(runErr, err)
	}
	latest, found, err := runrecord.CurrentGateLaneObligation(context.Background(), store)
	if err != nil || !found || latest.ID != running.ID {
		return errors.Join(runErr, err, errors.New("gate: lane obligation changed during execution"))
	}
	recipeID, err := g.gateRecipeID()
	if err != nil {
		return err
	}
	wall, err := g.clock.Elapsed()
	if err != nil {
		return err
	}
	record, err := runrecord.NewGateRecord(recipeID, g.environment.ID, running.CodeCommit, outcome, failure, wall, steps)
	if err != nil {
		return err
	}
	batch, err := record.Batch("gate/lanes/run/" + running.ID.String())
	if err != nil {
		return err
	}
	state := runrecord.LaneObligationPassed
	if outcome != runrecord.OutcomeSucceeded {
		state = runrecord.LaneObligationFailed
	}
	resolved, err := running.Transition(state, record.Result.ID, time.Now())
	if err != nil {
		return err
	}
	content, err := resolved.Content()
	if err != nil {
		return err
	}
	environmentContent, err := g.environment.Content()
	if err != nil {
		return err
	}
	previous := running.ID
	batch.Contents = append(batch.Contents, content, environmentContent)
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipeID})
	batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: runrecord.GateLaneObligationAlias, Target: resolved.ID, Previous: &previous})
	batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: resolved.ID, Parent: previous, Relation: artifact.RelationDerivedFrom})
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return fmt.Errorf("gate: record the lane outcome: %w", err)
	}
	_ = g.writeLaneLocator(resolved)
	fmt.Printf("LANES %s %.1fs commit=%.12s checks=%s result=%s\n", strings.ToUpper(string(state)),
		time.Since(g.start).Seconds(), running.CodeCommit, strings.Join(running.Checks, ","), record.Result.ID)
	if state == runrecord.LaneObligationFailed {
		return fmt.Errorf("gate: deferred lanes failed on %.12s: %w", running.CodeCommit, runErr)
	}
	return nil
}

func acquireLaneRunner(repo string) (*processlock.Lock, error) {
	path := filepath.Join(repo, filepath.FromSlash(gateLanesLockFile))
	if err := os.MkdirAll(filepath.Dir(path), clioptions.OutputDirectoryMode); err != nil {
		return nil, err
	}
	return processlock.Acquire(path, gatePrivateFileMode)
}

// resumeLaneObligation retains an interrupted run's exact obligation. Its
// caller holds the authority lock and has rejected a live previous runner.
func (g *gateContext) resumeLaneObligation(current runrecord.GateLaneObligation) (runrecord.GateLaneObligation, error) {
	if current.State == runrecord.LaneObligationRunning {
		return current, nil
	}
	running, err := current.Transition(runrecord.LaneObligationRunning, artifact.ID{}, time.Now())
	if err != nil {
		return running, err
	}
	return running, g.publishLaneState(running, current.ID)
}

// executeDeferredLanes plans the landed commit as the candidate and runs the
// obligation's lanes with every other check satisfied by the gate that
// deferred them.
func (g *gateContext) executeDeferredLanes(obligation runrecord.GateLaneObligation) (runrecord.Outcome, string, []runrecord.GateStep, error) {
	started := processmeasure.NewStopwatch()
	g.laneDebt = &obligation
	var steps []runrecord.GateStep
	outcome, failure := runrecord.OutcomeSucceeded, ""
	err := g.withCandidateWorktree(g.fixedTree, func(string) error {
		planned, err := g.planPipeline()
		if err != nil {
			return err
		}
		g.manifestPlan = planned.manifest
		if planned.manifest == nil {
			return errors.New("gate: deferred lanes lack source-bound verification authority")
		}
		wanted := map[string]bool{testPlanCheckName: true}
		for _, name := range obligation.Checks {
			wanted[name] = true
		}
		satisfied := map[string]bool{}
		var checks []automationcheck.Invocation
		inputs := map[artifact.ID]artifact.ID{}
		for _, check := range planned.invocations {
			name := check.Check.Name
			if !wanted[name] {
				satisfied[name] = true
				continue
			}
			input, err := g.phaseInputFingerprint(name)
			if err != nil {
				return err
			}
			if check, err = g.bindCheckExecution(*planned.manifest, check, input); err != nil {
				return err
			}
			inputs[check.ID] = input
			checks = append(checks, check)
		}
		checks = wireLaneRunnerDependencies(checks)
		cache := g.loadRetryCache()
		results, err := g.executeChecks(checks, satisfied, inputs, &cache, nil, nil)
		if err != nil {
			return err
		}
		for _, result := range results {
			if !result.Invocation.ID.Valid() {
				continue
			}
			name := result.Invocation.Check.Name
			steps = append(steps, gateEvidenceRecord(name, result.Invocation.Check.Phase, result.Evidence, result.Err, g.stepEvidence[name]))
			if result.Err != nil && failure == "" {
				outcome, failure = runrecord.OutcomeFailed, name
			}
		}
		if err := checkFailures(results); err != nil {
			return err
		}
		return validateManifestCommitAdmission(*planned.manifest, g.terminal, satisfied)
	})
	if err != nil && failure == "" {
		return deferredResolutionFailure(started, steps, err)
	}
	return outcome, failure, steps, err
}

func deferredResolutionFailure(started processmeasure.Stopwatch, steps []runrecord.GateStep, err error) (runrecord.Outcome, string, []runrecord.GateStep, error) {
	const failure = "lane-resolution"
	wall, wallErr := started.Elapsed()
	if wallErr == nil {
		steps = append(steps, gateEvidenceRecord(failure, runrecord.PhaseValidate, automationcheck.Evidence{DurationNS: wall}, err, ""))
	}
	return runrecord.OutcomeFailed, failure, steps, errors.Join(err, wallErr)
}

// wireLaneRunnerDependencies makes the device test group follow the test
// plan it consumes: in the gate the owners' tests stand between them, and
// the runner satisfies those without executing them.
func wireLaneRunnerDependencies(checks []automationcheck.Invocation) []automationcheck.Invocation {
	for index := range checks {
		if checks[index].Check.Name != testDeviceCheckName {
			continue
		}
		dependencies := checks[index].Check.Dependencies
		if !slices.Contains(dependencies, testPlanCheckName) {
			checks[index].Check.Dependencies = append(slices.Clone(dependencies), testPlanCheckName)
		}
	}
	return checks
}

func (g *gateContext) publishLaneState(state runrecord.GateLaneObligation, previous artifact.ID) error {
	batch, err := state.Batch(&previous)
	if err != nil {
		return err
	}
	_, err = g.store.Commit(context.Background(), batch)
	return err
}

func (g *gateContext) writeLaneLocator(obligation runrecord.GateLaneObligation) error {
	return writeJSON(g.repo, gateLanesLocatorFile, gateLanesLocator{
		Version: artifact.InitialDocumentVersion, State: obligation.State, Obligation: obligation.ID,
		CodeCommit: obligation.CodeCommit, PID: os.Getpid(), Updated: time.Now().UTC(),
	}, clioptions.OutputFileMode)
}
