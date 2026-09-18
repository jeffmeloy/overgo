// Command loop runs the harness-agnostic plan->implement->verify->commit
// driver. The worker is any command that accepts the dispatched prompt --
// claude, a GPT CLI, anything -- named in a machine-local loop config; the
// plan, verifier, gate, and findings stay the repository's deterministic Go.
// There is no turn concept: worker exit is an event, continuity is this loop.
//
//	go run ./cmd/loop -config docs/loop.json
//
// Config (machine-local, gitignored like local-models.json):
//
//	{
//	  "worker": ["claude", "-p", "{prompt}"],
//	  "strategy_id": "profile:sha256:<digest>",
//	  "worker_timeout_minutes": 90
//	}
//
// Pause by creating docs/.loop_pause; delete it to resume.
package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
)

const pauseMarker = "docs/.loop_pause"

type config struct {
	Worker []string `json:"worker"`
	// Strategy optionally names this worker configuration; attempt
	// records carry it as display metadata. Unset derives a stable digest
	// of the worker command.
	Strategy string `json:"strategy"`
	// StrategyID names the already-published, content-addressed strategy
	// profile. Its Loop field is the sole retry, invocation, saturation,
	// and closure-budget authority for this run.
	StrategyID artifact.ID `json:"strategy_id"`
	// Repository contains the strategy profile. Empty uses the same local
	// default as gate.
	Repository string `json:"repository"`
	// Proposals is the pending-proposal queue directory.
	Proposals            string `json:"proposals"`
	WorkerTimeoutMinutes int    `json:"worker_timeout_minutes"`
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "loop:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	flags := flag.NewFlagSet("loop", flag.ContinueOnError)
	configPath := flags.String("config", "docs/loop.json", "machine-local loop configuration")
	repoPath := flags.String("repo", "", "OvergoDB root for the evidence doors; empty resolves the store through the data-root contract")
	publishStrategySpec := flags.String("publish-strategy", "", "publish one strategy from this spec (worker, catalog, loop config) and exit")
	experimentSpec := flags.String("experiment", "", "replay one strategy experiment from this spec and print the comparison")
	publishFitnessSpec := flags.String("publish-fitness", "", "publish one pairwise improvement-fitness proof from this request spec")
	resourceLanesSpec := flags.String("compare-resource-fitness", "", "replay one resource no-regression proof from this lanes spec")
	resourceRunsSpec := flags.String("compare-resource-runs", "", "judge a baseline and a candidate run's committed resource observations through the resource no-regression owner from this spec ({name, baseline_run, repeat_runs, candidate_run, required_metrics}; repeat_runs are identical-protocol repeats of the baseline that set the measured noise envelope)")
	deriveRecipesSpec := flags.String("derive-recipes", "", "derive one materialized candidate's per-arm recipes from this spec")
	moeCoverageSpec := flags.String("moe-coverage", "", "run one indexed MoE router observation read from this spec")
	evaluateCandidateSpec := flags.String("evaluate-candidate", "", "judge one candidate's measured arms through the cross-domain evaluator from this spec")
	publishTaskSpec := flags.String("publish-task", "", "publish one agent task contract from this spec")
	publishDelegationSpec := flags.String("publish-delegation", "", "publish one delegated agent invocation from this spec")
	automationTransitionSpecPath := flags.String("publish-automation-transition", "", "publish one automation policy lifecycle transition from this spec")
	trajectoryPlanSpecPath := flags.String("bind-trajectory-plan", "", "bind agent trajectories onto one stored evaluation plan from this spec")
	efficiencyTraceSpecPath := flags.String("publish-efficiency-trace", "", "publish one measured interaction-work trace from this spec")
	directionSpecPath := flags.String("publish-direction", "", "publish one extracted residual-direction claim from this spec")
	attemptReceiptSpecPath := flags.String("attempt-receipt", "", "resolve and print one terminal attempt receipt from this spec")
	if err := flags.Parse(args); err != nil {
		return err
	}
	resolved, err := dataroot.StoreRoot(*repoPath)
	if err != nil {
		return err
	}
	*repoPath = resolved
	switch {
	case *publishStrategySpec != "":
		return publishStrategy(*repoPath, *publishStrategySpec, os.Stdout)
	case *experimentSpec != "":
		return compareStrategies(*repoPath, *experimentSpec, os.Stdout)
	case *publishFitnessSpec != "":
		return publishFitness(*repoPath, *publishFitnessSpec, os.Stdout)
	case *resourceLanesSpec != "":
		return compareResourceLanes(*repoPath, *resourceLanesSpec, os.Stdout)
	case *resourceRunsSpec != "":
		return compareResourceRuns(*repoPath, *resourceRunsSpec, os.Stdout)
	case *deriveRecipesSpec != "":
		return deriveRecipes(*repoPath, *deriveRecipesSpec, os.Stdout)
	case *moeCoverageSpec != "":
		return queryMoECoverage(*repoPath, *moeCoverageSpec, os.Stdout)
	case *evaluateCandidateSpec != "":
		return evaluateCandidate(*repoPath, *evaluateCandidateSpec, os.Stdout)
	case *publishTaskSpec != "":
		return publishTaskContract(*repoPath, *publishTaskSpec, os.Stdout)
	case *publishDelegationSpec != "":
		return publishDelegation(*repoPath, *publishDelegationSpec, os.Stdout)
	case *automationTransitionSpecPath != "":
		return publishAutomationTransition(*repoPath, *automationTransitionSpecPath, os.Stdout)
	case *trajectoryPlanSpecPath != "":
		return bindTrajectoryPlan(*repoPath, *trajectoryPlanSpecPath, os.Stdout)
	case *efficiencyTraceSpecPath != "":
		return publishEfficiencyTrace(*repoPath, *efficiencyTraceSpecPath, os.Stdout)
	case *directionSpecPath != "":
		return publishDirection(*repoPath, *directionSpecPath, os.Stdout)
	case *attemptReceiptSpecPath != "":
		return readAttemptReceipt(*repoPath, *attemptReceiptSpecPath, os.Stdout)
	}
	var loaded config
	if err := jsonfile.Decode(*configPath, &loaded); err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	if len(loaded.Worker) == 0 {
		return errors.New("config names no worker command")
	}
	if loaded.WorkerTimeoutMinutes <= 0 {
		loaded.WorkerTimeoutMinutes = 90
	}
	if loaded.Proposals == "" {
		loaded.Proposals = "docs/proposals"
	}
	strategy, err := resolveConfiguredStrategy(loaded)
	if err != nil {
		return err
	}
	campaign, err := processcontrol.BeginCampaign(loopWorktree, strings.TrimSpace(os.Getenv(plan.AutomationWorkerEnvironment)))
	if err != nil {
		return err
	}
	defer campaign.Close()
	priorSession, sessionPresent := os.LookupEnv(processcontrol.CampaignEnvironment)
	_, session, _ := strings.Cut(campaign.Environment(), "=")
	if err := os.Setenv(processcontrol.CampaignEnvironment, session); err != nil {
		return err
	}
	defer func() {
		if sessionPresent {
			_ = os.Setenv(processcontrol.CampaignEnvironment, priorSession)
		} else {
			_ = os.Unsetenv(processcontrol.CampaignEnvironment)
		}
	}()
	stopStore, err := overgodb.OpenReadOnly(gitauthority.CanonicalOvergoDBDirectory)
	if err != nil {
		return err
	}
	defer stopStore.Close()
	prompt, found, err := artifact.ReadContent(context.Background(), stopStore, strategy.Prompt)
	if err != nil || !found {
		return errors.Join(errors.New("loop: configured strategy prompt is unavailable"), err)
	}
	world := &execWorld{config: loaded, strategy: strategy, stopStore: stopStore, promptPrefix: string(prompt.Data), campaign: campaign}
	outcome, err := loop.Run(world, strategy.Loop)
	fmt.Printf("loop: %s after %d worker invocation(s); parked=%v; audit: the gate remains the sole commit path, workers cannot advance an unverified step\n",
		outcome.Reason, outcome.Invocations, outcome.Parked)
	return err
}

func resolveConfiguredStrategy(loaded config) (loop.Strategy, error) {
	if loaded.StrategyID.Kind() != artifact.KindProfile {
		return loop.Strategy{}, errors.New("config requires an exact strategy_id profile")
	}
	repository := loaded.Repository
	repository = cmp.Or(repository, gitauthority.CanonicalOvergoDBDirectory)
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return loop.Strategy{}, fmt.Errorf("open strategy repository: %w", err)
	}
	strategy, strategyErr := loop.RequireStrategy(context.Background(), store, loaded.StrategyID)
	closeErr := store.Close()
	if strategyErr != nil {
		return loop.Strategy{}, fmt.Errorf("resolve strategy: %w", strategyErr)
	}
	if closeErr != nil {
		return loop.Strategy{}, fmt.Errorf("close strategy repository: %w", closeErr)
	}
	return strategy, nil
}

// execWorld adapts the driver to the repository's real owners.
type execWorld struct {
	stopStore      *overgodb.Store
	config         config
	strategy       loop.Strategy
	validationDebt string
	promptPrefix   string
	campaign       *processcontrol.Campaign
}

func (w *execWorld) Current() (loop.Step, bool, error) {
	if err := w.awaitValidation(); err != nil {
		return loop.Step{}, false, err
	}
	// The dispatch arrives as data; the prose line is never parsed back.
	out, err := planCommand("-prompt", "-json")
	if err != nil {
		return loop.Step{}, false, err
	}
	var dispatch plan.Dispatch
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &dispatch); err != nil {
		return loop.Step{}, false, fmt.Errorf("unparseable dispatch %q: %w", strings.TrimSpace(out), err)
	}
	if dispatch.Complete {
		return loop.Step{}, false, nil
	}
	if dispatch.Waiting != "" {
		return loop.Step{}, false, fmt.Errorf("%w: %s", loop.ErrWorkWaiting, dispatch.Waiting)
	}
	return loop.Step{Item: dispatch.Item, ID: dispatch.Step}, true, nil
}

// awaitValidation waits for a live gate to release its mutation ownership. A
// worker's supervised exit may also have ended its detached lane child; the
// durable obligation then resumes through the existing gate owner.
func (w *execWorld) awaitValidation() error {
	if err := authoritylock.Wait(context.Background(), loopWorktree); err != nil {
		return err
	}
	if w.stopStore == nil {
		return nil
	}
	if err := w.stopStore.Refresh(context.Background()); err != nil {
		return err
	}
	obligation, found, err := runrecord.CurrentGateLaneObligation(context.Background(), w.stopStore)
	w.validationDebt = ""
	if err != nil || !found || obligation.Resolved() {
		return err
	}
	if obligation.State == runrecord.LaneObligationFailed {
		w.validationDebt = fmt.Sprintf("Deferred validation failed for commit %s; inspect result %s and repair the recorded failure before the next commit. Do not rerun unchanged tests blindly.", obligation.CodeCommit, obligation.Outcome)
		return nil
	}
	if w.Paused() {
		return nil
	}
	out, err := runTool("go", "run", "./cmd/gate", "-lanes")
	if err != nil {
		// A recorded lane failure is work for the next repair invocation, not
		// an invitation to rerun identical measurements until they turn green.
		if refreshErr := w.stopStore.Refresh(context.Background()); refreshErr != nil {
			return refreshErr
		}
		current, found, readErr := runrecord.CurrentGateLaneObligation(context.Background(), w.stopStore)
		if readErr == nil && found && current.Resolved() {
			return nil
		}
		if readErr == nil && found && current.State == runrecord.LaneObligationFailed {
			w.validationDebt = fmt.Sprintf("Deferred validation failed for commit %s; inspect result %s and repair the recorded failure before the next commit. Do not rerun unchanged tests blindly.", current.CodeCommit, current.Outcome)
			return nil
		}
		return fmt.Errorf("loop: resume deferred validation: %w: %s", err, tailOf(out, 4000))
	}
	return nil
}

func (w *execWorld) Prompt(loop.Step) (string, error) {
	prompt, err := planCommand("-prompt")
	if err == nil && w.validationDebt != "" {
		prompt += "\n\n" + w.validationDebt
	}
	return prompt, err
}

func (w *execWorld) RunWorker(step loop.Step, prompt, feedback string) (string, error) {
	text := prompt
	if w.promptPrefix != "" {
		text = w.promptPrefix + "\n\n" + text
	}
	if feedback != "" {
		text += "\n\nPREVIOUS ATTEMPT FEEDBACK (fix this, then commit through the gate):\n" + feedback
	}
	arguments := make([]string, len(w.config.Worker)-1)
	for index, argument := range w.config.Worker[1:] {
		arguments[index] = strings.ReplaceAll(argument, "{prompt}", text)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(w.config.WorkerTimeoutMinutes)*time.Minute)
	defer cancel()
	watchDone := make(chan struct{})
	if w.campaign != nil {
		go func() {
			defer close(watchDone)
			for {
				select {
				case <-ctx.Done():
					return
				case <-w.campaign.Stops():
					if w.Paused() {
						cancel()
						return
					}
				}
			}
		}()
	} else {
		close(watchDone)
	}
	var combined bytes.Buffer
	output := io.MultiWriter(os.Stdout, &combined)
	// The declared strategy identity reaches every gate run inside the
	// worker, so each attempt record states which strategy produced it.
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: w.config.Worker[0],
		Args: arguments,
		Env: append(loopEnvironment(),
			loop.StrategyEnvironment+"="+loop.StrategyIdentity(w.config.Worker, w.config.Strategy),
			loop.StrategyIDEnvironment+"="+w.strategy.ID.String()),
		Stdin:  strings.NewReader(text),
		Stdout: output,
		Stderr: output,
	})
	cancel()
	<-watchDone
	tail := tailOf(combined.String(), 4000)
	if err != nil && receipt.WallNS == 0 {
		return tail, &loop.LaunchError{Err: err}
	}
	// A nonzero or timed-out worker still exited; the plan decides what it
	// accomplished. Log the event and continue.
	fmt.Printf("loop: worker exited for %s (err=%v)\n", step.Key(), err)
	return tail, nil
}

func (w *execWorld) Verify(loop.Step) (string, error) {
	out, err := runTool("go", "run", "./cmd/plan", "-verify")
	if err == nil {
		return "", nil
	}
	return tailOf(out, 4000), nil
}

func (w *execWorld) Park(step loop.Step, reason string) error {
	out, err := runTool("go", "run", "./cmd/finding",
		"-title", "loop parked "+step.Key()+" after exhausted worker attempts",
		"-severity", "high",
		"-owner", step.Key(),
		"-evidence", reason,
		"-closure", "fix the step or its verifier, then rerun cmd/loop",
		"-check", "go run ./cmd/plan -verify",
	)
	if err != nil {
		return fmt.Errorf("park finding: %w: %s", err, tailOf(out, 500))
	}
	fmt.Printf("loop: PARKED %s -- %s\n", step.Key(), reason)
	return nil
}

// Paused retains the local kill switch and reads scoped stop authority through
// the already-open store. It does not re-resolve the plan or reopen its journal.
func (w *execWorld) Paused() bool {
	if _, err := os.Stat(pauseMarker); err == nil {
		return true
	}
	var reader artifact.Reader
	if w.stopStore != nil {
		reader = w.stopStore
	}
	stop, err := plan.ReadStop(context.Background(), reader, loopWorktree, "", plan.ExecutionUnattended)
	if err != nil {
		fmt.Fprintln(os.Stderr, "loop stop authority:", err)
		return true
	}
	if stop.Blocked {
		fmt.Fprintln(os.Stderr, stop.String())
	}
	return stop.Blocked
}

// AdmitNext admits the lexically first pending proposal through
// deterministic admission and consumes its spec file into tmp/ so the
// queue drains exactly once per proposal.
func (w *execWorld) AdmitNext() (string, bool, error) {
	entries, err := os.ReadDir(w.config.Proposals)
	if err != nil {
		if os.IsNotExist(err) {
			return "", false, nil
		}
		return "", false, err
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") {
			names = append(names, entry.Name())
		}
	}
	if len(names) == 0 {
		return "", false, nil
	}
	slices.Sort(names)
	spec := filepath.Join(w.config.Proposals, names[0])
	out, err := runTool("go", "run", "./cmd/plan", "-admit-proposal", spec)
	if err != nil {
		// A refused proposal is consumed, never retried forever: the
		// refusal moves with the spec for the operator to read.
		_ = consumeProposal(spec, "refused")
		return "", false, fmt.Errorf("admit %s: %w: %s", names[0], err, tailOf(out, 500))
	}
	item := strings.TrimSpace(string(out))
	if index := strings.LastIndex(item, "as plan row "); index >= 0 {
		item = strings.TrimSpace(item[index+len("as plan row "):])
	}
	if err := consumeProposal(spec, "admitted"); err != nil {
		return "", false, err
	}
	fmt.Printf("loop: admitted proposal %s as %s\n", names[0], item)
	return item, true, nil
}

// Block marks a parked proposal row blocked so dispatch moves past it
// to the next proposal; the row and its finding stay for the operator.
func (w *execWorld) Block(step loop.Step, reason string) (err error) {
	lock, err := authoritylock.Acquire(loopWorktree)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	document, err := plan.Load("")
	if err != nil {
		return err
	}
	if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
		return err
	}
	for index := range document.Items {
		if document.Items[index].ID == step.Item {
			document.Items[index].Status = "blocked:proposal-parked"
			return plan.Save("", document)
		}
	}
	return fmt.Errorf("block: plan item %q is absent", step.Item)
}

func consumeProposal(spec, disposition string) error {
	target := filepath.Join("tmp", "proposals-"+disposition)
	if err := os.MkdirAll(target, clioptions.OutputDirectoryMode); err != nil {
		return err
	}
	return os.Rename(spec, filepath.Join(target, filepath.Base(spec)))
}

func planCommand(verbs ...string) (string, error) {
	out, err := runTool(append([]string{"go", "run", "./cmd/plan"}, verbs...)...)
	if err != nil {
		return "", fmt.Errorf("plan %s: %w: %s", strings.Join(verbs, " "), err, tailOf(out, 500))
	}
	return string(out), nil
}

// runTool supervises one repository tool invocation and returns its
// combined output.
func runTool(arguments ...string) (string, error) {
	var combined bytes.Buffer
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path:   arguments[0],
		Env:    loopEnvironment(),
		Args:   arguments[1:],
		Stdout: &combined,
		Stderr: &combined,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("exit status %d", receipt.ExitCode)
	}
	return combined.String(), err
}

func tailOf(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value
	}
	return value[len(value)-limit:]
}

// Every tool and worker subprocess inherits unattended scope, including gates.
func loopEnvironment() []string {
	return append(os.Environ(), plan.AutomationModeEnvironment+"="+plan.ExecutionUnattended)
}

// The loop executes against its current registered checkout.
const loopWorktree = "."
