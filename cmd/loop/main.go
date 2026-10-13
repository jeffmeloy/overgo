// Command loop runs the harness-agnostic plan->implement->verify->commit
// driver. The worker is any command that accepts the dispatched prompt --
// claude, a GPT CLI, anything -- named in a machine-local loop config; the
// plan, verifier, gate, and findings stay the repository's deterministic Go.
// There is no turn concept: worker exit is an event, continuity is this loop.
//
//	go run ./cmd/loop
//
// Config (machine-local, in the checkout's process state at
// .overgo-runtime/loop.json unless -config names another file):
//
//	{
//	  "worker": ["claude", "-p", "{prompt}"],
//	  "strategy_id": "profile:sha256:<digest>",
//	  "worker_timeout_minutes": 90
//	}
//
// Pause by creating .overgo-runtime/loop_pause; delete it to resume.
//
//	go run ./cmd/loop -leftovers
//
// lists every place files accumulate and removes nothing.
//
//	go run ./cmd/loop -leftovers -delete <place> [-confirm <token>]
//
// lists the finished entries of one place with a token, and removes them when
// it is given that token back.
//
//	go run ./cmd/loop -leftovers -withdraw <directory> [-confirm <token>]
//
// lists the locations the store registers for the checkpoints of one run with
// a token, and withdraws them, and no file, when it is given that token back.
//
//	go run ./cmd/loop -closeout
//
// is a campaign's close-out check: the listing without sizes, which fails
// while a finished entry remains on a line whose action is delete.
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
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// pushRemote is the remote -push publishes the named commit to.
const pushRemote = "origin"

// The loop's pause marker lives in the checkout's process state; the legacy
// path under docs is carried across on first use.
const (
	pauseMarker       = "loop_pause"
	legacyPauseMarker = "docs/.loop_pause"
)

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
	Repository           string `json:"repository"`
	WorkerTimeoutMinutes int    `json:"worker_timeout_minutes"`
	// AutoCycleLandings, when positive, starts an improvement cycle between steps once that
	// many landings have moved what a cycle measures since the last cycle that completed: the
	// count at which finding which of them broke a failed cycle would cost more than running
	// one, which the operator sets from the cycle's measured wall (about 1 h 40 min) and a
	// landing's. Zero starts a cycle only when the plan empties and something moved.
	AutoCycleLandings int `json:"auto_cycle_landings"`
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "loop:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	flags := flag.NewFlagSet("loop", flag.ContinueOnError)
	configPath := flags.String("config", "", "machine-local loop configuration (default .overgo-runtime/loop.json)")
	repoPath := flags.String("repo", "", "OvergoDB root for the evidence doors; empty resolves the store through the data-root contract")
	publishStrategySpec := flags.String("publish-strategy", "", "publish one strategy from this spec (worker, catalog, loop config) and exit")
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
	resumeMediaExperimentSpec := flags.String("resume-media-experiment", "", "resume one media experiment from this spec, reusing retained acquisitions and publishing the report without opening a model")
	landReference := flags.String("land", "", "land one row through the small gate over the dirty tree and confirm the landing from git: -land <item>/<step> -message-file <file>")
	landMessage := flags.String("message-file", "", "with -land: the gate commit message file; - reads the message from standard input")
	improveOnce := flags.Bool("improve", false, "run one improvement cycle now, as an emptied plan does: measure the scorecard, compare each cell with its newest earlier complete measurement, and file the declines and the other gaps as rows")
	pushCommit := flags.String("push", "", "the owner's command: push the named commit to origin's main, fast-forward only; refused from a worker's environment and without -confirm-push")
	confirmPush := flags.Bool("confirm-push", false, "with -push: say the push is wanted")
	leftovers := flags.Bool("leftovers", false, "list every place files accumulate (helper copies, worktrees, branches, stashes, experiment outputs, temporary folders, bin, tmp, store backups), one line an entry with its bytes and the rule that decided it, and every directory no line of the table names; writes the same listing to the trunk checkout's process state and removes nothing")
	deletePlace := flags.String("delete", "", "with -leftovers: list the finished entries of one place whose finished entries are deleted (helper-copy, branch, experiment, temporary-build, tmp), each with its bytes and its rule, and print a token; removes nothing without -confirm, and is refused from a worker's environment for the owner's data, the places experiment and branch")
	withdrawRun := flags.String("withdraw", "", "with -leftovers: list the locations the store registers for the checkpoints in one run's output directory, which lies in the experiments folder and is named itself by a committed record, and print a token; withdraws nothing without -confirm, never a location of a model no checkpoint of the run produced, and nothing when a served model or such a model would be left a component with no location; the owner's data, so refused from a worker's environment")
	confirmToken := flags.String("confirm", "", "with -leftovers -delete or -withdraw: the token the listing printed; removes or withdraws exactly the entries that listing showed, a line each, appended to a receipt in the trunk checkout's process state, and is refused when the listing is no longer the same")
	closeout := flags.Bool("closeout", false, "the campaign's close-out check: list every place files accumulate, without sizes, write the listing to the trunk checkout's process state, and fail while an entry of a line whose action is delete is finished; removes nothing, and takes no -delete and no -withdraw")
	if err := flags.Parse(args); err != nil {
		return err
	}
	// A close-out is never a removal: with -closeout a place or a run's directory is refused as it is without -leftovers.
	if err := removalFlags(*leftovers && !*closeout, *deletePlace, *withdrawRun, *confirmToken); err != nil {
		return err
	}
	if *landReference != "" {
		return landCommand(*landReference, *landMessage, os.Stdin, os.Stdout)
	}
	if *leftovers || *closeout {
		return leftoversFor(*repoPath, *deletePlace, *withdrawRun, *confirmToken, os.Getenv(plan.AutomationWorkerEnvironment) != "", *closeout, os.Stdout)
	}
	resolved, err := dataroot.StoreRoot(*repoPath)
	if err != nil {
		return err
	}
	*repoPath = resolved
	switch {
	case *pushCommit != "":
		commit, err := pushNamed(loopWorktree, *pushCommit, pushRemote, *confirmPush, os.Getenv(plan.AutomationWorkerEnvironment) != "")
		if err == nil {
			fmt.Printf("loop: %s/main holds %s\n", pushRemote, commit)
		}
		return err
	case *improveOnce:
		added, err := loop.Improve(improvementFor(*repoPath))
		fmt.Printf("loop: improvement cycle proposed=%d row(s)\n", added)
		return err
	case *publishStrategySpec != "":
		return publishStrategy(*repoPath, *publishStrategySpec, os.Stdout)
	case *resumeMediaExperimentSpec != "":
		return resumeMediaExperiment(*repoPath, *resumeMediaExperimentSpec, os.Stdout)
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
	if *configPath == "" {
		// A state directory that cannot be made leaves no path; loading fails below.
		*configPath, _ = processlock.StateFile(".", processcontrol.LoopConfigFile, processcontrol.LegacyLoopConfigFile)
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
	store, err := overgodb.OpenReadOnly(gitauthority.CanonicalOvergoDBDirectory)
	if err != nil {
		return err
	}
	defer store.Close()
	prompt, found, err := artifact.ReadContent(context.Background(), store, strategy.Prompt)
	if err != nil || !found {
		return errors.Join(errors.New("loop: configured strategy prompt is unavailable"), err)
	}
	// An empty plan runs the full scorecard and turns the gaps into the next rows.
	world := &execWorld{config: loaded, strategy: strategy, promptPrefix: string(prompt.Data), campaign: campaign, improvement: improvementFor(*repoPath)}
	world.improvement.limit = loaded.AutoCycleLandings
	outcome, err := loop.Run(world, strategy.Loop)
	fmt.Printf("loop: %s after %d worker invocation(s); parked=%v; audit: the gate remains the sole commit path, workers cannot advance an unverified step\n",
		outcome.Reason, outcome.Invocations, outcome.Parked)
	return err
}

// removalFlags refuses the flags of a removal where they do not belong: a
// place to delete from, or a run's directory to withdraw, goes with
// -leftovers, a token with one of the two, and the two never together, so a
// flag mistyped or left out never turns a removal into another command, or a
// listing into one.
func removalFlags(leftovers bool, place, directory, token string) error {
	if named := place + directory; named != "" && !leftovers || token != "" && named == "" || place != "" && directory != "" {
		return errors.New("usage: loop -leftovers (-delete <place> | -withdraw <directory>) [-confirm <token>]")
	}
	return nil
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
	improvement
	config       config
	strategy     loop.Strategy
	promptPrefix string
	campaign     *processcontrol.Campaign
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

// awaitValidation waits for a live plan or gate mutation to release its
// ownership before the driver reads the plan.
func (w *execWorld) awaitValidation() error {
	return authoritylock.Wait(context.Background(), loopWorktree)
}

func (w *execWorld) Prompt(loop.Step) (string, error) {
	return planCommand("-prompt")
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
	// A pause marker the state directory cannot hold pauses: the kill switch
	// fails closed.
	marker, err := processlock.StateFile(".", pauseMarker, legacyPauseMarker)
	if err != nil {
		return true
	}
	if _, err := os.Stat(marker); err == nil {
		return true
	}
	stop, err := plan.ReadStop(context.Background(), loopWorktree, "", plan.ExecutionUnattended)
	if err != nil {
		fmt.Fprintln(os.Stderr, "loop stop authority:", err)
		return true
	}
	if stop.Blocked {
		fmt.Fprintln(os.Stderr, stop.String())
	}
	return stop.Blocked
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
	err := runToolEnv(loopEnvironment(), nil, &combined, &combined, arguments...)
	return combined.String(), err
}

// runToolEnv supervises one tool under the caller's environment and streams.
func runToolEnv(environment []string, input io.Reader, stdout, stderr io.Writer, arguments ...string) error {
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path:   arguments[0],
		Env:    environment,
		Args:   arguments[1:],
		Stdin:  input,
		Stdout: stdout,
		Stderr: stderr,
	})
	if err == nil && receipt.ExitCode != 0 {
		err = toolExit(receipt.ExitCode)
	}
	return err
}

// toolExit is a tool's nonzero exit status; the resource-busy status is
// processcontrol.ErrResourceBusy, so a caller tells contention from failure.
type toolExit int

// Error reads as the process status.
func (code toolExit) Error() string { return fmt.Sprintf("exit status %d", int(code)) }

// Is reports the resource-busy status as processcontrol.ErrResourceBusy.
func (code toolExit) Is(target error) bool {
	return target == processcontrol.ErrResourceBusy && int(code) == processcontrol.ResourceBusyExitCode
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
