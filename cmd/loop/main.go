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
//	  "name": "claude-master-lead",
//	  "prompt": "loop_prompt.md",
//	  "max_attempts_per_step": 3,
//	  "max_invocations": 40
//	}
//
// The prompt file, beside the config, opens every worker's prompt; the name is the
// worker's identity, to which its landings are held; the two budgets bound the run. A worker
// runs until it exits or a stop ends it: a landing inside it can outlast any fixed bound.
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
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/guard"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
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
	// Worker is the command each dispatched prompt goes to, as {prompt} in its arguments and on its standard input.
	Worker []string `json:"worker"`
	// Workers are the stronger workers, in order, a step goes to after Worker spends its attempts on it, each written
	// as Worker is; after the last the step is set aside with a finding.
	Workers [][]string `json:"workers,omitempty"`
	// Name is the worker's stable identity: the campaign records it, and the worker's landings are held to the step
	// dispatch names for it.
	Name string `json:"name"`
	// Prompt is the file, beside the config, whose text opens every worker's prompt.
	Prompt string `json:"prompt"`
	// Environment is what the driver sets in its own environment before it starts, the data roots and the role among
	// it, since a scheduled task that starts the driver sets none.
	Environment map[string]string `json:"environment,omitempty"`
	// Config is the run's two budgets: the attempts a step takes and the worker's invocations.
	loop.Config
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
	landReference := flags.String("land", "", "land one row through the small gate over the dirty tree and confirm the landing from git: -land <item>/<step>; the landing writes the commit message")
	landMessage := flags.String("message-file", "", "with -land: optional words of the lead the landing's message opens with, from a file; - reads them from standard input")
	improveOnce := flags.Bool("improve", false, "run one improvement cycle now if one is due (HandStartDue says why when none is), as an emptied plan does: measure the scorecard, judge each cell against what its earlier readings held, and file the declines and the other gaps as rows")
	statusView := flags.Bool("status", false, "read the state of the work in this checkout, the dispatched step, the last commit, the decisions waiting for an answer and the week's pace, keep it in the process state as status.json and print it; runs nothing")
	sessionPrompt := flags.Bool("session-prompt", false, "print the instructions a session the Claude app's scheduled task starts follows, with the environment its commands run in from loop.json; runs nothing")
	toolsServer := flags.Bool("tools", false, "serve the worker its six tools over the Model Context Protocol on standard input and output: read, edit within the step's files, run its check, run a package's tests, show the step and run an approved probe")
	startup := flags.Bool("install-startup", false, "register a Windows scheduled task that runs go run ./cmd/loop in this checkout at the user's logon and again after it fails; replaces a task of its name")
	initialize := flags.Bool("init", false, "write the driver's machine configuration, loop.json and its worker's prompt, into the checkout's process state from the defaults the repository holds; writes over neither")
	pushCommit := flags.String("push", "", "the owner's command: push the named commit to origin's main, fast-forward only; refused from a worker's environment and without -confirm-push")
	confirmPush := flags.Bool("confirm-push", false, "with -push: say the push is wanted")
	leftovers := flags.Bool("leftovers", false, "list every place files accumulate (helper copies, worktrees, branches, stashes, experiment outputs, temporary folders, bin, tmp, store backups), one line an entry with its bytes and the rule that decided it, and every directory no line of the table names; writes the same listing to the trunk checkout's process state and removes nothing")
	deletePlace := flags.String("delete", "", "with -leftovers: list the finished entries of one place whose finished entries are deleted (helper-copy, branch, experiment, temporary-build, build-cache, tmp), each with its bytes and its rule, and print a token; removes nothing without -confirm. A confirmed commit clears the same places with no token")
	withdrawRun := flags.String("withdraw", "", "with -leftovers: list the locations the store registers for the checkpoints in one run's output directory, which lies in the experiments folder and is named itself by a committed record, and print a token; withdraws nothing without -confirm, never a location of a model no checkpoint of the run produced, and nothing when a served model or such a model would be left a component with no location")
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
		return leftoversFor(*repoPath, *deletePlace, *withdrawRun, *confirmToken, *closeout, os.Stdout)
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
		// A cycle started by hand is due by the trigger's terms, never for a fix its own check judges (HandStartDue).
		world := improvementFor(*repoPath)
		dispatch, err := dispatched()
		if err != nil {
			return err
		}
		due, reason, err := world.HandStartDue(dispatch)
		if err != nil {
			return err
		}
		if !due {
			return fmt.Errorf("loop: no measurement run is due: %s", reason)
		}
		added, err := loop.Improve(world)
		fmt.Printf("loop: improvement cycle proposed=%d row(s)\n", added)
		return err
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
	switch {
	case *initialize:
		return writeDefaultConfig(*configPath, os.Stdout)
	case *toolsServer:
		return serveTools(os.Stdin, os.Stdout, workerTools(loopWorktree, stepFiles))
	case *sessionPrompt:
		return writeSessionPrompt(*configPath, os.Stdout)
	case *statusView:
		return writeStatus(loopWorktree, os.Stdout)
	case *startup:
		root, err := filepath.Abs(loopWorktree)
		if err != nil {
			return err
		}
		return installStartup(root, os.Stdout)
	}
	var loaded config
	if err := jsonfile.DecodeStrict(*configPath, &loaded); err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	if len(loaded.Worker) == 0 || loaded.Prompt == "" {
		return errors.New("config names no worker command or no prompt file")
	}
	prompt, err := os.ReadFile(filepath.Join(filepath.Dir(*configPath), loaded.Prompt))
	if err != nil {
		return fmt.Errorf("load prompt: %w", err)
	}
	// A worker takes no hold by its prompt: one that opened with the takeover overwrote a working session's (SQA-337).
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(string(prompt))), guard.TakeoverDeclaration) {
		return fmt.Errorf("loop: the worker prompt %s opens with %q, which takes a working session's hold; delete those words and start again",
			loaded.Prompt, guard.TakeoverDeclaration)
	}
	for name, value := range loaded.Environment {
		if err := os.Setenv(name, value); err != nil {
			return err
		}
	}
	campaign, err := processcontrol.BeginCampaign(loopWorktree, strings.TrimSpace(loaded.Name))
	if err != nil {
		return err
	}
	defer campaign.Close()
	// Only an admitted start reads the last run's record: a refused one never erases a working run's (SQA-338).
	if err := noteUncleanEnd(loopWorktree, time.Now(), os.Stdout); err != nil {
		return err
	}
	// The worker and every gate inside it act under the config's name.
	priorWorker, workerPresent := os.LookupEnv(plan.AutomationWorkerEnvironment)
	if err := os.Setenv(plan.AutomationWorkerEnvironment, strings.TrimSpace(loaded.Name)); err != nil {
		return err
	}
	defer func() {
		if workerPresent {
			_ = os.Setenv(plan.AutomationWorkerEnvironment, priorWorker)
		} else {
			_ = os.Unsetenv(plan.AutomationWorkerEnvironment)
		}
	}()
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
	// An empty plan runs the full scorecard and turns the gaps into the next rows.
	world := &execWorld{config: loaded, promptPrefix: string(prompt), campaign: campaign, improvement: improvementFor(*repoPath)}
	outcome, err := loop.Run(world, loaded.Config)
	fmt.Printf("loop: %s after %d worker invocation(s); parked=%v; findings to file=%v; audit: the gate remains the sole commit path, workers cannot advance an unverified step\n",
		outcome.Reason, outcome.Invocations, outcome.Parked, outcome.Findings)
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

// execWorld adapts the driver to the repository's real owners.
type execWorld struct {
	improvement
	config       config
	promptPrefix string
	campaign     *processcontrol.Campaign
	// level is the worker the next runs go to: 0 for Worker, then each of Workers in turn (loop.LadderWorld).
	level int
}

func (w *execWorld) Current() (loop.Step, bool, error) {
	if err := w.awaitValidation(); err != nil {
		return loop.Step{}, false, err
	}
	dispatch, err := dispatched()
	if err != nil {
		return loop.Step{}, false, err
	}
	if dispatch.Complete {
		return loop.Step{}, false, nil
	}
	if dispatch.Waiting != "" {
		return loop.Step{}, false, fmt.Errorf("%w: %s", loop.ErrWorkWaiting, dispatch.Waiting)
	}
	step, err := dispatchedStep(dispatch)
	return loop.Step{Item: dispatch.Item, ID: dispatch.Step, Run: step.Run, Record: step.Record}, err == nil, err
}

// dispatchedStep is the dispatched step as the plan holds it, its command, record and files among it, which the
// dispatch line does not carry.
func dispatchedStep(dispatch plan.Dispatch) (plan.Step, error) {
	document, err := plan.Load("")
	step, _ := plan.ExactStep(document, dispatch.Item, dispatch.Step)
	return step, err
}

// stepFiles are the files the dispatched step lists, which the worker's edits stay inside.
func stepFiles() ([]string, error) {
	dispatch, err := dispatched()
	if err != nil {
		return nil, err
	}
	step, err := dispatchedStep(dispatch)
	return step.Files, err
}

// dispatched is the plan's current row, as data: the prose line is never parsed back.
func dispatched() (plan.Dispatch, error) {
	out, err := planCommand("-prompt", "-json")
	if err != nil {
		return plan.Dispatch{}, err
	}
	var dispatch plan.Dispatch
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &dispatch); err != nil {
		return plan.Dispatch{}, fmt.Errorf("unparseable dispatch %q: %w", strings.TrimSpace(out), err)
	}
	return dispatch, nil
}

// awaitValidation waits for a live plan or gate mutation to release its
// ownership before the driver reads the plan.
func (w *execWorld) awaitValidation() error {
	return authoritylock.Wait(context.Background(), loopWorktree)
}

func (w *execWorld) Prompt(loop.Step) (string, error) {
	return planCommand("-prompt")
}

// Workers counts the workers a step can go to: Worker, then each of Workers.
func (w *execWorld) Workers() int { return 1 + len(w.config.Workers) }

// UseWorker sends the next runs to the worker at level.
func (w *execWorld) UseWorker(level int) { w.level = level }

func (w *execWorld) RunWorker(step loop.Step, prompt, feedback string) (string, error) {
	text := prompt
	if w.promptPrefix != "" {
		text = w.promptPrefix + "\n\n" + text
	}
	if feedback != "" {
		text += "\n\nPREVIOUS ATTEMPT FEEDBACK (fix this, then commit through the gate):\n" + feedback
	}
	worker := w.config.Worker
	if w.level > 0 {
		worker = w.config.Workers[w.level-1]
	}
	arguments := make([]string, len(worker)-1)
	for index, argument := range worker[1:] {
		arguments[index] = strings.ReplaceAll(argument, "{prompt}", text)
	}
	ctx, cancel := context.WithCancel(context.Background())
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
	// The record a worker leaves while it works says, at the next start, what a run that never exited was doing.
	if err := keepRunning(loopWorktree, step.Key(), time.Now()); err != nil {
		cancel()
		<-watchDone
		return "", err
	}
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path:   worker[0],
		Args:   arguments,
		Env:    workerEnvironment(loopEnvironment()),
		Stdin:  strings.NewReader(text),
		Stdout: output,
		Stderr: output,
	})
	cancel()
	err = errors.Join(err, dropRunning(loopWorktree))
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

// RunCommand works a run step with no worker: it runs the step's command in the checkout, requires the record the step
// names written, and lands the step through the gate, whose landing writes the message.
func (w *execWorld) RunCommand(step loop.Step) (string, error) {
	shell, err := clioptions.POSIXShell()
	if err != nil {
		return "", &loop.LaunchError{Err: err}
	}
	out, err := runTool(shell, "-c", step.Run)
	if err != nil {
		return out, err
	}
	written, err := gitauthority.Query(context.Background(), loopWorktree, "status", "--porcelain", "--", step.Record)
	if err != nil {
		return out, err
	}
	if strings.TrimSpace(string(written)) == "" {
		return out, fmt.Errorf("loop: %s ran and left its record %s as it was", step.Key(), step.Record)
	}
	landed, err := landRow(&execRowWorld{}, step.Key(), "")
	return landed.Detail, err
}

func (w *execWorld) Verify(loop.Step) (string, error) {
	out, err := runTool("go", "run", "./cmd/plan", "-verify")
	if err == nil {
		return "", nil
	}
	return tailOf(out, 4000), nil
}

func (w *execWorld) Park(step loop.Step, reason string) error {
	if err := plan.ParkRow(context.Background(), loopWorktree, step.Key()); err != nil {
		return fmt.Errorf("park row: %w", err)
	}
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
