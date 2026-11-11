package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
)

// A replay's checkout, beside the cycle's in the process state: each landed step is checked out there at its parent.
const (
	replayCheckout = "replay"
	replayLock     = replayCheckout + ".lock"
	// replayFailure is how much of a check's failure the next attempt is told, as the driver tells its next worker.
	replayFailure = 2000
	// defaultReplayLimit is the most steps a replay takes unless told otherwise, newest first.
	defaultReplayLimit = 50
	// landedRowTrailer is the trailer a landing names each step it landed by.
	landedRowTrailer = "Landed-Row"
)

// replayed is a landed step as a task with a known answer: the commit that landed it, the commit it landed on, and the
// step as that parent's plan held it.
type replayed struct {
	reference, landed, parent string
	step                      plan.Step
}

// replayRecord is one worker's replay of one landed step: whether the step's own check passed after it, in how many
// attempts, with how many tool calls and in what wall.
type replayRecord struct {
	Worker   string `json:"worker"`
	Kind     string `json:"kind"`
	Step     string `json:"step"`
	Landed   string `json:"landed"`
	Passed   bool   `json:"passed"`
	Attempts int    `json:"attempts"`
	Calls    int    `json:"tool_calls"`
	WallMS   int64  `json:"wall_ms"`
	Failure  string `json:"failure,omitzero"`
	// Skipped says why a step was not put to the worker: its check passed at its parent, or the replay could not set
	// it up. A skipped step counts neither way.
	Skipped string `json:"skipped,omitzero"`
}

// replayWorker is a worker a replay runs: it works a task in the replay checkout.
type replayWorker struct {
	name string
	work func(ctx context.Context, checkout string, task replayTask) (localWork, error)
}

// replayTask is what a replay hands a worker for one step: the prompt, the step's check, the files it may change and
// the tests the replay holds as they landed. A worker in a process of its own is served the task's tools by
// cmd/loop -tools -tools-step, which reads it from a file.
type replayTask struct {
	Prompt string   `json:"prompt"`
	Verify string   `json:"verify"`
	Files  []string `json:"files"`
	Tests  []string `json:"tests"`
}

// check runs the task's check in checkout, answering with its failure where it fails.
func (task replayTask) check(ctx context.Context, checkout string) (string, error) {
	if _, err := planverify.Execute(ctx, checkout, task.Verify, nil); err != nil {
		return err.Error(), err
	}
	return "the check passed", nil
}

// tools are the worker's tools over checkout serving the task, less the probes, which name the trunk's commands: the
// step shown as the prompt, its check run in checkout, and an edit of a held test refused.
func (task replayTask) tools(ctx context.Context, checkout string) []workerTool {
	tools := workerTools(checkout, workerStep{
		show:   func() (string, error) { return task.Prompt, nil },
		check:  func() (string, error) { return task.check(ctx, checkout) },
		verify: func() (string, error) { return task.Verify, nil },
		files:  func() ([]string, error) { return task.Files, nil },
	})
	tools = slices.DeleteFunc(tools, func(tool workerTool) bool { return tool.name == "run_probe" })
	// A held test is neither edited nor deleted.
	for index, tool := range tools {
		if tool.name != "edit_file" && tool.name != "delete_file" {
			continue
		}
		changes := tool.run
		tools[index].run = func(arguments map[string]string) (string, error) {
			if slices.Contains(task.Tests, filepath.ToSlash(filepath.Clean(arguments["path"]))) {
				return "", fmt.Errorf("%s is the step's check, which the replay holds as it landed", arguments["path"])
			}
			return changes(arguments)
		}
	}
	return tools
}

// landedSteps are the steps landed in the checkout at root since the date, newest first, whose check, in the plan of
// the commit each landed on, runs one Go test.
func landedSteps(ctx context.Context, root, since string) ([]replayed, error) {
	log, err := gitauthority.Query(ctx, root, "log", "--since="+since, "--grep="+landedRowTrailer+":",
		"--format=%H %P%x00%(trailers:key="+landedRowTrailer+",valueonly,separator=%x2C)")
	if err != nil {
		return nil, err
	}
	var steps []replayed
	for line := range strings.Lines(strings.TrimSpace(string(log))) {
		commits, rows, _ := strings.Cut(strings.TrimSpace(line), "\x00")
		landed, parents, _ := strings.Cut(commits, " ")
		parent, _, _ := strings.Cut(parents, " ")
		if parent == "" || strings.TrimSpace(rows) == "" {
			continue
		}
		data, err := gitauthority.Query(ctx, root, "show", parent+":docs/plan.json")
		if err != nil {
			continue
		}
		held, err := plan.Parse(data)
		if err != nil {
			continue
		}
		for reference := range strings.SplitSeq(rows, ",") {
			item, id, _ := strings.Cut(strings.TrimSpace(reference), "/")
			for _, candidate := range held.Items {
				at := slices.IndexFunc(candidate.Steps, func(step plan.Step) bool { return step.ID == id })
				if candidate.ID == item && at >= 0 && oneTest(candidate.Steps[at].Verify) {
					steps = append(steps, replayed{reference: item + "/" + id, landed: landed, parent: parent, step: candidate.Steps[at]})
				}
			}
		}
	}
	return steps, nil
}

// oneTest is whether a check runs one Go test, named whole.
func oneTest(verify string) bool {
	parsed, err := runrecord.ParseVerify(verify)
	if err != nil {
		return false
	}
	tests := parsed.GoTests()
	if len(tests) != 1 || len(tests[0].Packages) != 1 || tests[0].Target == nil {
		return false
	}
	pattern := tests[0].Target.Pattern
	return strings.HasPrefix(pattern, "^Test") && strings.HasSuffix(pattern, "$") && !strings.ContainsAny(pattern, "|/")
}

// prompt is the step as a worker is handed it: as the plan showed it, with its reasoning.
func (r replayed) prompt() string {
	var text strings.Builder
	fmt.Fprintf(&text, "TASK %s\n%s\nVERIFY %s\n", r.reference, r.step.Title, r.step.Verify)
	if len(r.step.Files) != 0 {
		fmt.Fprintf(&text, "FILES %s: the change writes these and nothing else\n", strings.Join(r.step.Files, " "))
	}
	fmt.Fprintf(&text, "\n%s\n", r.step.Rationale)
	return text.String()
}

// tests are the test files the landing changed: put in place before a worker starts, so the step's check is the test
// that held the landing, and no worker's to edit.
func (r replayed) tests(ctx context.Context, root string) ([]string, error) {
	changed, err := gitauthority.Query(ctx, root, "diff", "--name-only", "--diff-filter=AM", r.parent, r.landed, "--", "*_test.go")
	if err != nil {
		return nil, err
	}
	return strings.Fields(string(changed)), nil
}

// moveReplay checks the replay checkout out at commit, adding the worktree on first use, discarding what an earlier
// replay's worker wrote there, and gives it the data roots the main checkout reads.
func moveReplay(root, checkout, commit string) error {
	if _, err := os.Lstat(filepath.Join(checkout, ".git")); errors.Is(err, os.ErrNotExist) {
		if err := gitWrite(root, "worktree", "add", "--quiet", "--detach", checkout, commit); err != nil {
			return err
		}
	} else if err := errors.Join(gitWrite(checkout, "checkout", "--quiet", "--detach", "--force", commit), gitWrite(checkout, "clean", "--quiet", "-d", "--force")); err != nil {
		return err
	}
	roots, err := os.ReadFile(filepath.Join(root, dataroot.ConfigFile))
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(checkout, dataroot.ConfigFile), roots, clioptions.PrivateFileMode)
}

// measurement is the records under docs/verification the landing of r wrote, which a model run on the card makes: a
// step whose check reads one is a measurement, which the worker's tools cannot make (loop-proof/local-demonstrated's
// vision-bridge steps failed every try of runs 2 to 4), so it is no task to replay.
func (r replayed) measurement(ctx context.Context, root string) ([]string, error) {
	names, err := gitauthority.Query(ctx, root, "diff", "--name-only", r.parent, r.landed, "--", "docs/verification")
	return strings.Fields(string(names)), err
}

// replay works step with worker in checkout, from the step's parent with the landing's tests in place: the check must
// fail there first, then the worker works up to attempts times, each after the first told the last failure.
func replay(ctx context.Context, root, checkout string, step replayed, worker replayWorker, attempts int) (replayRecord, error) {
	records, err := step.measurement(ctx, root)
	if err != nil || len(records) != 0 {
		return replayRecord{Worker: worker.name, Kind: stepKind(step.step.Files, step.step.NeedsCard), Step: step.reference, Landed: step.landed,
			Skipped: "a measurement: its landing wrote " + strings.Join(records, ", ") + ", which a model run makes and no worker's tools can"}, err
	}
	if err := moveReplay(root, checkout, step.parent); err != nil {
		return replayRecord{}, err
	}
	tests, err := step.tests(ctx, root)
	if err != nil {
		return replayRecord{}, err
	}
	for _, test := range tests {
		data, err := gitauthority.Query(ctx, root, "show", step.landed+":"+test)
		if err != nil {
			return replayRecord{}, err
		}
		if err := os.MkdirAll(filepath.Dir(filepath.Join(checkout, test)), clioptions.OutputDirectoryMode); err != nil {
			return replayRecord{}, err
		}
		if err := os.WriteFile(filepath.Join(checkout, test), data, clioptions.OutputFileMode); err != nil {
			return replayRecord{}, err
		}
	}
	task := replayTask{Prompt: step.prompt(), Verify: step.step.Verify, Files: step.step.Files, Tests: tests}
	record := replayRecord{Worker: worker.name, Kind: stepKind(step.step.Files, step.step.NeedsCard), Step: step.reference, Landed: step.landed}
	if _, err := task.check(ctx, checkout); err == nil {
		record.Skipped = fmt.Sprintf("the check passes at the parent %.12s with the landing's tests in place: the step is no task there", step.parent)
		return record, nil
	}
	started := time.Now()
	for record.Attempts < attempts && !record.Passed {
		record.Attempts++
		work, err := worker.work(ctx, checkout, task)
		record.Calls += work.Calls
		if ctx.Err() != nil {
			return record, ctx.Err()
		}
		failure, failed := task.check(ctx, checkout)
		record.Passed = failed == nil
		if !record.Passed {
			record.Failure = clioptions.Tail(errors.Join(err, errors.New(failure)).Error(), replayFailure)
			task.Prompt = step.prompt() + "\nThe last attempt left the check failing:\n" + record.Failure
		}
	}
	record.WallMS = time.Since(started).Milliseconds()
	return record, nil
}

// replaySteps replays the steps landed since the date, newest first, up to limit of them, with worker, writing each
// record as a line of JSON to out as it is made.
func replaySteps(ctx context.Context, root, since string, limit int, worker replayWorker, out io.Writer) error {
	steps, err := landedSteps(ctx, root, since)
	if err != nil {
		return err
	}
	state := filepath.Join(root, processlock.StateDirectory)
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	lock, err := processlock.Acquire(filepath.Join(state, replayLock), clioptions.PrivateFileMode)
	if err != nil {
		return fmt.Errorf("another replay holds the replay checkout: %w", err)
	}
	defer lock.Close()
	checkout, err := filepath.Abs(filepath.Join(state, replayCheckout))
	if err != nil {
		return err
	}
	encoder, passed, tasks := json.NewEncoder(out), 0, 0
	for _, step := range steps[:min(limit, len(steps))] {
		record, err := replay(ctx, root, checkout, step, worker, defaultAttempts)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err != nil {
			record.Worker, record.Step, record.Landed, record.Skipped = worker.name, step.reference, step.landed, err.Error()
		}
		if record.Skipped == "" {
			tasks++
		}
		if record.Passed {
			passed++
		}
		if err := encoder.Encode(record); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "loop: %s passed %d of %d replayed steps\n", worker.name, passed, tasks)
	return nil
}

// localReplayWorker is the local worker as a replay runs it: the model served at server, its tools in this process.
func localReplayWorker(server string, model workerModel, log io.Writer) replayWorker {
	return replayWorker{name: model.name, work: func(ctx context.Context, checkout string, task replayTask) (localWork, error) {
		return workLocally(ctx, server, model, task.Prompt, task.tools(ctx, checkout), log)
	}}
}

// loggedTools are tools that each append their name as a line to log when called.
func loggedTools(tools []workerTool, log string) []workerTool {
	logged := slices.Clone(tools)
	for index, tool := range logged {
		logged[index].run = func(arguments map[string]string) (string, error) {
			file, err := os.OpenFile(log, os.O_APPEND|os.O_CREATE|os.O_WRONLY, clioptions.PrivateFileMode)
			if err != nil {
				return "", err
			}
			_, err = fmt.Fprintln(file, tool.name)
			if err = errors.Join(err, file.Close()); err != nil {
				return "", err
			}
			return tool.run(arguments)
		}
	}
	return logged
}
