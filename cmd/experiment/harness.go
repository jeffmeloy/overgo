package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/loop"
)

// Strategy is one competing worker configuration.
type Strategy struct {
	Name   string   `json:"name"`
	Worker []string `json:"worker"`
}

// Spec declares one controlled comparison: one plan step, one baseline
// commit, and the strategies competing to implement it.
type Spec struct {
	Step           string     `json:"step"`
	Baseline       string     `json:"baseline"`
	Strategies     []Strategy `json:"strategies"`
	TimeoutMinutes int        `json:"timeout_minutes"`
}

// Trial is one strategy's measured outcome. A failed trial is a
// durable counterexample, never discarded.
type Trial struct {
	Strategy     string `json:"strategy"`
	Worktree     string `json:"worktree"`
	VerifyPassed bool   `json:"verify_passed"`
	VerifyOutput string `json:"verify_output,omitempty"`
	WallNS       uint64 `json:"wall_ns"`
	DiffFiles    int    `json:"diff_files"`
	LaunchError  string `json:"launch_error,omitempty"`
}

// Report is the durable comparison: every trial plus the selection.
type Report struct {
	Step     string  `json:"step"`
	Baseline string  `json:"baseline"`
	Trials   []Trial `json:"trials"`
	// Winner names the selected strategy: verify outcome first, then
	// the lowest measured wall. Empty when no trial verified.
	Winner string `json:"winner,omitempty"`
}

// harness is the pluggable world: the real one shells out to git, the
// worker, and the step verify; tests supply fakes.
type harness interface {
	CreateWorktree(strategy, baseline string) (string, error)
	Prompt(worktree string) (string, error)
	RunWorker(worktree string, strategy Strategy, prompt string, timeout time.Duration) error
	Verify(worktree string) (bool, string)
	DiffFiles(worktree string) int
}

// experimentDefaultTimeout bounds one worker trial when the spec does
// not declare its own budget.
const experimentDefaultTimeout = 90 * time.Minute

// verifyOutputTail bounds a recorded verify transcript: the tail
// carries the verdict and its immediate cause, and the full transcript
// stays in the trial worktree.
const verifyOutputTail = 2000

// Run executes every strategy against the same baseline and selects by
// verify outcome then measured cost. Trials are sequential: the
// comparison controls for machine load by never racing strategies.
func Run(world harness, spec Spec) (Report, error) {
	if spec.Step == "" || spec.Baseline == "" || len(spec.Strategies) == 0 {
		return Report{}, errors.New("experiment: spec requires a step, a baseline, and strategies")
	}
	timeout := experimentDefaultTimeout
	if spec.TimeoutMinutes > 0 {
		timeout = time.Duration(spec.TimeoutMinutes) * time.Minute
	}
	report := Report{Step: spec.Step, Baseline: spec.Baseline}
	seen := map[string]bool{}
	for _, strategy := range spec.Strategies {
		name := loop.StrategyIdentity(strategy.Worker, strategy.Name)
		if name == "" || seen[name] {
			return Report{}, fmt.Errorf("experiment: strategy identity %q is absent or duplicated", name)
		}
		seen[name] = true
		trial := Trial{Strategy: name}
		worktree, err := world.CreateWorktree(name, spec.Baseline)
		if err != nil {
			return Report{}, fmt.Errorf("experiment: worktree for %s: %w", name, err)
		}
		trial.Worktree = worktree
		prompt, err := world.Prompt(worktree)
		if err != nil {
			return Report{}, fmt.Errorf("experiment: prompt in %s: %w", worktree, err)
		}
		start := time.Now()
		if err := world.RunWorker(worktree, strategy, prompt, timeout); err != nil {
			// A worker that cannot run is a recorded counterexample,
			// not a discarded trial: the comparison stays complete.
			trial.LaunchError = err.Error()
		} else {
			trial.VerifyPassed, trial.VerifyOutput = world.Verify(worktree)
		}
		trial.WallNS = uint64(time.Since(start).Nanoseconds())
		trial.DiffFiles = world.DiffFiles(worktree)
		report.Trials = append(report.Trials, trial)
	}
	for _, trial := range report.Trials {
		if !trial.VerifyPassed {
			continue
		}
		if report.Winner == "" || trial.WallNS < winnerWall(report) {
			report.Winner = trial.Strategy
		}
	}
	return report, nil
}

func winnerWall(report Report) uint64 {
	for _, trial := range report.Trials {
		if trial.Strategy == report.Winner {
			return trial.WallNS
		}
	}
	return 0
}

// execHarness is the real world: git worktrees under tmp/, the plan's
// own prompt and verify commands run inside each worktree, and the
// strategy identity exported to the worker exactly as cmd/loop does.
type execHarness struct{}

// CreateWorktree adds one detached worktree at the baseline commit.
func (execHarness) CreateWorktree(strategy, baseline string) (string, error) {
	root := filepath.Join("tmp", "experiments", strategy)
	if err := os.MkdirAll(filepath.Dir(root), clioptions.OutputDirectoryMode); err != nil {
		return "", err
	}
	if _, err := os.Stat(root); err == nil {
		return "", fmt.Errorf("experiment: worktree %s already exists; remove it or name the strategy differently", root)
	}
	output, err := exec.Command("git", "worktree", "add", "--detach", root, baseline).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git worktree add: %w\n%s", err, output)
	}
	return root, nil
}

// Prompt renders the dispatched step prompt inside the trial worktree.
func (execHarness) Prompt(worktree string) (string, error) {
	command := exec.Command("go", "run", "./cmd/plan", "-prompt")
	command.Dir = worktree
	output, err := command.Output()
	if err != nil {
		return "", err
	}
	return string(output), nil
}

// RunWorker executes one strategy worker in its worktree under the
// trial budget, with the strategy identity exported to its gate runs.
func (execHarness) RunWorker(worktree string, strategy Strategy, prompt string, timeout time.Duration) error {
	if len(strategy.Worker) == 0 {
		return errors.New("strategy declares no worker command")
	}
	arguments := make([]string, len(strategy.Worker)-1)
	for index, argument := range strategy.Worker[1:] {
		arguments[index] = strings.ReplaceAll(argument, "{prompt}", prompt)
	}
	command := exec.Command(strategy.Worker[0], arguments...)
	command.Dir = worktree
	command.Stdin = strings.NewReader(prompt)
	command.Env = append(os.Environ(),
		loop.StrategyEnvironment+"="+loop.StrategyIdentity(strategy.Worker, strategy.Name))
	done := make(chan error, 1)
	if err := command.Start(); err != nil {
		return err
	}
	go func() { done <- command.Wait() }()
	select {
	case err := <-done:
		return err
	case <-time.After(timeout):
		_ = command.Process.Kill()
		return errors.New("worker timed out")
	}
}

// Verify runs the dispatched step machine verify inside the worktree.
func (execHarness) Verify(worktree string) (bool, string) {
	command := exec.Command("go", "run", "./cmd/plan", "-verify")
	command.Dir = worktree
	output, err := command.CombinedOutput()
	tail := string(output)
	if len(tail) > verifyOutputTail {
		tail = tail[len(tail)-verifyOutputTail:]
	}
	return err == nil, tail
}

// DiffFiles observes the trial worktree change size.
func (execHarness) DiffFiles(worktree string) int {
	output, err := exec.Command("git", "-C", worktree, "status", "--porcelain").Output()
	if err != nil {
		return 0
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return 0
	}
	return len(lines)
}
