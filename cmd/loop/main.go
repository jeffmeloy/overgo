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
//	  "max_attempts_per_step": 3,
//	  "max_invocations": 20,
//	  "worker_timeout_minutes": 90
//	}
//
// Pause by creating docs/.loop_pause; delete it to resume.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/loop"
)

const pauseMarker = "docs/.loop_pause"

type config struct {
	Worker []string `json:"worker"`
	// Strategy optionally names this worker configuration; attempt
	// records carry it so history compares strategies. Unset derives a
	// stable digest of the worker command.
	Strategy             string `json:"strategy"`
	MaxAttemptsPerStep   int    `json:"max_attempts_per_step"`
	MaxInvocations       int    `json:"max_invocations"`
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
	if err := flags.Parse(args); err != nil {
		return err
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
	world := &execWorld{config: loaded}
	outcome, err := loop.Run(world, loop.Config{
		MaxAttemptsPerStep: loaded.MaxAttemptsPerStep,
		MaxInvocations:     loaded.MaxInvocations,
	})
	fmt.Printf("loop: %s after %d worker invocation(s); parked=%v; honesty: the gate remains the sole commit path, workers cannot advance an unverified step\n",
		outcome.Reason, outcome.Invocations, outcome.Parked)
	return err
}

// execWorld adapts the driver to the repository's real owners.
type execWorld struct {
	config config
}

func (w *execWorld) Current() (loop.Step, bool, error) {
	out, err := planCommand("-next")
	if err != nil {
		return loop.Step{}, false, err
	}
	line := strings.TrimSpace(out)
	if strings.HasPrefix(line, "plan complete") {
		return loop.Step{}, false, nil
	}
	head, _, ok := strings.Cut(line, ":")
	item, step, ok2 := strings.Cut(strings.TrimSpace(head), " / ")
	if !ok || !ok2 {
		return loop.Step{}, false, fmt.Errorf("unparseable dispatch %q", line)
	}
	return loop.Step{Item: strings.TrimSpace(item), ID: strings.TrimSpace(step)}, true, nil
}

func (w *execWorld) Prompt(loop.Step) (string, error) {
	return planCommand("-prompt")
}

func (w *execWorld) RunWorker(step loop.Step, prompt, feedback string) (string, error) {
	text := prompt
	if feedback != "" {
		text += "\n\nPREVIOUS ATTEMPT FEEDBACK (fix this, then commit through the gate):\n" + feedback
	}
	arguments := make([]string, len(w.config.Worker)-1)
	for index, argument := range w.config.Worker[1:] {
		arguments[index] = strings.ReplaceAll(argument, "{prompt}", text)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(w.config.WorkerTimeoutMinutes)*time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, w.config.Worker[0], arguments...)
	// The declared strategy identity reaches every gate run inside the
	// worker, so each attempt record states which strategy produced it.
	command.Env = append(os.Environ(),
		loop.StrategyEnvironment+"="+loop.StrategyIdentity(w.config.Worker, w.config.Strategy))
	command.Stdin = strings.NewReader(text)
	out, err := command.CombinedOutput()
	tail := tailOf(string(out), 4000)
	if err != nil && command.ProcessState == nil {
		return tail, &loop.LaunchError{Err: err}
	}
	// A nonzero or timed-out worker still exited; the plan decides what it
	// accomplished. Log the event and continue.
	fmt.Printf("loop: worker exited for %s (err=%v)\n", step.Key(), err)
	return tail, nil
}

func (w *execWorld) Verify(loop.Step) (string, error) {
	command := exec.Command("go", "run", "./cmd/plan", "-verify")
	out, err := command.CombinedOutput()
	if err == nil {
		return "", nil
	}
	return tailOf(string(out), 4000), nil
}

func (w *execWorld) Park(step loop.Step, reason string) error {
	command := exec.Command("go", "run", "./cmd/finding",
		"-title", "loop parked "+step.Key()+" after exhausted worker attempts",
		"-severity", "high",
		"-owner", step.Key(),
		"-evidence", reason,
		"-closure", "fix the step or its verifier, then rerun cmd/loop",
		"-check", "go run ./cmd/plan -verify",
	)
	out, err := command.CombinedOutput()
	if err != nil {
		return fmt.Errorf("park finding: %w: %s", err, tailOf(string(out), 500))
	}
	fmt.Printf("loop: PARKED %s -- %s\n", step.Key(), reason)
	return nil
}

func (w *execWorld) Paused() bool {
	_, err := os.Stat(pauseMarker)
	return err == nil
}

func planCommand(verb string) (string, error) {
	command := exec.Command("go", "run", "./cmd/plan", verb)
	out, err := command.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("plan %s: %w: %s", verb, err, tailOf(string(out), 500))
	}
	return string(out), nil
}

func tailOf(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value
	}
	return value[len(value)-limit:]
}
