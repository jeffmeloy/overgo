package main

import (
	"context"
	"errors"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// fakeHarness scripts each strategy's world so the comparison contract
// tests hermetically: worktrees are labels, verify outcomes and walls
// are declared per strategy.
type fakeHarness struct {
	verify map[string]bool
	wall   map[string]time.Duration
	launch map[string]error
}

func (f *fakeHarness) CreateWorktree(strategy, baseline string) (string, error) {
	return "tmp/experiments/" + strategy, nil
}
func (f *fakeHarness) Prompt(string) (string, error) { return "do the step", nil }
func (f *fakeHarness) RunWorker(worktree string, strategy Strategy, prompt string, timeout time.Duration) error {
	name := strategy.Name
	if delay, declared := f.wall[name]; declared {
		time.Sleep(delay)
	}
	return f.launch[name]
}
func (f *fakeHarness) Verify(worktree string) (bool, string) {
	for name, passed := range f.verify {
		if worktree == "tmp/experiments/"+name {
			return passed, "verify output"
		}
	}
	return false, "unknown"
}
func (f *fakeHarness) DiffFiles(string) int { return 2 }

// TestExperimentSelectsByVerifyThenCost pins the comparison contract:
// every strategy runs from the same baseline, failed trials persist as
// counterexamples, and the winner is a verify-pass with the lowest
// measured wall -- never a fast failure.
func TestExperimentSelectsByVerifyThenCost(t *testing.T) {
	world := &fakeHarness{
		verify: map[string]bool{"fast-fail": false, "slow-pass": true, "quick-pass": true},
		wall: map[string]time.Duration{
			"fast-fail": 0, "slow-pass": 30 * time.Millisecond, "quick-pass": 5 * time.Millisecond,
		},
		launch: map[string]error{},
	}
	report, err := Run(world, Spec{
		Step: "alpha/do", Baseline: "abc123",
		Strategies: []Strategy{
			{Name: "fast-fail", Worker: []string{"w1"}},
			{Name: "slow-pass", Worker: []string{"w2"}},
			{Name: "quick-pass", Worker: []string{"w3"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Trials) != 3 || report.Winner != "quick-pass" {
		t.Fatalf("report = winner %q over %d trials", report.Winner, len(report.Trials))
	}
	if report.Trials[0].Strategy != "fast-fail" || report.Trials[0].VerifyPassed {
		t.Fatalf("counterexample trial = %+v", report.Trials[0])
	}
}

// TestExperimentRecordsLaunchFailures pins the counterexample rule: a
// strategy whose worker cannot run stays in the report with its error,
// and an experiment with no verified trial selects no winner.
func TestExperimentRecordsLaunchFailures(t *testing.T) {
	world := &fakeHarness{
		verify: map[string]bool{},
		wall:   map[string]time.Duration{},
		launch: map[string]error{"broken": errors.New("worker exploded")},
	}
	report, err := Run(world, Spec{
		Step: "alpha/do", Baseline: "abc123",
		Strategies: []Strategy{{Name: "broken", Worker: []string{"w"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Winner != "" || report.Trials[0].LaunchError != "worker exploded" {
		t.Fatalf("launch-failure report = %+v", report)
	}
}

// TestExperimentRefusesDegenerateSpecs pins the boundary: a spec
// without a step, baseline, or strategies -- or with a duplicated
// strategy identity -- is refused before any worktree exists.
func TestExperimentRefusesDegenerateSpecs(t *testing.T) {
	world := &fakeHarness{verify: map[string]bool{}, wall: map[string]time.Duration{}, launch: map[string]error{}}
	if _, err := Run(world, Spec{Step: "alpha/do", Baseline: "abc"}); err == nil {
		t.Fatal("empty strategy set was accepted")
	}
	if _, err := Run(world, Spec{
		Step: "alpha/do", Baseline: "abc",
		Strategies: []Strategy{{Name: "same", Worker: []string{"w"}}, {Name: "same", Worker: []string{"v"}}},
	}); err == nil {
		t.Fatal("duplicated strategy identity was accepted")
	}
}

// TestExperimentReportPersists pins the durable record: the report
// commits to the store as typed evidence and reads back byte-exact.
func TestExperimentReportPersists(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	report := Report{
		Step: "alpha/do", Baseline: "abc123", Winner: "quick-pass",
		Trials: []Trial{{Strategy: "quick-pass", VerifyPassed: true, WallNS: 5}},
	}
	id, err := publishReport(context.Background(), store, report)
	if err != nil || id.Kind() != artifact.KindEvidence {
		t.Fatalf("publish = (%s, %v)", id, err)
	}
	if _, err := publishReport(context.Background(), store, report); err != nil {
		t.Fatalf("re-publish of the identical report = %v", err)
	}
}
