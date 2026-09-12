package testevidence

import (
	"context"
	"errors"
	"io"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
)

func TestActiveTestDeadlineAcceptance(t *testing.T) {
	t.Parallel()
	t.Run("suite progress and parallel pause", func(t *testing.T) {
		start := time.Unix(0, 0)
		budget := time.Minute
		d := &testDeadline{budget: budget, idle: start.Add(budget), tests: map[string]testClock{}}
		send := func(action, name string, elapsed time.Duration) {
			t.Helper()
			now := start.Add(elapsed)
			if next, reason := d.next(); !now.Before(next) {
				t.Fatalf("premature expiry: %s at %s", reason, elapsed)
			}
			d.advance(goTestEvent{Action: action, Package: "fixture", Test: name}, now)
		}
		send("run", "TestParallel", 0)
		send("pause", "TestParallel", budget/4)
		for i := range 4 {
			send("run", "TestSerial", time.Duration(i)*budget/2+budget/4)
			send("pass", "TestSerial", time.Duration(i+1)*budget/2+budget/4)
		}
		send("cont", "TestParallel", 2*budget+budget/4)
		deadline, _ := d.next()
		if !deadline.Equal(start.Add(3 * budget)) {
			t.Fatalf("paused time consumed or reset active budget: %s", deadline.Sub(start))
		}
		send("pass", "TestParallel", 2*budget+budget/2)
		if len(d.tests) != 0 {
			t.Fatal("completed clocks retained")
		}
	})
	t.Run("noise and sibling progress cannot hide a stall", func(t *testing.T) {
		start := time.Unix(0, 0)
		budget := time.Minute
		d := &testDeadline{budget: budget, idle: start.Add(budget), tests: map[string]testClock{}}
		d.advance(goTestEvent{Action: "run", Package: "fixture", Test: "TestStalled"}, start)
		for _, event := range []goTestEvent{
			{Action: "output", Package: "fixture", Test: "TestStalled"},
			{Action: "run", Package: "fixture", Test: "TestStalled"},
			{Action: "pass", Package: "fixture", Test: "TestStalled/child"},
			{Action: "run", Package: "sibling", Test: "TestGood"},
			{Action: "pass", Package: "sibling", Test: "TestGood"},
		} {
			d.advance(event, start.Add(budget/2))
			deadline, _ := d.next()
			if !deadline.Equal(start.Add(budget)) {
				t.Fatalf("%+v renewed stalled test: %s", event, deadline.Sub(start))
			}
		}
	})
	t.Run("startup and teardown silence", func(t *testing.T) {
		start := time.Unix(0, 0)
		budget := time.Minute
		d := &testDeadline{budget: budget, idle: start.Add(budget), tests: map[string]testClock{}}
		d.advance(goTestEvent{Action: "output", Package: "fixture"}, start.Add(budget/2))
		if next, _ := d.next(); !next.Equal(start.Add(budget)) {
			t.Fatal("build chatter renewed startup")
		}
		d.advance(goTestEvent{Action: "pass", Package: "fixture"}, start.Add(budget/2))
		if next, _ := d.next(); !next.Equal(start.Add(budget + budget/2)) {
			t.Fatal("terminal process drain has no finite bound")
		}
	})
	for _, mode := range []string{"deadline", "operator cancellation"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			operator := errors.New("operator stopped test")
			options := GoTestOptions{DiagnosticBytes: 1024, ActiveTimeout: 5 * time.Second}
			if mode == "operator cancellation" {
				options.ActiveTimeout = 0
				options.Observe = func(pkg string, passed bool) error {
					if pkg == "good" && passed {
						cancel(operator)
					}
					return nil
				}
			}
			report, err := RunGoTestCommand(ctx, processcontrol.Command{
				Path: os.Args[0], Args: []string{"-test.run=^TestActiveDeadlineProcess$"},
				Env: append(os.Environ(), "OVERGO_TEST_ACTIVE_DEADLINE=1"),
			}, options)
			want := error(context.DeadlineExceeded)
			if mode == "operator cancellation" {
				want = operator
			}
			if !errors.Is(err, want) || !report.PackagePassed("good") || report.PackagePassed("blocked") || !slices.Contains(report.Unfinished, "blocked: TestBlocked") || RequireComplete(report) == nil {
				t.Fatalf("cancellation lost cause or evidence: err=%v report=%+v", err, report)
			}
		})
	}
}

func TestActiveDeadlineProcess(t *testing.T) {
	if os.Getenv("OVERGO_TEST_ACTIVE_DEADLINE") == "" {
		return
	}
	_, _ = io.WriteString(os.Stdout,
		packageEvent("start", "good", "", "")+
			packageEvent("run", "good", "TestGood", "")+
			packageEvent("pass", "good", "TestGood", "")+
			packageEvent("start", "blocked", "", "")+
			packageEvent("run", "blocked", "TestBlocked", "")+
			packageEvent("pass", "good", "", ""))
	// Continuous output exercises cancellation while the supervised pipes drain.
	for {
		_, _ = io.WriteString(os.Stdout, packageEvent("output", "blocked", "TestBlocked", strings.Repeat("x", 4096)))
		time.Sleep(time.Millisecond)
	}
}
