package testevidence

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"overgo/internal/processcontrol"
)

// GoTestOptions binds evidence, diagnostics and optional active-test timing.
type GoTestOptions struct {
	Short           bool
	DiagnosticBytes int
	Observe         func(string, bool) error
	// Progress receives each package's terminal event as the stream reports
	// it, in stream order, before the report is complete.
	Progress func(PackageProgress)
	// Tests receives each top-level test's start and end as the stream reports them, in stream order.
	Tests func(TestProgress)
	// ActiveTimeout bounds each top-level test's unpaused time and event silence.
	// Zero leaves timing to the command and parent context. Callers disabling
	// Go's aggregate timeout must supply this budget or an explicit parent limit.
	ActiveTimeout time.Duration
}

// PackageProgress is one package's terminal event: the action go test
// reported (pass, fail or skip) and the elapsed time it measured.
type PackageProgress struct {
	Package string
	Action  string
	Elapsed time.Duration
}

// TestProgress is a top-level test's lifecycle event: run when it starts, and the action that ended it (pass, fail
// or skip) with the elapsed time go test measured. A failed test carries the tail of its output and its subtests',
// up to GoTestOptions.DiagnosticBytes, so its failure is read as it fails, not when the whole command ends.
type TestProgress struct {
	Package, Test, Action string
	Elapsed               time.Duration
	Output                string
}

// testProgressObserver forwards every event to next and each top-level test's start and end to tests, keeping the
// newest tail bytes of each running test's output for its failure.
func testProgressObserver(tests func(TestProgress), tail int, next func(goTestEvent)) func(goTestEvent) {
	if tests == nil {
		return next
	}
	outputs := map[string][]byte{}
	return func(event goTestEvent) {
		next(event)
		top, _, _ := strings.Cut(event.Test, "/")
		key := event.Package + " " + top
		if top != "" && event.Action == "output" && tail > 0 {
			kept := append(outputs[key], event.Output...)
			outputs[key] = kept[max(len(kept)-tail, 0):]
		}
		if top == "" || top != event.Test || !slices.Contains([]string{"run", "pass", "fail", "skip"}, event.Action) {
			return
		}
		progress := TestProgress{Package: event.Package, Test: event.Test, Action: event.Action}
		if event.Elapsed != nil && *event.Elapsed > 0 {
			progress.Elapsed = time.Duration(*event.Elapsed * float64(time.Second))
		}
		if event.Action == "fail" {
			progress.Output = string(outputs[key])
		}
		if event.Action != "run" {
			delete(outputs, key)
		}
		tests(progress)
	}
}

// packageProgressObserver forwards every event to next and each package
// terminal event to progress.
func packageProgressObserver(progress func(PackageProgress), next func(goTestEvent)) func(goTestEvent) {
	if progress == nil {
		return next
	}
	return func(event goTestEvent) {
		next(event)
		if event.Test != "" || event.Action != "pass" && event.Action != "fail" && event.Action != "skip" {
			return
		}
		report := PackageProgress{Package: event.Package, Action: event.Action}
		if event.Elapsed != nil && *event.Elapsed > 0 {
			report.Elapsed = time.Duration(*event.Elapsed * float64(time.Second))
		}
		progress(report)
	}
}

// RunGoTestCommand drains a test process through the bounded evidence parser.
// The caller owns command arguments, environment, diagnostic budget and verdict.
func RunGoTestCommand(ctx context.Context, command processcontrol.Command, options GoTestOptions) (GoTestReport, error) {
	if options.DiagnosticBytes <= 0 || options.ActiveTimeout < 0 {
		return GoTestReport{}, errors.New("test diagnostics must be positive and active timeout nonnegative")
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	observe, flush := queuedPackageUpdates(options.Observe)
	deadline := newTestDeadline(options.ActiveTimeout, cancel)
	defer deadline.stop()
	reader, writer := io.Pipe()
	defer reader.Close()
	command.Stdout, command.Stderr = writer, writer
	finished := make(chan error, 1)
	go func() {
		receipt, err := processcontrol.Run(ctx, command)
		if receipt.ExitCode != 0 {
			err = errors.Join(err, fmt.Errorf("test process exited with status %d", receipt.ExitCode))
		}
		writer.Close()
		finished <- err
	}()
	report, parseErr := readGoTestJSON(reader, options.Short, false, options.DiagnosticBytes, observe, packageProgressObserver(options.Progress, testProgressObserver(options.Tests, options.DiagnosticBytes, deadline.observe)))
	if parseErr != nil {
		cancel(parseErr)
	}
	// Even an oversized or unreadable event must not strand a writer on a full pipe.
	_, drainErr := io.Copy(io.Discard, reader)
	runErr := <-finished
	deadline.stop()
	return report, errors.Join(parseErr, drainErr, runErr, context.Cause(ctx), flush())
}
