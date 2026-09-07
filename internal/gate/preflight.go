package gate

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// preflightChecks: the validate-phase checks of the pipeline in their
// declared order; nothing that admits, prepares, tests or commits.
func (g *gateContext) preflightChecks() []automationcheck.Check {
	var checks []automationcheck.Check
	for _, check := range g.pipelineChecks() {
		if check.Descriptor.Phase == runrecord.PhaseValidate {
			checks = append(checks, check)
		}
	}
	return checks
}

// runPreflight executes every check in order without stopping, prints one
// line per check with its wall and the finding when it fails, then the
// finding count; the error names the first failed check.
func runPreflight(ctx context.Context, checks []automationcheck.Check, output io.Writer) error {
	var failed []string
	for _, check := range checks {
		started := time.Now()
		_, _, err := check.Run(ctx, automationcheck.Invocation{Check: check.Descriptor})
		wall := time.Since(started).Round(time.Millisecond)
		if err != nil {
			failed = append(failed, check.Descriptor.Name)
			fmt.Fprintf(output, "preflight: %s FAIL %s: %v\n", check.Descriptor.Name, wall, err)
			continue
		}
		fmt.Fprintf(output, "preflight: %s ok %s\n", check.Descriptor.Name, wall)
	}
	fmt.Fprintf(output, "preflight: %d finding(s) in %d check(s)\n", len(failed), len(checks))
	if len(failed) != 0 {
		return fmt.Errorf("preflight: %d finding(s); first=%s", len(failed), failed[0])
	}
	return nil
}

// Preflight runs the validate phases against the working tree: no store
// writes, no lifecycle preparation, no candidate worktree, every finding
// reported at once so a row is corrected before a gate is spent on it.
func (g *gateContext) Preflight(output io.Writer) error {
	if g == nil || len(g.paths) == 0 {
		return errors.New("preflight: -paths is required")
	}
	return runPreflight(context.Background(), g.preflightChecks(), output)
}
