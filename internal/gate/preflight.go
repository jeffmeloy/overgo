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

// preflightChecks selects validation checks in pipeline order.
func (g *gateContext) preflightChecks() []automationcheck.Check {
	var checks []automationcheck.Check
	for _, check := range g.pipelineChecks() {
		if check.Descriptor.Phase == runrecord.PhaseValidate {
			checks = append(checks, check)
		}
	}
	return checks
}

// runPreflight reports all findings; skipped checks earn no pass.
func runPreflight(ctx context.Context, checks []automationcheck.Check, output io.Writer) error {
	var failed []string
	skippedCount := 0
	for _, check := range checks {
		started := time.Now()
		skipped, detail, err := check.Run(ctx, automationcheck.Invocation{Check: check.Descriptor})
		wall := time.Since(started).Round(time.Millisecond)
		if err != nil {
			failed = append(failed, check.Descriptor.Name)
			fmt.Fprintf(output, "preflight: %s FAIL %s: %v\n", check.Descriptor.Name, wall, err)
			continue
		}
		if skipped {
			skippedCount++
			fmt.Fprintf(output, "preflight: %s SKIP %s: %s\n", check.Descriptor.Name, wall, detail)
			continue
		}
		fmt.Fprintf(output, "preflight: %s ok %s\n", check.Descriptor.Name, wall)
	}
	fmt.Fprintf(output, "preflight: %d finding(s) in %d check(s); %d skipped; diagnostic only, no acceptance credit\n", len(failed), len(checks), skippedCount)
	if len(failed) != 0 {
		return fmt.Errorf("preflight: %d finding(s); first=%s", len(failed), failed[0])
	}
	return nil
}

// Preflight diagnoses the working tree without admission or store repair.
// Findings require correction before the normal commit gate.
func (g *gateContext) Preflight(output io.Writer) error {
	if g == nil || len(g.paths) == 0 {
		return errors.New("preflight: -paths is required")
	}
	g.preflight = true
	if g.stepEvidence == nil {
		g.stepEvidence = map[string]string{}
	}
	return runPreflight(context.Background(), g.preflightChecks(), output)
}
