package gate

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"

	"overgo/internal/automationcheck"
)

// preflightChecks selects the pipeline's static checks in pipeline order: the
// declared requirements admit a check, never its phase. Descriptors are the
// gate's own, so the planned identities are the gate's.
func (g *gateContext) preflightChecks() []automationcheck.Check {
	var checks []automationcheck.Check
	for _, check := range g.pipelineChecks() {
		if check.Descriptor.Requirements.Static() {
			checks = append(checks, check)
		}
	}
	return checks
}

// preflightInvocations plans the whole pipeline exactly as the gate does and
// splits it: the static invocations run; the rest count as satisfied
// dependencies, so a static check that follows expensive work in the gate
// still runs here, while a check that needs the frozen candidate, a pipeline
// intermediate or an expensive process never does. A static check keeps the
// gate's order through a skipped check: it inherits that check's own
// dependencies, so cheap refusals still precede the heavier analysis.
func preflightInvocations(pipeline []automationcheck.Check) ([]automationcheck.Invocation, map[string]bool, error) {
	// A partial pipeline (one check diagnosed alone) drops dependencies on
	// absent checks; the whole pipeline keeps every descriptor as the gate's.
	present := map[string]bool{}
	for _, check := range pipeline {
		present[check.Descriptor.Name] = true
	}
	pipeline = slices.Clone(pipeline)
	for index := range pipeline {
		dependencies := pipeline[index].Descriptor.Dependencies
		if slices.ContainsFunc(dependencies, func(name string) bool { return !present[name] }) {
			pipeline[index].Descriptor.Dependencies = slices.DeleteFunc(slices.Clone(dependencies), func(name string) bool { return !present[name] })
		}
	}
	planned, err := automationcheck.Plan(pipeline, automationcheck.Impact{})
	if err != nil {
		return nil, nil, err
	}
	dependencies := map[string][]string{}
	static := map[string]bool{}
	for _, invocation := range planned {
		dependencies[invocation.Check.Name] = invocation.Check.Dependencies
		static[invocation.Check.Name] = invocation.Check.Requirements.Static()
	}
	satisfied := map[string]bool{}
	var selected []automationcheck.Invocation
	for _, invocation := range planned {
		if !static[invocation.Check.Name] {
			satisfied[invocation.Check.Name] = true
			continue
		}
		invocation.Check.Dependencies = staticDependencies(invocation.Check.Name, dependencies, static)
		selected = append(selected, invocation)
	}
	return selected, satisfied, nil
}

// staticDependencies resolves a check's dependencies through the skipped
// checks to the static checks behind them, in planned order.
func staticDependencies(name string, dependencies map[string][]string, static map[string]bool) []string {
	var resolved []string
	visited := map[string]bool{name: true}
	var walk func(names []string)
	walk = func(names []string) {
		for _, dependency := range names {
			if visited[dependency] {
				continue
			}
			visited[dependency] = true
			if static[dependency] {
				resolved = append(resolved, dependency)
				continue
			}
			walk(dependencies[dependency])
		}
	}
	walk(dependencies[name])
	return resolved
}

// preflightReport measures the wall to the first actionable finding and the
// whole diagnostic, the cost a clean attempt adds ahead of the gate.
type preflightReport struct {
	Findings     []string
	Skipped      int
	Unstarted    int
	Total        time.Duration
	FirstFinding time.Duration
}

// runPreflight runs the pipeline's static checks through the gate's DAG
// executor under the diagnostic policy: independent checks run together and
// every finding is reported; a failure blocks only the work behind it.
// Skipped checks earn no pass.
func runPreflight(ctx context.Context, pipeline []automationcheck.Check, output io.Writer) error {
	static, satisfied, err := preflightInvocations(pipeline)
	if err != nil {
		return fmt.Errorf("preflight: %w", err)
	}
	report := preflightReport{}
	var reportMutex sync.Mutex
	started := time.Now()
	results, err := automationcheck.ExecuteDAGIndependent(ctx, static, satisfied, func(ctx context.Context, invocation automationcheck.Invocation) (automationcheck.Evidence, error) {
		evidence, err := automationcheck.Run(ctx, invocation)
		if err != nil {
			reportMutex.Lock()
			if elapsed := time.Since(started); report.FirstFinding == 0 || elapsed < report.FirstFinding {
				report.FirstFinding = elapsed
			}
			reportMutex.Unlock()
		}
		return evidence, err
	})
	if err != nil {
		return fmt.Errorf("preflight: %w", err)
	}
	report.Total = time.Since(started).Round(time.Millisecond)
	report.FirstFinding = report.FirstFinding.Round(time.Millisecond)
	for index, result := range results {
		// A blocked invocation leaves its result empty.
		name := static[index].Check.Name
		wall := time.Duration(result.Evidence.DurationNS).Round(time.Millisecond)
		switch {
		case result.Err != nil:
			report.Findings = append(report.Findings, name)
			fmt.Fprintf(output, "preflight: %s FAIL %s: %v\n", name, wall, result.Err)
		case !result.Evidence.ID.Valid():
			report.Unstarted++
			fmt.Fprintf(output, "preflight: %s not run: a check it depends on failed\n", name)
		case result.Evidence.Inapplicable:
			report.Skipped++
			fmt.Fprintf(output, "preflight: %s SKIP %s: %s\n", name, wall, result.Evidence.Detail)
		default:
			fmt.Fprintf(output, "preflight: %s ok %s\n", name, wall)
		}
	}
	fmt.Fprintf(output, "preflight: %d finding(s) in %d check(s); %d skipped; %d not run; diagnostic only, no acceptance credit\n",
		len(report.Findings), len(static), report.Skipped, report.Unstarted)
	if len(report.Findings) != 0 {
		fmt.Fprintf(output, "preflight: first finding after %s; total %s\n", report.FirstFinding, report.Total)
		return fmt.Errorf("preflight: %d finding(s); first=%s", len(report.Findings), report.Findings[0])
	}
	fmt.Fprintf(output, "preflight: clean in %s; the gate repeats these checks against the frozen candidate\n", report.Total)
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
	return runPreflight(context.Background(), g.pipelineChecks(), output)
}
