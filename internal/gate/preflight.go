package gate

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/plan"
	"overgo/internal/planverify"
)

// Generated authority outputs reported by the existing scope inspection.
var generatedAuthorityPaths = []string{
	apiManifestFile, compatibilityManifestFile, compatibilityMatrixFile,
	"SBOM.cdx.json", "kernels/manifest.json",
	"docs/modern_go_census.json", "docs/modern_go_baseline.json",
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

// Preflight applies the derived-file repairs admission applies to the
// working tree, diagnoses it without admission or store repair, and when
// the static checks hold, prints what the gate would run for the planned
// paths and runs the dispatched row's acceptance against the working tree.
// Findings require correction before the normal commit gate.
func (g *gateContext) Preflight(output io.Writer) error {
	if g == nil || len(g.paths) == 0 {
		return errors.New("preflight: -paths is required")
	}
	g.preflight = true
	if g.stepEvidence == nil {
		g.stepEvidence = map[string]string{}
	}
	if !g.environment.ID.Valid() {
		environment, err := discoverEnvironment(g.repo)
		if err != nil {
			return err
		}
		g.environment = environment
	}
	if err := g.preflightRepairs(output); err != nil {
		return err
	}
	err := runPreflight(context.Background(), g.pipelineChecks(), output)
	if g.pendingGenerated != nil {
		fmt.Fprintf(output, "preflight: dirty-generated: ok (%d generated file(s) pending commit)\n", *g.pendingGenerated)
	}
	if err != nil {
		return err
	}
	if err := g.preflightSelection(output); err != nil {
		return err
	}
	return g.preflightAcceptance(context.Background(), output)
}

// preflightAcceptance runs the dispatched row's verify against the working
// tree, as the gate runs it against the frozen candidate, and prints the
// verdict; without a row or a verify it prints so and passes.
func (g *gateContext) preflightAcceptance(ctx context.Context, output io.Writer) error {
	if g.planRef == "" {
		fmt.Fprintln(output, "preflight: acceptance skipped: -plan names no row")
		return nil
	}
	verify, err := plannedStepVerify(g.repo, g.planRef)
	if err != nil {
		return fmt.Errorf("preflight: acceptance: %w", err)
	}
	if verify == "" {
		fmt.Fprintf(output, "preflight: acceptance skipped: %s declares no verify\n", g.planRef)
		return nil
	}
	environment, err := g.sourceEnvironment()
	if err != nil {
		return err
	}
	started := time.Now()
	verdict, err := planverify.Execute(ctx, g.repo, verify, environment)
	wall := time.Since(started).Round(time.Millisecond)
	if err != nil {
		fmt.Fprintf(output, "preflight: acceptance FAIL %s: %s: %v\n", wall, verify, err)
		return fmt.Errorf("preflight: acceptance %s: %w", g.planRef, err)
	}
	fmt.Fprintf(output, "preflight: acceptance ok %s verdict=%s: %s\n", wall, verdict, verify)
	return nil
}

// plannedStepVerify reads the verify the plan declares for one item/step.
func plannedStepVerify(repo, reference string) (string, error) {
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return "", err
	}
	itemID, stepID, _ := strings.Cut(reference, "/")
	for _, item := range document.Items {
		if item.ID != itemID {
			continue
		}
		for _, step := range item.Steps {
			if step.ID == stepID {
				return step.Verify, nil
			}
		}
	}
	return "", fmt.Errorf("%s is absent from the plan", reference)
}

// selectionGroups is what the test steps would run for the planned paths:
// the pending packages of each group, the receipts they would reuse, the
// changed owners that run first and the packages under the device lease.
type selectionGroups struct {
	owners, short, complete, devices []string
	shortReused, completeReused      int
	excluded                         int
}

// preflightSelection plans the pipeline as the gate does and prints the
// checks it selects and excludes, the lanes among them, and the package
// groups of the test steps with the receipts the retry cache would reuse.
func (g *gateContext) preflightSelection(output io.Writer) error {
	planned, err := g.planPipeline()
	if err != nil {
		return fmt.Errorf("preflight: selection: %w", err)
	}
	scope, err := g.deriveTestScope()
	if err != nil {
		return fmt.Errorf("preflight: selection: %w", err)
	}
	graph, err := g.inputGraph()
	if err != nil {
		return err
	}
	direct := slices.Concat(scope.direct, scope.uncertain)
	directInputs, err := packageInputIdentities(graph, direct)
	if err != nil {
		return err
	}
	dependentInputs, err := packageInputIdentities(graph, scope.dependent)
	if err != nil {
		return err
	}
	groups := selectionGroups{excluded: scope.excluded}
	if groups.short, groups.shortReused, err = g.packageCachePartition(direct, "short", directInputs); err != nil {
		return err
	}
	if groups.complete, groups.completeReused, err = g.packageCachePartition(scope.dependent, "complete", dependentInputs); err != nil {
		return err
	}
	for _, pkg := range groups.short {
		if slices.Contains(scope.edited, pkg) {
			groups.owners = append(groups.owners, pkg)
		}
	}
	if groups.devices, err = graph.devicePackages(slices.Concat(groups.short, groups.complete)); err != nil {
		return err
	}
	writeSelectionReport(output, buildGatePlanReport(planned), groups)
	return nil
}

// writeSelectionReport prints the selection: counts first, then each list
// on its own line.
func writeSelectionReport(output io.Writer, report gatePlanReport, groups selectionGroups) {
	var lanes, excludedLanes []string
	selected := map[string]bool{}
	for _, disposition := range report.Selected {
		selected[disposition.Name] = true
	}
	for _, lane := range deferredLaneChecks {
		if selected[lane] {
			lanes = append(lanes, lane)
		} else {
			excludedLanes = append(excludedLanes, lane)
		}
	}
	var excluded []string
	for _, disposition := range report.Excluded {
		excluded = append(excluded, disposition.Name)
	}
	fmt.Fprintf(output, "preflight: selection: checks selected=%d excluded=%d unresolved=%d; excluded=[%s]\n",
		len(report.Selected), len(report.Excluded), len(report.Unresolved), strings.Join(excluded, ","))
	fmt.Fprintf(output, "preflight: selection: lanes selected=[%s] excluded=[%s]; a selected lane runs after the commit unless a failed obligation forces it inline\n",
		strings.Join(lanes, ","), strings.Join(excludedLanes, ","))
	fmt.Fprintf(output, "preflight: selection: test scope: short group %d pending + %d reused receipts, complete group %d pending + %d reused, %d packages excluded, %d under the device lease\n",
		len(groups.short), groups.shortReused, len(groups.complete), groups.completeReused, groups.excluded, len(groups.devices))
	for _, list := range []struct {
		name     string
		packages []string
	}{{"owners first", groups.owners}, {"short pending", groups.short}, {"complete pending", groups.complete}, {"device lease", groups.devices}} {
		if len(list.packages) != 0 {
			fmt.Fprintf(output, "preflight: selection: %s=[%s]\n", list.name, strings.Join(list.packages, ","))
		}
	}
}

// preflightRepairs stages the registry against the working tree and prints
// each repair with the files it rewrote; a store repair is left to the gate.
func (g *gateContext) preflightRepairs(output io.Writer) error {
	if err := g.stageMechanicalRepairs(); err != nil {
		return err
	}
	g.auditMutex.Lock()
	defer g.auditMutex.Unlock()
	for _, line := range g.audit {
		if strings.HasPrefix(line, "staged repair: ") {
			fmt.Fprintf(output, "preflight: %s\n", line)
		}
	}
	return nil
}
