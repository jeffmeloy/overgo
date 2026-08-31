// Command preflight runs every cheap read-only repository validation
// before expensive tests or a gate run: nested competing plans, the
// generated authority documents' check modes (API manifest,
// compatibility hashes, SBOM with its kernel identities), production
// formatting, and generated files left dirty in the worktree. It writes
// and mutates nothing, so stale generated state surfaces in seconds
// instead of after a full gate run.
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/closurescan"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/repoanalysis"
)

// generatedFiles enumerates the generated authority outputs whose
// uncommitted worktree changes are worth naming before a gate run; a
// dirty entry is not a failure -- it ships with the next commit -- but
// an unexpected one means a regeneration ran without an owner.
var generatedFiles = []string{
	"docs/api_manifest.json", "docs/API_MANIFEST.md",
	"compatibility.json", "docs/COMPATIBILITY.md",
	"SBOM.cdx.json", "kernels/manifest.json",
	"docs/modern_go_census.json", "docs/modern_go_baseline.json",
}

func main() {
	clioptions.MainNamed("preflight", run)
}

func run() error {
	flag.Parse()
	checks := []struct {
		name string
		run  func() (string, error)
	}{
		{"nested-plans", checkNestedPlans},
		{"api-manifest", repoCheck("./cmd/api-manifest")},
		{"compatibility", repoCheck("./cmd/compatibility")},
		{"sbom", repoCheck("./cmd/sbom")},
		{"formatting", checkFormatting},
		{"structure-budgets", checkStructureBudgets},
		{"dirty-generated", checkDirtyGenerated},
	}
	failures := 0
	for _, check := range checks {
		detail, err := check.run()
		if err != nil {
			failures++
			fmt.Printf("preflight: %s: FAILED: %v\n", check.name, err)
			continue
		}
		fmt.Printf("preflight: %s: ok%s\n", check.name, detail)
	}
	if failures > 0 {
		return fmt.Errorf("%d check(s) failed", failures)
	}
	return nil
}

func checkNestedPlans() (string, error) {
	competing, err := plan.CompetingPlans(".")
	if err != nil {
		return "", err
	}
	if len(competing) > 0 {
		return "", fmt.Errorf("competing live plan files: %s", strings.Join(competing, ", "))
	}
	return "", nil
}

// repoCheck runs one repository command's own -check mode, so preflight
// asserts exactly what the gate's corresponding phase asserts.
func repoCheck(commandPath string) func() (string, error) {
	return func() (string, error) {
		output, err := runCommand("go", "run", commandPath, "-check")
		if err != nil {
			return "", fmt.Errorf("%s -check: %w: %s", commandPath, err, output)
		}
		return "", nil
	}
}

func checkFormatting() (string, error) {
	output, err := runCommand("gofmt", "-l", "cmd", "internal")
	if err != nil {
		return "", fmt.Errorf("gofmt -l: %w: %s", err, output)
	}
	if unformatted := strings.TrimSpace(output); unformatted != "" {
		return "", fmt.Errorf("unformatted files: %s", strings.Join(strings.Fields(unformatted), ", "))
	}
	return "", nil
}

// checkStructureBudgets holds the tree to its reviewed size and coupling
// ceilings and the agent-harness surface to its stored baseline: file
// sizes, per-package internal imports, command-package spread, the
// harness consolidation audit, and harness-surface regressions all refuse
// silent growth here, before any gate run.
func checkStructureBudgets() (string, error) {
	budgets, err := repoanalysis.LoadStructureBudgets(repoanalysis.StructureBudgetsFile)
	if err != nil {
		return "", err
	}
	snapshot, err := repoanalysis.DiscoverGo(".", "internal", "cmd")
	if err != nil {
		return "", err
	}
	findings := repoanalysis.MeasureStructureBudgets(snapshot, budgets)
	if len(findings) > 0 {
		first := findings[0]
		return "", fmt.Errorf("%d budget exceedance(s); first: %s %s = %d over limit %d",
			len(findings), first.Budget, first.Subject, first.Value, first.Limit)
	}
	consolidation, err := repoanalysis.AuditAgentHarnessConsolidation(snapshot)
	if err != nil {
		return "", err
	}
	if len(consolidation.Findings) > 0 {
		first := consolidation.Findings[0]
		return "", fmt.Errorf("harness consolidation: %d finding(s); first: %s %s expected %s got %s",
			len(consolidation.Findings), first.Kind, first.Symbol, first.Expected, first.Actual)
	}
	var baseline closurescan.HarnessSurface
	if err := jsonfile.DecodeStrict("docs/harness_surface_baseline.json", &baseline); err != nil {
		return "", err
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		return "", err
	}
	if regressions := closurescan.AgentHarnessSurfaceRegressions(baseline, surface); len(regressions) > 0 {
		first := regressions[0]
		return "", fmt.Errorf("harness surface: %d regression(s); first: %s %d -> %d %s",
			len(regressions), first.Metric, first.Base, first.Value, first.Detail)
	}
	return "", nil
}

func checkDirtyGenerated() (string, error) {
	arguments := append([]string{"status", "--porcelain", "--"}, generatedFiles...)
	output, err := runCommand("git", arguments...)
	if err != nil {
		return "", fmt.Errorf("git status: %w: %s", err, output)
	}
	dirty := strings.TrimSpace(output)
	if dirty == "" {
		return "", nil
	}
	lines := strings.Split(dirty, "\n")
	return fmt.Sprintf(" (%d generated file(s) pending commit)", len(lines)), nil
}

func runCommand(name string, arguments ...string) (string, error) {
	var output bytes.Buffer
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path: name, Args: arguments, Stdout: &output, Stderr: &output,
	})
	if err != nil {
		return output.String(), err
	}
	if receipt.ExitCode != 0 {
		return output.String(), fmt.Errorf("exit status %d", receipt.ExitCode)
	}
	return output.String(), nil
}
