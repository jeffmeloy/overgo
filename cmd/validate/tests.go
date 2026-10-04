package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

type testRunner func(packages []string) (testevidence.GoTestReport, error)

// runTests is validate -tests: the repository's hermetic short test lane,
// the one continuous integration runs. It makes incomplete evidence
// explicit: a failed or unfinished test, or a skip no short or inapplicable
// classification owns, refuses the run, and classified skips stay visible
// without being credited.
func runTests(packages []string, output io.Writer, runner testRunner) error {
	if len(packages) == 0 {
		packages = []string{"./..."}
	}
	for _, pkg := range packages {
		if strings.TrimSpace(pkg) == "" || strings.HasPrefix(pkg, "-") {
			return fmt.Errorf("validate -tests: package arguments only; got %q", pkg)
		}
	}
	report, runErr := runner(packages)
	evidenceErr := testevidence.RequireComplete(report)
	if evidenceErr == nil && report.PassedPackages+report.NoTestPackages == 0 {
		evidenceErr = errors.New("no completed test packages")
	}
	if runErr != nil || evidenceErr != nil {
		if runErr != nil {
			fmt.Fprintf(output, "validate -tests: go test failed: %v\n", runErr)
		}
		if evidenceErr != nil {
			fmt.Fprintf(output, "validate -tests: evidence rejected: %v\n", evidenceErr)
		}
		for _, failed := range report.Failed {
			fmt.Fprintf(output, "validate -tests: failed: %s\n", failed)
		}
		for _, unfinished := range report.Unfinished {
			fmt.Fprintf(output, "validate -tests: unfinished: %s\n", unfinished)
		}
		for _, diagnostic := range report.Diagnostics {
			fmt.Fprintf(output, "validate -tests: diagnostic: %s\n", diagnostic)
		}
		for _, skipped := range report.Skipped {
			fmt.Fprintf(output, "validate -tests: unclassified skip: %s\n", skipped)
		}
		return errors.New("validate -tests: the test evidence is incomplete")
	}
	_, err := fmt.Fprintf(output,
		"validate -tests: PASS packages=%d tests=%d classified_short_skips=%d no_test_packages=%d (skips are visible, not credited)\n",
		report.PassedPackages, report.PassedTests, len(report.ClassifiedSkipped), report.NoTestPackages,
	)
	return err
}

func runGoTest(ctx context.Context, packages []string) (testevidence.GoTestReport, error) {
	args := append([]string{"test", "-short", "-json", "-count=1"}, packages...)
	return testevidence.RunGoTestCommand(ctx, processcontrol.Command{Path: "go", Args: args}, testevidence.GoTestOptions{Short: true, DiagnosticBytes: clioptions.DiagnosticTailBytes})
}
