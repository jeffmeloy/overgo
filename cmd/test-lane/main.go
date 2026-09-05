// Command test-lane runs the repository's required hermetic test lane and
// makes incomplete evidence explicit. It is the common Go owner used by CI,
// release checks, and local automation rather than duplicating shell policy.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

type testRunner func(packages []string) (testevidence.GoTestReport, error)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	code := run(os.Args[1:], os.Stdout, os.Stderr, func(packages []string) (testevidence.GoTestReport, error) {
		return runGoTest(ctx, packages)
	})
	stop()
	os.Exit(code)
}

func run(args []string, stdout, stderr io.Writer, runner testRunner) int {
	packages := args
	if len(packages) == 0 {
		packages = []string{"./..."}
	}
	for _, pkg := range packages {
		if strings.TrimSpace(pkg) == "" || strings.HasPrefix(pkg, "-") {
			fmt.Fprintf(stderr, "test-lane: package arguments only; got %q\n", pkg)
			return 2
		}
	}
	report, runErr := runner(packages)
	evidenceErr := testevidence.RequireComplete(report)
	if evidenceErr == nil && report.PassedPackages+report.NoTestPackages == 0 {
		evidenceErr = errors.New("no completed test packages")
	}
	if runErr != nil || evidenceErr != nil {
		if runErr != nil {
			fmt.Fprintf(stderr, "test-lane: go test failed: %v\n", runErr)
		}
		if evidenceErr != nil {
			fmt.Fprintf(stderr, "test-lane: evidence rejected: %v\n", evidenceErr)
		}
		for _, failed := range report.Failed {
			fmt.Fprintf(stderr, "test-lane: failed: %s\n", failed)
		}
		for _, unfinished := range report.Unfinished {
			fmt.Fprintf(stderr, "test-lane: unfinished: %s\n", unfinished)
		}
		for _, diagnostic := range report.Diagnostics {
			fmt.Fprintf(stderr, "test-lane: diagnostic: %s\n", diagnostic)
		}
		for _, skipped := range report.Skipped {
			fmt.Fprintf(stderr, "test-lane: unclassified skip: %s\n", skipped)
		}
		for _, unavailable := range report.Unavailable {
			fmt.Fprintf(stderr, "test-lane: unavailable: %s\n", unavailable)
		}
		return 1
	}
	fmt.Fprintf(stdout,
		"test-lane: PASS packages=%d tests=%d classified_short_skips=%d no_test_packages=%d (skips are visible, not credited)\n",
		report.PassedPackages, report.PassedTests, len(report.ClassifiedSkipped), report.NoTestPackages,
	)
	return 0
}

func runGoTest(ctx context.Context, packages []string) (testevidence.GoTestReport, error) {
	args := append([]string{"test", "-short", "-json", "-count=1"}, packages...)
	return testevidence.RunGoTestCommand(ctx, processcontrol.Command{Path: "go", Args: args}, true, clioptions.DiagnosticTailBytes)
}
