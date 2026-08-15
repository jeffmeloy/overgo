// Command test-lane runs the repository's required hermetic test lane and
// makes incomplete evidence explicit. It is the common Go owner used by CI,
// release checks, and local automation rather than duplicating shell policy.
package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"overgo/internal/testevidence"
)

type testRunner func(packages []string) (string, error)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr, runGoTest))
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
	out, runErr := runner(packages)
	report, evidenceErr := testevidence.GoTestJSONShortReport(out)
	if evidenceErr == nil {
		evidenceErr = testevidence.RequireComplete(report)
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

func runGoTest(packages []string) (string, error) {
	args := append([]string{"test", "-short", "-json", "-count=1"}, packages...)
	out, err := exec.Command("go", args...).CombinedOutput()
	return string(out), err
}
