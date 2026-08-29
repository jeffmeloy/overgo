// Package testevidence validates that successful test processes ran assertions.
package testevidence

import (
	"bufio"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const ShortIntegrationSkip = "integration excluded by -short"

type GoTestReport struct {
	PassedTests       int
	PassedPackages    int
	NoTestPackages    int
	ClassifiedSkipped []string
	Skipped           []string
	Unavailable       []string
	Failed            []string
	tests             []testResult
}

type testResult struct {
	Package     string
	Name        string
	Action      string
	Unavailable string
}

// GoTestJSONShort permits only explicitly classified short-mode exclusions.
func GoTestJSONShort(out string) error {
	report, err := GoTestJSONShortReport(out)
	if err != nil {
		return err
	}
	return RequireComplete(report)
}

// RequireComplete rejects evidence that contains unavailable or unclassified
// skipped tests. Classified short exclusions remain visible but are permitted.
func RequireComplete(report GoTestReport) error {
	if len(report.Unavailable) > 0 {
		return fmt.Errorf("%s", report.Unavailable[0])
	}
	if len(report.Skipped) > 0 {
		return fmt.Errorf("%s skipped", report.Skipped[0])
	}
	return nil
}

// GoTestJSONShortReport decodes a hermetic short-mode lane. Only skips whose
// output carries ShortIntegrationSkip are classified exclusions; they remain
// visible in the report and are never counted as passing evidence.
func GoTestJSONShortReport(out string) (GoTestReport, error) {
	return goTestJSONReport(out, true, false)
}

// GoTestJSONReport decodes complete test evidence without crediting skips.
// Callers decide whether the tested package is in the changed ownership cone.
func GoTestJSONReport(out string) (GoTestReport, error) {
	return goTestJSONReport(out, false, false)
}

func goTestJSONReport(out string, short, allowAuxiliary bool) (GoTestReport, error) {
	scanner := bufio.NewScanner(strings.NewReader(out))
	seen := false
	var report GoTestReport
	classified := map[string]bool{}
	results := map[string]*testResult{}
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event struct {
			Action  string
			Package string
			Test    string
			Output  string
		}
		if allowAuxiliary && !strings.HasPrefix(line, "{") {
			if reason := unavailable(line); reason != "" {
				return GoTestReport{}, fmt.Errorf("auxiliary verifier: %s", reason)
			}
			continue
		}
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			return GoTestReport{}, fmt.Errorf("decode go test event: %w", err)
		}
		seen = true
		key := event.Package + "\x00" + event.Test
		if event.Test != "" && results[key] == nil {
			results[key] = &testResult{Package: event.Package, Name: event.Test}
		}
		if short && strings.Contains(event.Output, ShortIntegrationSkip) {
			classified[key] = true
		}
		if reason := unavailable(event.Output); reason != "" {
			report.Unavailable = append(report.Unavailable, event.Package+": "+reason)
			if event.Test != "" {
				results[key].Unavailable = reason
			}
		}
		switch {
		case event.Action == "pass" && event.Test != "":
			report.PassedTests++
		case event.Action == "pass" && event.Test == "":
			report.PassedPackages++
		case event.Action == "skip" && event.Test != "" && classified[key]:
			report.ClassifiedSkipped = append(report.ClassifiedSkipped, event.Package+": "+event.Test)
		case event.Action == "skip" && event.Test != "":
			report.Skipped = append(report.Skipped, event.Package+": "+event.Test)
		case event.Action == "skip" && event.Test == "":
			report.NoTestPackages++
		case event.Action == "fail" && event.Test != "":
			report.Failed = append(report.Failed, event.Package+": "+event.Test)
		case event.Action == "fail" && event.Test == "":
			report.Failed = append(report.Failed, event.Package)
		}
		if event.Test != "" && (event.Action == "pass" || event.Action == "skip" || event.Action == "fail") {
			results[key].Action = event.Action
		}
	}
	if err := scanner.Err(); err != nil {
		return GoTestReport{}, fmt.Errorf("read go test events: %w", err)
	}
	if !seen {
		return GoTestReport{}, fmt.Errorf("go test emitted no events")
	}
	for _, result := range results {
		report.tests = append(report.tests, *result)
	}
	return report, nil
}

// FailureSummary names failed tests and packages from a mixed verifier stream.
// It is diagnostic only: callers still use the process exit code as authority.
func FailureSummary(out string) string {
	report, err := goTestJSONReport(out, false, true)
	if err != nil || len(report.Failed) == 0 {
		return ""
	}
	return strings.Join(report.Failed, ", ")
}

var goTestRunFlag = regexp.MustCompile(`(?:^|[ \t])-run(?:=|[ \t]+)(?:'([^']*)'|"([^"]*)"|([^ \t;&|]+))`)
var goTestShortFlag = regexp.MustCompile(`(?:^|[ \t])-short(?:=true)?(?:[ \t;&|]|$)`)

// JSONCommand enables structured events for every go test in a verifier.
func JSONCommand(command string) string {
	if !strings.Contains(command, "go test -json") {
		command = strings.ReplaceAll(command, "go test ", "go test -json ")
	}
	return command
}

// VerifyGoTestEvidence accepts a passing explicit -run oracle or a complete
// broad run that executed tests. Targeted runs do not credit unrelated skips;
// broad runs own and therefore reject every skip or unavailable fixture.
func VerifyGoTestEvidence(command, out string) error {
	report, err := goTestJSONReport(out, goTestShortFlag.MatchString(command), hasNonTestCommand(command))
	if err != nil {
		return err
	}
	if !goTestRunFlag.MatchString(command) {
		if err := RequireComplete(report); err != nil {
			return err
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 {
			return fmt.Errorf("go test verifier executed no complete passing test package")
		}
		return nil
	}
	target, err := goTestTarget(command)
	if err != nil {
		return err
	}
	passed := false
	for _, result := range report.tests {
		if !target.MatchString(result.Name) {
			continue
		}
		if result.Unavailable != "" {
			return fmt.Errorf("%s:%s: %s", result.Package, result.Name, result.Unavailable)
		}
		switch result.Action {
		case "pass":
			passed = true
		case "skip":
			return fmt.Errorf("%s:%s skipped", result.Package, result.Name)
		case "fail":
			return fmt.Errorf("%s:%s failed", result.Package, result.Name)
		}
	}
	if !passed {
		return fmt.Errorf("go test -run %q matched no passing test", target.String())
	}
	return nil
}

// hasNonTestCommand permits explicitly mixed shell verifiers to carry ordinary
// output between structured go-test event streams. JSON-looking lines remain
// strict so malformed test events cannot disappear as auxiliary output.
func hasNonTestCommand(command string) bool {
	for segment := range strings.SplitSeq(command, "&&") {
		if trimmed := strings.TrimSpace(segment); trimmed != "" && !strings.HasPrefix(trimmed, "go test ") {
			return true
		}
	}
	return false
}

// ValidateGoTestCommand checks that a declared verifier is a go test command
// naming its acceptance target -- the same admission bar plan verifiers meet.
// It validates the declaration only; execution semantics stay with the
// Verify* functions.
func ValidateGoTestCommand(command string) error {
	if !strings.Contains(command, "go test") {
		return fmt.Errorf("verifier %q is not a go test command", command)
	}
	_, err := goTestTarget(command)
	return err
}

func goTestTarget(command string) (*regexp.Regexp, error) {
	match := goTestRunFlag.FindStringSubmatch(command)
	if match == nil {
		return nil, fmt.Errorf("go test verifier must declare its acceptance target with -run")
	}
	pattern := match[1]
	if pattern == "" {
		pattern = match[2]
	}
	if pattern == "" {
		pattern = match[3]
	}
	target, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("compile go test -run target: %w", err)
	}
	return target, nil
}

// VerifyOutput rejects successful shell verification that did not prove its
// Go tests ran. Non-Go commands retain ordinary exit-code semantics.
func VerifyOutput(command, out string) error {
	if reason := unavailable(out); reason != "" {
		return fmt.Errorf("%s", reason)
	}
	if strings.Contains(out, "[no tests to run]") {
		return fmt.Errorf("go test filter matched no tests")
	}
	if strings.Contains(out, "[no test files]") {
		return fmt.Errorf("package has no tests")
	}
	if strings.Contains(out, "--- SKIP:") {
		return fmt.Errorf("go test skipped a test")
	}
	if strings.Contains(command, "go test") &&
		!strings.Contains(out, "--- PASS:") &&
		!(strings.Contains(out, "Benchmark") && strings.Contains(out, "ns/op") && strings.Contains(out, "\nPASS\n")) {
		return fmt.Errorf("go test output contains no explicit passing test")
	}
	return nil
}

func unavailable(out string) string {
	switch {
	case strings.Contains(out, "UNAVAILABLE"):
		return "output contains UNAVAILABLE"
	case strings.Contains(out, "parity NOT verified"):
		return "output reports parity NOT verified"
	default:
		return ""
	}
}
