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

// GoTestJSON rejects skipped tests and unavailable oracle markers in go test
// -json output. Package rows without tests remain neutral: derived gate scope
// may include importers that intentionally own no tests.
func GoTestJSON(out string) error {
	report, err := GoTestJSONReport(out)
	if err != nil {
		return err
	}
	if len(report.Unavailable) > 0 {
		return fmt.Errorf("%s", report.Unavailable[0])
	}
	if len(report.Skipped) > 0 {
		return fmt.Errorf("%s skipped", report.Skipped[0])
	}
	return nil
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
	return goTestJSONReport(out, true)
}

// GoTestJSONReport decodes complete test evidence without crediting skips.
// Callers decide whether the tested package is in the changed ownership cone.
func GoTestJSONReport(out string) (GoTestReport, error) {
	return goTestJSONReport(out, false)
}

func goTestJSONReport(out string, short bool) (GoTestReport, error) {
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

var goTestRunFlag = regexp.MustCompile(`(?:^|[ \t])-run(?:=|[ \t]+)(?:'([^']*)'|"([^"]*)"|([^ \t;&|]+))`)

// ValidateGoTestCommand checks that a declared verifier selects a named test.
// Execution evidence remains the responsibility of VerifyGoTestTarget.
func ValidateGoTestCommand(command string) error {
	if !strings.Contains(command, "go test") {
		return fmt.Errorf("verifier is not a go test command")
	}
	_, err := goTestTarget(command)
	return err
}

// JSONCommand enables structured events for every go test in a verifier.
func JSONCommand(command string) string {
	if !strings.Contains(command, "go test -json") {
		command = strings.ReplaceAll(command, "go test ", "go test -json ")
	}
	return command
}

// VerifyGoTestTarget requires the declared -run target to pass while retaining
// unrelated skips as visible, uncredited evidence.
func VerifyGoTestTarget(command, out string) error {
	target, err := goTestTarget(command)
	if err != nil {
		return err
	}
	report, err := GoTestJSONReport(out)
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

// VerifyGoTestTargetAbsent accepts only a healthy package run in which the
// declared target did not exist. It is the bootstrap verdict for roadmap work.
func VerifyGoTestTargetAbsent(command, out string) error {
	target, err := goTestTarget(command)
	if err != nil {
		return err
	}
	report, err := GoTestJSONReport(out)
	if err != nil {
		return err
	}
	for _, result := range report.tests {
		if target.MatchString(result.Name) {
			return fmt.Errorf("go test -run %q already matched %s", target.String(), result.Name)
		}
	}
	if report.PassedPackages == 0 {
		return fmt.Errorf("go test -run %q produced no passing package", target.String())
	}
	return nil
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
