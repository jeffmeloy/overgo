// Package testevidence validates that successful test processes ran assertions.
package testevidence

import (
	"bufio"
	"encoding/json"
	"fmt"
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
		if short && strings.Contains(event.Output, ShortIntegrationSkip) {
			classified[key] = true
		}
		if reason := unavailable(event.Output); reason != "" {
			report.Unavailable = append(report.Unavailable, event.Package+": "+reason)
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
	}
	if err := scanner.Err(); err != nil {
		return GoTestReport{}, fmt.Errorf("read go test events: %w", err)
	}
	if !seen {
		return GoTestReport{}, fmt.Errorf("go test emitted no events")
	}
	return report, nil
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
