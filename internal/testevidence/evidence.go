// Package testevidence validates that successful test processes ran assertions.
package testevidence

import (
	"bufio"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/processcontrol"
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
	Unfinished        []string
	// Contended names, once each, the packages whose failures carried the
	// typed refusal of a physical resource claim: their tests claim the
	// device exclusively and cannot under a holder's lease.
	Contended   []string
	Diagnostics []string
	tests       []testResult
	packages    map[string]packageEvidence
}

// ContentionOnly reports a failed run whose every failure was a refused
// resource claim and nothing was left unfinished: the packages named in
// Contended may pass once admitted.
func (report GoTestReport) ContentionOnly() bool {
	if len(report.Failed) == 0 || len(report.Unfinished) != 0 || len(report.Contended) == 0 {
		return false
	}
	for _, failed := range report.Failed {
		failedPackage, _, _ := strings.Cut(failed, ": ")
		if !slices.Contains(report.Contended, failedPackage) {
			return false
		}
	}
	return true
}

type packageEvidence struct {
	passed, tested, incomplete bool
}

type testResult struct {
	Package     string
	Name        string
	Action      string
	Unavailable string
	started     bool
	ineligible  bool
	excluded    bool
}

// PackagePassed reports complete, non-vacuous package evidence independently
// of sibling failures, within the parsed execution profile. Declared short
// exclusions contribute no passes; all other skips prevent reuse.
func (report GoTestReport) PackagePassed(packagePath string) bool {
	result := report.packages[packagePath]
	return packagePath != "" && result.passed && result.tested && !result.incomplete
}

// GoTestJSONShort permits only explicitly classified short-mode exclusions.
func GoTestJSONShort(out string) error {
	report, err := GoTestJSONShortReport(out)
	if err != nil {
		return err
	}
	return RequireComplete(report)
}

// RequireComplete rejects failed, unfinished, unavailable or unclassified
// skipped tests. Classified short exclusions remain visible but are permitted.
func RequireComplete(report GoTestReport) error {
	if len(report.Failed) > 0 {
		return fmt.Errorf("%s failed", report.Failed[0])
	}
	if len(report.Unfinished) > 0 {
		return fmt.Errorf("%s did not finish", report.Unfinished[0])
	}
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

// GoTestJSONReader streams test evidence, retaining bounded failure
// diagnostics and completed verdicts even when a later event is malformed.
func GoTestJSONReader(reader io.Reader, short bool, diagnosticBytes int, observe func(string, bool) error) (GoTestReport, error) {
	if diagnosticBytes <= 0 {
		return GoTestReport{}, fmt.Errorf("diagnostic byte limit must be positive")
	}
	return readGoTestJSON(reader, short, false, diagnosticBytes, observe)
}

// GoTestJSONReport decodes complete test evidence without crediting skips.
// Callers decide whether the tested package is in the changed ownership cone.
func GoTestJSONReport(out string) (GoTestReport, error) {
	return goTestJSONReport(out, false, false)
}

func goTestJSONReport(out string, short, allowAuxiliary bool) (GoTestReport, error) {
	return readGoTestJSON(strings.NewReader(out), short, allowAuxiliary, 0, nil)
}

func readGoTestJSON(reader io.Reader, short, allowAuxiliary bool, diagnosticBytes int, observe func(string, bool) error) (report GoTestReport, err error) {
	scanner := bufio.NewScanner(reader)
	seen := false
	classified := map[string]bool{}
	contended := map[string]bool{}
	results := map[string]*testResult{}
	updates := packageUpdates{observe: observe, passed: map[string]bool{}}
	tails := map[string]diagnosticTail{}
	var malformed diagnosticTail
	defer func() {
		if err == nil && !allowAuxiliary {
			report.packages = map[string]packageEvidence{}
		}
		if malformed.tail != "" {
			report.Diagnostics = append(report.Diagnostics, "malformed output:\n"+malformed.text())
		}
		for _, key := range slices.Sorted(maps.Keys(results)) {
			result := results[key]
			if result.Name != "" {
				report.tests = append(report.tests, *result)
			}
			name := result.Package
			if result.Name != "" {
				name += ": " + result.Name
			}
			if result.started && result.Action == "" {
				report.Unfinished = append(report.Unfinished, name)
			}
			if report.packages != nil {
				report.packages[result.Package] = includePackageResult(report.packages[result.Package], result)
			}
			if tail, found := tails[key]; found {
				report.Diagnostics = append(report.Diagnostics, name+":\n"+tail.text())
			}
		}
	}()
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event struct {
			Action     string
			Package    string
			ImportPath string
			Test       string
			Output     string
		}
		if allowAuxiliary && !strings.HasPrefix(line, "{") {
			if reason := unavailable(line); reason != "" {
				return report, fmt.Errorf("auxiliary verifier: %s", reason)
			}
			continue
		}
		if decodeErr := json.Unmarshal([]byte(line), &event); decodeErr != nil {
			err = cmp.Or(err, fmt.Errorf("decode go test event: %w", decodeErr))
			if updateErr := updates.invalidate(); updateErr != nil {
				return report, errors.Join(err, updateErr)
			}
			if diagnosticBytes > 0 {
				malformed.append(line+"\n", diagnosticBytes)
			}
			continue
		}
		seen = true
		event.Package = cmp.Or(event.Package, event.ImportPath)
		if terminal := results[event.Package+"\x00"]; terminal != nil && terminal.Action != "" && event.Test != "" {
			terminal.ineligible = true
		}
		key := event.Package + "\x00" + event.Test
		if results[key] == nil {
			results[key] = &testResult{Package: event.Package, Name: event.Test}
		}
		if results[key].Action != "" && event.Action != "output" || event.Action == "fail" {
			results[key].ineligible = true
		}
		if event.Action == "run" || event.Action == "start" {
			results[key].started = true
		}
		if event.Output != "" && diagnosticBytes > 0 {
			tail := tails[key]
			tail.append(event.Output, diagnosticBytes)
			tails[key] = tail
		}
		if short && strings.Contains(event.Output, ShortIntegrationSkip) {
			classified[key] = true
		}
		if reason := unavailable(event.Output); reason != "" {
			report.Unavailable = append(report.Unavailable, event.Package+": "+reason)
			results[key].Unavailable = reason
		}
		if strings.Contains(event.Output, processcontrol.ErrResourceBusy.Error()) {
			contended[key] = true
		}
		switch {
		case event.Action == "pass" && event.Test != "":
			report.PassedTests++
		case event.Action == "pass" && event.Test == "":
			report.PassedPackages++
		case event.Action == "skip" && event.Test != "" && classified[key]:
			results[key].excluded = true
			report.ClassifiedSkipped = append(report.ClassifiedSkipped, event.Package+": "+event.Test)
		case event.Action == "skip" && event.Test != "":
			report.Skipped = append(report.Skipped, event.Package+": "+event.Test)
		case event.Action == "skip" && event.Test == "":
			report.NoTestPackages++
		case event.Action == "fail" && event.Test != "":
			report.Failed = append(report.Failed, event.Package+": "+event.Test)
			if contended[key] && !slices.Contains(report.Contended, event.Package) {
				report.Contended = append(report.Contended, event.Package)
			}
		case event.Action == "fail" && event.Test == "":
			report.Failed = append(report.Failed, event.Package)
		}
		if event.Action == "pass" || event.Action == "skip" || event.Action == "fail" {
			results[key].Action = event.Action
			if event.Action == "pass" || event.Action == "skip" && (!short || event.Test == "" || classified[key]) {
				delete(tails, key)
			}
		}
		if updateErr := updates.update(results, event.Package, err == nil); updateErr != nil {
			return report, errors.Join(err, updateErr)
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return report, errors.Join(err, fmt.Errorf("read go test events: %w", scanErr), updates.invalidate())
	}
	if !seen {
		return report, errors.Join(err, fmt.Errorf("go test emitted no events"))
	}
	return report, err
}

// diagnosticTail preserves a cause separately because a panic stack can push
// its first line out of the bounded output tail.
type diagnosticTail struct {
	tail  string
	cause string
}

func (tail *diagnosticTail) append(output string, limit int) {
	tail.tail = boundedDiagnostic(tail.tail+output, limit)
	if tail.cause == "" {
		for line := range strings.SplitSeq(output, "\n") {
			if strings.Contains(line, "panic:") || strings.Contains(line, "fatal error:") || strings.Contains(line, "test timed out") {
				tail.cause = boundedDiagnostic(line, limit)
				break
			}
		}
	}
}

func boundedDiagnostic(text string, limit int) string {
	return strings.Clone(text[max(0, len(text)-limit):])
}

func (tail diagnosticTail) text() string {
	if tail.cause != "" && !strings.Contains(tail.tail, tail.cause) {
		return tail.cause + "\n" + tail.tail
	}
	return tail.tail
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
