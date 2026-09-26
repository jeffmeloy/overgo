// Package testevidence validates that successful test processes ran assertions.
package testevidence

import (
	"bufio"
	"cmp"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
)

// ShortIntegrationSkip is the skip reason under its retained name, kept
// for harness files whose bytes an acquisition record binds; a new test
// calls testskip.Short.
const ShortIntegrationSkip = testskip.ShortIntegration

type GoTestReport struct {
	// StreamDigest binds diagnostics to consumed bytes without retaining raw output.
	StreamDigest      string
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
	// Executions retain package lifecycle costs independently of evidence credit.
	Executions []PackageExecution
	tests      []testResult
	packages   map[string]packageEvidence
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
	elapsed     *float64
	costStart   string
	costEnd     string
	costInvalid bool
}

// PackageExecution reports a package process, not a test or compiler duration.
// Empty Action means interrupted; absent Elapsed means no terminal measurement.
type PackageExecution struct {
	Package string   `json:"package"`
	Action  string   `json:"action"`
	Started bool     `json:"started"`
	Elapsed *float64 `json:"elapsed_seconds,omitempty"`
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

// GoTestJSONShortReport decodes a hermetic short-mode lane. Only short-mode
// skips (testskip.Short, or its message) are classified exclusions; they remain
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
	return readGoTestJSON(reader, short, false, diagnosticBytes, observe, nil)
}

// GoTestJSONReport decodes complete test evidence without crediting skips.
// Callers decide whether the tested package is in the changed ownership cone.
func GoTestJSONReport(out string) (GoTestReport, error) {
	return goTestJSONReport(out, false, false)
}

func goTestJSONReport(out string, short, allowAuxiliary bool) (GoTestReport, error) {
	return readGoTestJSON(strings.NewReader(out), short, allowAuxiliary, 0, nil, nil)
}

type goTestEvent struct {
	Action, Package, ImportPath, Test, Output string
	Key, Value                                string
	Time                                      string
	Elapsed                                   *float64
}

func readGoTestJSON(reader io.Reader, short, allowAuxiliary bool, diagnosticBytes int, observe func(string, bool) error, eventObserved func(goTestEvent)) (report GoTestReport, err error) {
	digest := sha256.New()
	reader = io.TeeReader(reader, digest)
	scanner := bufio.NewScanner(reader)
	seen := false
	classified := map[string]bool{}
	typed := map[string]bool{}
	contended := map[string]bool{}
	results := map[string]*testResult{}
	updates := packageUpdates{observe: observe, passed: map[string]bool{}}
	tails := map[string]diagnosticTail{}
	var malformed diagnosticTail
	defer func() {
		report.StreamDigest = fmt.Sprintf("%x", digest.Sum(nil))
		if err == nil && !allowAuxiliary {
			report.packages = map[string]packageEvidence{}
		}
		if malformed.tail != "" {
			report.Diagnostics = append(report.Diagnostics, "malformed output:\n"+malformed.text())
		}
		for _, key := range slices.Sorted(maps.Keys(results)) {
			result := results[key]
			if result.Name == "" && result.Package != "" && (result.started || result.Action != "") {
				report.Executions = append(report.Executions, PackageExecution{
					Package: result.Package, Action: result.Action, Started: result.started, Elapsed: result.elapsed,
				})
			}
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
		var event goTestEvent
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
		if eventObserved != nil {
			eventObserved(event)
		}
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
		if results[key].Action != "" && event.Action != "output" {
			results[key].costInvalid = true
		}
		if event.Action == "run" || event.Action == "start" {
			results[key].costInvalid = results[key].costInvalid || results[key].started
			results[key].costStart = event.Time
			results[key].started = true
		}
		if event.Output != "" && diagnosticBytes > 0 {
			tail := tails[key]
			tail.append(event.Output, diagnosticBytes)
			tails[key] = tail
		}
		// A recorded kind decides; a test that recorded none -- a stored
		// receipt, a frozen harness -- is read by its skip message.
		typed[key] = typed[key] || event.Action == "attr" && event.Key == testskip.Key
		if typed[key] && event.Key == testskip.Key && (short && event.Value == testskip.KindShort || event.Value == testskip.KindInapplicable) ||
			!typed[key] && (short && strings.Contains(event.Output, testskip.ShortIntegration) || strings.Contains(event.Output, testskip.Inapplicable)) {
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
			results[key].costEnd = event.Time
			if event.Elapsed != nil && *event.Elapsed >= 0 {
				results[key].elapsed = event.Elapsed
			}
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

// FailureSummary names failed tests and packages from a mixed verifier stream
// and appends each failed name's retained assertion tail, so a caller shows the
// final failing lines rather than the whole output wall. It is diagnostic only:
// callers still use the process exit code as authority, and keep the raw stream
// (its digest is report.StreamDigest) as the evidence of record.
func FailureSummary(out string) string {
	report, err := readGoTestJSON(strings.NewReader(out), false, true, clioptions.DiagnosticTailBytes, nil, nil)
	if err != nil || len(report.Failed) == 0 {
		return ""
	}
	failed := make(map[string]bool, len(report.Failed))
	for _, name := range report.Failed {
		failed[name] = true
	}
	summary := []string{strings.Join(report.Failed, ", ")}
	for _, diagnostic := range report.Diagnostics {
		if name, _, ok := strings.Cut(diagnostic, ":\n"); ok && failed[name] {
			summary = append(summary, diagnostic)
		}
	}
	return strings.Join(summary, "\n")
}

// VerifyGoTestEvidence accepts a passing explicit -run oracle or a complete
// broad run that executed tests. Targeted runs do not credit unrelated skips;
// broad runs own and therefore reject every skip or unavailable fixture.
func VerifyGoTestEvidence(command, out string) error {
	parsed, err := runrecord.ParseVerify(command)
	if err != nil {
		return err
	}
	targets, short, auxiliary := parsed.Targets(), parsed.Short(), auxiliaryOutput(parsed)
	if len(targets) == 0 {
		report, err := goTestJSONReport(out, short, auxiliary)
		if err != nil {
			return err
		}
		if err := RequireComplete(report); err != nil {
			return err
		}
		if report.PassedTests == 0 || report.PassedPackages == 0 {
			return fmt.Errorf("go test verifier executed no complete passing test package")
		}
		return nil
	}
	reports, err := goTestInvocationReports(out, short, auxiliary)
	if err != nil {
		return err
	}
	if len(reports) == 0 {
		return fmt.Errorf("go test selectors require complete per-invocation evidence: targets=%d packages=%d", len(targets), len(reports))
	}
	return verifyInvocations(goTestSegments(parsed), reports)
}

func verifyTarget(target *runrecord.GoTestTarget, report GoTestReport) error {
	if len(report.Failed) != 0 {
		return fmt.Errorf("go test failed: %s", strings.Join(report.Failed, ", "))
	}
	passed := false
	for _, result := range report.tests {
		if !target.MatchString(result.Name) {
			continue
		}
		if result.Unavailable != "" {
			return fmt.Errorf("%s:%s: %s", result.Package, result.Name, result.Unavailable)
		}
		if result.ineligible || !report.packages[result.Package].passed {
			return fmt.Errorf("%s:%s has incomplete package evidence", result.Package, result.Name)
		}
		switch result.Action {
		case "pass":
			passed = true
		case "skip":
			return fmt.Errorf("%s:%s skipped", result.Package, result.Name)
		case "fail":
			return fmt.Errorf("%s:%s failed", result.Package, result.Name)
		default:
			return fmt.Errorf("%s:%s unfinished", result.Package, result.Name)
		}
	}
	if !passed {
		return fmt.Errorf("go test -run %q matched no passing test", target.Pattern)
	}
	return nil
}

// auxiliaryOutput permits explicitly mixed shell verifiers to carry ordinary
// output between structured go-test event streams: any segment other than a
// bare go test, an environment-prefixed one included, may write it. JSON-
// looking lines remain strict so malformed test events cannot disappear as
// auxiliary output.
func auxiliaryOutput(parsed runrecord.VerifyCommand) bool {
	return slices.ContainsFunc(parsed.Segments, func(segment runrecord.VerifySegment) bool {
		return segment.Kind != runrecord.SegmentGoTest || len(segment.Env) != 0
	})
}

// VerifyGoTestNames parses out once and verifies that every named test passed
// with complete package evidence. A caller checking many declared contracts
// against one command's output calls this instead of re-parsing the output per
// name.
func VerifyGoTestNames(out string, short bool, names []string) error {
	report, err := goTestJSONReport(out, short, false)
	if err != nil {
		return err
	}
	for _, name := range names {
		targets, err := runrecord.GoTestTargets("go test -run '^"+regexp.QuoteMeta(name)+"$'", true)
		if err != nil {
			return err
		}
		if len(targets) != 1 {
			return fmt.Errorf("go test names verifier: %q is not one target", name)
		}
		if err := verifyTarget(targets[0], report); err != nil {
			return err
		}
	}
	return nil
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

// goTestSegment is one go test invocation of a chained verifier: the packages
// it names and the -run selector it declares, if any.
type goTestSegment struct {
	packages []string
	target   *runrecord.GoTestTarget
}

// goTestSegments parses the go test invocations of a chained command in
// order, each with its package arguments and its selector. A segment that is
// not a go test invocation is auxiliary and yields no report. A verifier that
// chains a targeted invocation with a broad one is judged invocation by
// invocation: the selector answers for the reports its packages produced and
// the broad invocation for its own, so an unselected invocation is not read
// as the selector matching nothing.
func goTestSegments(parsed runrecord.VerifyCommand) []goTestSegment {
	var segments []goTestSegment
	for _, test := range parsed.GoTests() {
		item := goTestSegment{target: test.Target}
		for _, argument := range test.Packages {
			if relative, found := strings.CutPrefix(argument, "./"); found {
				item.packages = append(item.packages, relative)
			}
		}
		segments = append(segments, item)
	}
	return segments
}

// owns reports whether the segment's package arguments name a report's
// package: the import path ends with the argument, or the segment names a
// pattern or nothing this parser reads and so may own any package.
func (segment goTestSegment) owns(pkg string) bool {
	if len(segment.packages) == 0 {
		return true
	}
	for _, arg := range segment.packages {
		if strings.HasSuffix(arg, "/...") || pkg == arg || strings.HasSuffix(pkg, "/"+arg) {
			return true
		}
	}
	return false
}

// verifyInvocations pairs each package report with the invocation that ran it
// and judges it by that invocation's contract: a selector must match a passing
// test, a broad invocation must complete with a pass. Every invocation must
// have produced a report.
func verifyInvocations(segments []goTestSegment, reports []GoTestReport) error {
	// Reports arrive in invocation order. A report stays with the current
	// invocation until that invocation has one and this report is either a
	// package it does not name or a package it already reported, which is
	// the next invocation of the same package. Names guide the pairing but
	// never refuse it: a file run reports as command-line-arguments.
	seen := make([]bool, len(segments))
	consumed := map[string]bool{}
	index := 0
	for _, report := range reports {
		pkg := ""
		for name := range report.packages {
			pkg = name
		}
		if index+1 < len(segments) && seen[index] && (!segments[index].owns(pkg) || consumed[pkg]) {
			index++
			consumed = map[string]bool{}
		}
		seen[index] = true
		consumed[pkg] = true
		if target := segments[index].target; target != nil {
			if err := verifyTarget(target, report); err != nil {
				return err
			}
			continue
		}
		if err := RequireComplete(report); err != nil {
			return err
		}
		if report.PassedTests == 0 {
			return fmt.Errorf("go test invocation for %s executed no passing test", pkg)
		}
	}
	for position, ran := range seen {
		if !ran {
			return fmt.Errorf("go test selectors require complete per-invocation evidence: invocation %d produced no report", position+1)
		}
	}
	return nil
}
