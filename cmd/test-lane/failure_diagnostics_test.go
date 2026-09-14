package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

func diagnosticEvent(action, pkg, test, output string) string {
	data, err := json.Marshal(map[string]string{"Action": action, "Package": pkg, "Test": test, "Output": output})
	if err != nil {
		panic(err)
	}
	return string(data) + "\n"
}

func TestFailureDiagnostics(t *testing.T) {
	if os.Getenv("OVERGO_FAILURE_DIAGNOSTICS_CHILD") == "wait" {
		fmt.Fprint(os.Stdout, diagnosticEvent("run", "fixture", "TestBlocked", ""))
		// The observer acknowledges this package only after the parser has
		// consumed the preceding unfinished test's run event.
		fmt.Fprint(os.Stdout, diagnosticEvent("start", "ready", "", ""))
		fmt.Fprint(os.Stdout, diagnosticEvent("run", "ready", "TestReady", ""))
		fmt.Fprint(os.Stdout, diagnosticEvent("pass", "ready", "TestReady", ""))
		fmt.Fprint(os.Stdout, diagnosticEvent("pass", "ready", "", ""))
		time.Sleep(time.Minute)
		os.Exit(0)
	}
	if os.Getenv("OVERGO_FAILURE_DIAGNOSTICS_CHILD") == "1" {
		fmt.Fprint(os.Stdout, diagnosticEvent("output", "fixture", "TestBroken", "assertion: expected preserved value\n"))
		fmt.Fprint(os.Stderr, diagnosticEvent("fail", "fixture", "TestBroken", ""))
		fmt.Fprintln(os.Stdout, "{malformed")
		for range 1000 {
			fmt.Fprintln(os.Stdout, strings.Repeat("x", clioptions.DiagnosticTailBytes))
		}
		os.Exit(1)
	}
	runFixture := func(t *testing.T, input string, processErr error) (string, string, testevidence.GoTestReport) {
		t.Helper()
		var stdout, stderr strings.Builder
		report, parseErr := testevidence.GoTestJSONReader(strings.NewReader(input), true, clioptions.DiagnosticTailBytes, nil)
		code := run(nil, &stdout, &stderr, func([]string) (testevidence.GoTestReport, error) {
			return report, errors.Join(parseErr, processErr)
		})
		if code == 0 {
			t.Fatalf("failure accepted: report=%+v stdout=%s", report, stdout.String())
		}
		return stdout.String(), stderr.String(), report
	}
	t.Run("named assertion survives unrelated passing output", func(t *testing.T) {
		input := diagnosticEvent("run", "broken", "TestWrong", "") +
			diagnosticEvent("output", "broken", "TestWrong", "expected 4, got 7\n") +
			diagnosticEvent("fail", "broken", "TestWrong", "") +
			diagnosticEvent("fail", "broken", "", "")
		for range 100 {
			input += diagnosticEvent("output", "passing", "TestGood", strings.Repeat("irrelevant", 100)+"\n")
		}
		input += diagnosticEvent("pass", "passing", "TestGood", "") + diagnosticEvent("pass", "passing", "", "")
		_, output, report := runFixture(t, input, nil)
		if !strings.Contains(output, "broken: TestWrong") || !strings.Contains(output, "expected 4, got 7") || strings.Contains(output, "irrelevant") {
			t.Fatalf("diagnostic=%s", output)
		}
		if len(report.Diagnostics) != 1 {
			t.Fatalf("diagnostics=%v", report.Diagnostics)
		}
	})
	t.Run("timeout cause survives stack tail", func(t *testing.T) {
		input := diagnosticEvent("start", "hung", "", "") + diagnosticEvent("run", "hung", "TestBlocked", "") +
			diagnosticEvent("output", "hung", "", "panic: test timed out after 1s\n")
		for range 100 {
			input += diagnosticEvent("output", "hung", "", "goroutine stack "+strings.Repeat("frame ", 100)+"\n")
		}
		input += diagnosticEvent("output", "hung", "", "last stack frame\n") + diagnosticEvent("fail", "hung", "", "")
		_, output, report := runFixture(t, input, errors.New("exit status 1"))
		for _, want := range []string{"panic: test timed out after 1s", "unfinished: hung: TestBlocked", "last stack frame"} {
			if !strings.Contains(output, want) {
				t.Fatalf("missing %q in %s", want, output)
			}
		}
		if len(strings.Join(report.Diagnostics, "")) > 3*clioptions.DiagnosticTailBytes {
			t.Fatal("full stack retained")
		}
	})
	t.Run("panic reason survives test output", func(t *testing.T) {
		input := diagnosticEvent("run", "panicpkg", "TestPanic", "") +
			diagnosticEvent("output", "panicpkg", "TestPanic", "panic: invalid buffer ownership\n")
		for range 100 {
			input += diagnosticEvent("output", "panicpkg", "TestPanic", strings.Repeat("frame ", 100)+"\n")
		}
		input += diagnosticEvent("fail", "panicpkg", "TestPanic", "")
		_, output, _ := runFixture(t, input, errors.New("exit status 2"))
		if !strings.Contains(output, "panic: invalid buffer ownership") || !strings.Contains(output, "panicpkg: TestPanic") {
			t.Fatal(output)
		}
	})
	t.Run("malformed event preserves earlier verdicts", func(t *testing.T) {
		input := diagnosticEvent("output", "broken", "TestFirst", "original assertion\n") +
			diagnosticEvent("fail", "broken", "TestFirst", "") + "{bad event\n"
		_, output, report := runFixture(t, input, nil)
		for _, want := range []string{"decode go test event", "original assertion", "broken: TestFirst", "{bad event"} {
			if !strings.Contains(output, want) {
				t.Fatalf("missing %q in %s", want, output)
			}
		}
		if len(report.Failed) != 1 {
			t.Fatalf("failed evidence lost: %+v", report)
		}
	})
	t.Run("truncated stream is unfinished", func(t *testing.T) {
		_, output, _ := runFixture(t, diagnosticEvent("start", "partial", "", "")+diagnosticEvent("run", "partial", "TestStillRunning", ""), nil)
		if !strings.Contains(output, "unfinished: partial: TestStillRunning") {
			t.Fatal(output)
		}
	})
	t.Run("malformed prefix preserves later diagnostics", func(t *testing.T) {
		input := "# compiler output\n" + diagnosticEvent("output", "later", "TestAfterMalformed", "assertion after malformed event\n") +
			diagnosticEvent("fail", "later", "TestAfterMalformed", "") + "source.go:42: undefined: missingSymbol\n"
		_, output, report := runFixture(t, input, errors.New("exit status 1"))
		for _, want := range []string{"decode go test event", "assertion after malformed event", "later: TestAfterMalformed", "undefined: missingSymbol"} {
			if !strings.Contains(output, want) {
				t.Fatalf("missing %q in %s", want, output)
			}
		}
		if len(report.Failed) != 1 {
			t.Fatal("later failure verdict was discarded")
		}
	})
	t.Run("build output names its import path", func(t *testing.T) {
		input := "{\"Action\":\"build-output\",\"ImportPath\":\"broken/build\",\"Output\":\"undefined: missingBuildSymbol\\n\"}\n"
		_, output, _ := runFixture(t, input, errors.New("exit status 1"))
		if !strings.Contains(output, "broken/build") || !strings.Contains(output, "missingBuildSymbol") {
			t.Fatal(output)
		}
	})
	t.Run("advisory skips do not bury failures", func(t *testing.T) {
		input := diagnosticEvent("output", "broken", "TestFailure", "causal assertion\n") + diagnosticEvent("fail", "broken", "TestFailure", "")
		for index := range 100 {
			name := fmt.Sprintf("TestOptional%d", index)
			input += diagnosticEvent("output", "optional", name, strings.Repeat("unavailable fixture ", 100)) + diagnosticEvent("skip", "optional", name, "")
		}
		report, err := testevidence.GoTestJSONReader(strings.NewReader(input), false, clioptions.DiagnosticTailBytes, nil)
		if err != nil || len(report.Skipped) != 100 || len(report.Diagnostics) != 1 || !strings.Contains(report.Diagnostics[0], "causal assertion") {
			t.Fatalf("err=%v report=%+v", err, report)
		}
	})
	t.Run("success stays quiet and counted", func(t *testing.T) {
		input := diagnosticEvent("run", "okpkg", "TestWorks", "") +
			diagnosticEvent("output", "okpkg", "TestWorks", "diagnostic from passing test\n") +
			diagnosticEvent("pass", "okpkg", "TestWorks", "") + diagnosticEvent("pass", "okpkg", "", "")
		var output, diagnostic strings.Builder
		code := run(nil, &output, &diagnostic, func([]string) (testevidence.GoTestReport, error) {
			return testevidence.GoTestJSONReader(strings.NewReader(input), true, clioptions.DiagnosticTailBytes, nil)
		})
		if code != 0 || diagnostic.Len() != 0 || !strings.Contains(output.String(), "PASS packages=1 tests=1") {
			t.Fatalf("code=%d output=%s diagnostic=%s", code, output.String(), diagnostic.String())
		}
	})
	t.Run("malformed subprocess drains pipe", func(t *testing.T) {
		ctx, cancel := context.WithTimeoutCause(t.Context(), 5*time.Second, errors.New("diagnostic subprocess drain stalled"))
		defer cancel()
		report, err := testevidence.RunGoTestCommand(ctx, processcontrol.Command{
			Path: os.Args[0], Args: []string{"-test.run=^TestFailureDiagnostics$"},
			Env: append(os.Environ(), "OVERGO_FAILURE_DIAGNOSTICS_CHILD=1"),
		}, testevidence.GoTestOptions{Short: true, DiagnosticBytes: clioptions.DiagnosticTailBytes})
		if ctx.Err() != nil {
			t.Fatal(context.Cause(ctx))
		}
		if err == nil || len(report.Failed) != 1 || !strings.Contains(strings.Join(report.Diagnostics, ""), "expected preserved value") {
			t.Fatalf("err=%v report=%+v", err, report)
		}
	})
	t.Run("streamed successful process", func(t *testing.T) {
		report, err := runGoTest(t.Context(), []string{"../../internal/testevidence"})
		if err != nil || testevidence.RequireComplete(report) != nil || report.PassedTests == 0 || report.PassedPackages != 1 {
			t.Fatalf("err=%v report=%+v", err, report)
		}
		if len(report.Diagnostics) != 0 {
			t.Fatalf("successful transcript retained: %v", report.Diagnostics)
		}
	})
	t.Run("cancellation preserves unfinished evidence", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		report, err := testevidence.RunGoTestCommand(ctx, processcontrol.Command{
			Path: os.Args[0], Args: []string{"-test.run=^TestFailureDiagnostics$"},
			Env: append(os.Environ(), "OVERGO_FAILURE_DIAGNOSTICS_CHILD=wait"),
		}, testevidence.GoTestOptions{Short: true, DiagnosticBytes: clioptions.DiagnosticTailBytes,
			Observe: func(name string, passed bool) error {
				if name == "ready" && passed {
					cancel(context.DeadlineExceeded)
				}
				return nil
			}})
		if !errors.Is(err, context.DeadlineExceeded) || len(report.Unfinished) != 1 || report.Unfinished[0] != "fixture: TestBlocked" {
			t.Fatalf("err=%v report=%+v", err, report)
		}
	})
	t.Run("start failure returns without a blocked pipe", func(t *testing.T) {
		if _, err := testevidence.RunGoTestCommand(t.Context(), processcontrol.Command{}, testevidence.GoTestOptions{Short: true, DiagnosticBytes: clioptions.DiagnosticTailBytes}); err == nil {
			t.Fatal("absent process accepted")
		}
	})
	t.Run("read errors retain failures", func(t *testing.T) {
		reader := io.MultiReader(strings.NewReader(diagnosticEvent("fail", "broken", "TestRead", "")), diagnosticReadError{})
		report, err := testevidence.GoTestJSONReader(reader, true, clioptions.DiagnosticTailBytes, nil)
		if err == nil || len(report.Failed) != 1 {
			t.Fatalf("err=%v report=%+v", err, report)
		}
	})
}

type diagnosticReadError struct{}

func (diagnosticReadError) Read([]byte) (int, error) { return 0, errors.New("fixture read failed") }
