package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/testevidence"
)

// TestDeviceDiagnosticSurfacesEarlyFailure proves a failed package is reported
// even when a later package passes with more output than a positional tail
// holds: the device lane must not hide the first failure behind passing output.
func TestDeviceDiagnosticSurfacesEarlyFailure(t *testing.T) {
	var stream strings.Builder
	event := func(format string, args ...any) { fmt.Fprintf(&stream, format+"\n", args...) }

	const failedPackage = "overgo/internal/rxbrain"
	const failureDetail = "device output diverged from host reference"
	event(`{"Action":"run","Package":%q,"Test":"TestRxBrainKernel"}`, failedPackage)
	event(`{"Action":"output","Package":%q,"Test":"TestRxBrainKernel","Output":"--- FAIL: TestRxBrainKernel (0.10s)\n"}`, failedPackage)
	event(`{"Action":"output","Package":%q,"Test":"TestRxBrainKernel","Output":"    rxbrain_test.go:42: %s\n"}`, failedPackage, failureDetail)
	event(`{"Action":"fail","Package":%q,"Test":"TestRxBrainKernel","Elapsed":0.1}`, failedPackage)
	event(`{"Action":"fail","Package":%q,"Elapsed":0.1}`, failedPackage)

	// A later package passes with far more output than the bounded tail, so a
	// positional tail would show only this filler and drop the failure above.
	for range 400 {
		event(`{"Action":"output","Package":"overgo/internal/laterpass","Test":"TestLaterPass","Output":"    passing filler line that overflows the diagnostic tail budget\n"}`)
	}
	event(`{"Action":"pass","Package":"overgo/internal/laterpass","Test":"TestLaterPass","Elapsed":0.2}`)
	event(`{"Action":"pass","Package":"overgo/internal/laterpass","Elapsed":0.2}`)

	diagnostic := deviceDiagnostic(stream.String(), errors.New("exit code 1"))
	if !strings.Contains(diagnostic, "rxbrain") {
		t.Fatalf("diagnostic hid the failed package:\n%s", diagnostic)
	}
	if !strings.Contains(diagnostic, failureDetail) {
		t.Fatalf("diagnostic dropped the failure detail:\n%s", diagnostic)
	}
	if strings.Contains(diagnostic, "passing filler") {
		t.Fatalf("diagnostic retained passing sibling output:\n%s", diagnostic)
	}
	if len(diagnostic) > 4*clioptions.DiagnosticTailBytes {
		t.Fatalf("diagnostic is not bounded: %d bytes", len(diagnostic))
	}
}

// TestALaneFailureNamesItsFailingTests holds the device lane's refusal to its failed tests (lane-failure/named) when a
// step's output holds the go command's own lines beside its test JSON, as its stdout and stderr joined do: the
// diagnostic opens with the line the landing names the lane's failed tests by, naming the failed test and its package
// and keeping the test's words. Read strictly, one such line cost the refusal every test's name, and the landing named
// the lane's command alone.
func TestALaneFailureNamesItsFailingTests(t *testing.T) {
	var stream strings.Builder
	event := func(format string, args ...any) { fmt.Fprintf(&stream, format+"\n", args...) }
	const failedPackage = "overgo/internal/latentimage"
	event(`{"Action":"run","Package":%q,"Test":"TestDepthMapIsNear"}`, failedPackage)
	event(`{"Action":"output","Package":%q,"Test":"TestDepthMapIsNear","Output":"    depth_test.go:12: the map is far\n"}`, failedPackage)
	event(`{"Action":"fail","Package":%q,"Test":"TestDepthMapIsNear","Elapsed":0.1}`, failedPackage)
	event(`{"Action":"fail","Package":%q,"Elapsed":0.1}`, failedPackage)
	stream.WriteString("go: downloading example.com/module v1.0.0\n")

	diagnostic := deviceDiagnostic(stream.String(), errors.New("exit code 1"))
	if !strings.HasPrefix(diagnostic, testevidence.FailedTests+failedPackage+": TestDepthMapIsNear") || !strings.Contains(diagnostic, "the map is far") {
		t.Fatalf("a step's failure beside the go command's own line was rendered as:\n%s\nwant it to open by its failed test", diagnostic)
	}
}

// TestDeviceDiagnosticFallsBackToTail keeps the bounded tail for output that is
// not test JSON, such as a build or launch failure.
func TestDeviceDiagnosticFallsBackToTail(t *testing.T) {
	out := "# overgo/internal/cuda\nlink: undefined reference to launchRxBrain\n"
	diagnostic := deviceDiagnostic(out, errors.New("exit status 2"))
	if !strings.Contains(diagnostic, "undefined reference to launchRxBrain") {
		t.Fatalf("build failure was not surfaced:\n%s", diagnostic)
	}
}
