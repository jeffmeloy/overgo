//go:build windows

package gate

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// TestDeviceBatchRerunsRefusedExclusiveClaimsOutsideTheLease pins the device
// batch: a batch whose only failures were refused exclusive claims runs each
// refused package again alone and outside the lease until it admits, a
// batch that fails for any other reason returns that failure at once, and a
// batch that passes runs once.
func TestDeviceBatchRerunsRefusedExclusiveClaimsOutsideTheLease(t *testing.T) {
	t.Parallel()
	g := &gateContext{repo: t.TempDir(), deviceResource: "overgo-test-device-" + t.Name()}
	type call struct {
		packages []string
		leased   bool
	}
	var calls []call
	contended := testevidence.GoTestReport{Failed: []string{"overgo/internal/routedlm: TestTrainerDescends"}, Contended: []string{"overgo/internal/routedlm"}}
	refusals := 2
	run := func(_ context.Context, packages []string, _ bool, _ func(string, bool, map[string]string) error, leased bool) (testevidence.GoTestReport, error) {
		calls = append(calls, call{packages: packages, leased: leased})
		if leased || refusals > 0 {
			if !leased {
				refusals--
				// The foreign holder releases the device as it refuses; its
				// release is the signal the rerun waits on.
				hold, err := processcontrol.ShareResource(g.deviceResource)
				if err != nil {
					t.Error(err)
				} else if err := hold(); err != nil {
					t.Error(err)
				}
			}
			return contended, errors.New("go test evidence: refused")
		}
		return testevidence.GoTestReport{PassedTests: 1}, nil
	}
	batch := []string{"overgo/internal/devicemath", "overgo/internal/routedlm"}
	report, err := g.runDeviceBatch(t.Context(), batch, false, nil, run)
	if err != nil || report.PassedTests != 1 {
		t.Fatalf("contended batch = %+v, %v", report, err)
	}
	if len(calls) != 4 || !slices.Equal(calls[0].packages, batch) || !calls[0].leased {
		t.Fatalf("calls = %+v, want the leased batch then three unleased reruns of the refused package", calls)
	}
	for _, rerun := range calls[1:] {
		if rerun.leased || !slices.Equal(rerun.packages, []string{"overgo/internal/routedlm"}) {
			t.Fatalf("rerun = %+v, want the refused package alone outside the lease", rerun)
		}
	}
	if !slices.ContainsFunc(g.audit, func(line string) bool { return strings.HasPrefix(line, "device contention: 1 package(s) refused") }) ||
		!slices.ContainsFunc(g.audit, func(line string) bool { return strings.Contains(line, "admitted alone after 3 attempt(s)") }) {
		t.Fatalf("audit = %q", g.audit)
	}

	other := errors.New("go test evidence: a real failure")
	calls = nil
	_, err = g.runDeviceBatch(t.Context(), batch, false, nil, func(context.Context, []string, bool, func(string, bool, map[string]string) error, bool) (testevidence.GoTestReport, error) {
		calls = append(calls, call{})
		return testevidence.GoTestReport{Failed: []string{"overgo/internal/devicemath: TestKernel"}}, other
	})
	if !errors.Is(err, other) || len(calls) != 1 {
		t.Fatalf("real failure = %v after %d call(s), want it returned at once", err, len(calls))
	}

	// A refusal that never lifts ends with the caller's cause.
	ended := errors.New("the caller ended the wait")
	wait, cancel := context.WithCancelCause(t.Context())
	cancel(ended)
	_, err = g.runDeviceBatch(wait, batch, false, nil, func(context.Context, []string, bool, func(string, bool, map[string]string) error, bool) (testevidence.GoTestReport, error) {
		return contended, errors.New("go test evidence: refused")
	})
	if !errors.Is(err, processcontrol.ErrResourceBusy) || !errors.Is(err, ended) {
		t.Fatalf("unlifted refusal = %v, want the contention and the caller's cause", err)
	}
}
