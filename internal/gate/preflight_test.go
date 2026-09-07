package gate

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// TestPreflightReportsValidateFindings pins: preflight runs exactly the
// validate-phase checks of the pipeline in their declared order; it never
// stops at a failure, prints one line per check and the finding count, and
// returns an error naming the first failed check; a clean run returns nil;
// a context without paths is refused.
func TestPreflightReportsValidateFindings(t *testing.T) {
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/gate/preflight.go"}}
	var names []string
	for _, check := range g.preflightChecks() {
		if check.Descriptor.Phase != runrecord.PhaseValidate {
			t.Fatalf("preflight selected %s of phase %s", check.Descriptor.Name, check.Descriptor.Phase)
		}
		names = append(names, check.Descriptor.Name)
	}
	want := []string{"protection", "scope", "architecture", "profile", "fmt", "style", "manifest", "sbom", "claims", "published", "docs", "magics", "modern-go"}
	if !slices.Equal(names, want) {
		t.Fatalf("preflight checks = %v, want %v", names, want)
	}
	for _, excluded := range []string{"acceptance", "vet", "build", "test", "device", automationcheck.WebUICheckName, "commit"} {
		if slices.Contains(names, excluded) {
			t.Fatalf("preflight selected the %s phase", excluded)
		}
	}

	fake := func(name string, err error, calls *[]string) automationcheck.Check {
		return automationcheck.Check{
			Descriptor: automationcheck.Descriptor{Name: name, Phase: runrecord.PhaseValidate, Always: true},
			Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
				*calls = append(*calls, name)
				return false, "", err
			},
		}
	}
	var calls []string
	checks := []automationcheck.Check{
		fake("first", nil, &calls), fake("second", errors.New("export-doc: Thing"), &calls), fake("third", errors.New("stale binding"), &calls), fake("fourth", nil, &calls),
	}
	var output bytes.Buffer
	err := runPreflight(t.Context(), checks, &output)
	if err == nil || !strings.Contains(err.Error(), "2 finding(s); first=second") {
		t.Fatalf("preflight error = %v", err)
	}
	if !slices.Equal(calls, []string{"first", "second", "third", "fourth"}) {
		t.Fatalf("preflight stopped early: %v", calls)
	}
	report := output.String()
	for _, want := range []string{"preflight: first ok", "preflight: second FAIL", "export-doc: Thing", "preflight: third FAIL", "stale binding", "preflight: fourth ok", "preflight: 2 finding(s) in 4 check(s)"} {
		if !strings.Contains(report, want) {
			t.Fatalf("report lacks %q:\n%s", want, report)
		}
	}
	output.Reset()
	if err := runPreflight(t.Context(), []automationcheck.Check{fake("only", nil, &calls)}, &output); err != nil || !strings.Contains(output.String(), "0 finding(s) in 1 check(s)") {
		t.Fatalf("clean preflight = %v:\n%s", err, output.String())
	}
	if err := (&gateContext{repo: t.TempDir()}).Preflight(&output); err == nil {
		t.Fatal("preflight without paths was accepted")
	}
}
