package gate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

func TestPreflightRejectsUnplannedDocsBeforeStore(t *testing.T) {
	root := t.TempDir()
	screens := filepath.Join(root, "docs", "gui", "screens")
	if err := os.MkdirAll(screens, 0o755); err != nil {
		t.Fatal(err)
	}
	for index := range 9 {
		if err := os.WriteFile(filepath.Join(screens, fmt.Sprintf("view-%d.png", index)), []byte("fixture"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	g := gateContext{repo: root, storePath: "absent-store"}
	var docs []automationcheck.Check
	for _, check := range g.preflightChecks() {
		if check.Descriptor.Name == "docs" {
			docs = append(docs, check)
		}
	}
	if len(docs) != 1 {
		t.Fatalf("documentation checks = %d", len(docs))
	}
	var output bytes.Buffer
	if err := runPreflight(t.Context(), docs, &output); err == nil {
		t.Fatal("unplanned images accepted")
	}
	for index := range 9 {
		if !strings.Contains(output.String(), fmt.Sprintf("gui/screens/view-%d.png", index)) {
			t.Fatalf("missing inventory finding: %s", output.String())
		}
	}
	if _, err := os.Stat(filepath.Join(root, g.storePath)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("preflight touched absent store: %v", err)
	}
}

func TestPreflightStructureBudget(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err = snapshot.Overlay(map[string][]byte{"internal/server/budget_fixture.go": []byte("package server\nimport _ \"overgo/internal/budgetfixture\"\n")})
	if err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: root, source: &snapshot, paths: []string{"docs/plan.json"}}
	if _, err := g.stepArchitectureRatchet(); err == nil || !strings.Contains(err.Error(), "internal-imports internal/server") {
		t.Fatalf("early coupling refusal = %v", err)
	}
}

// TestPreflightReportsValidateFindings pins order, findings and path admission.
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

func TestPreflightSeparatesSkippedChecks(t *testing.T) {
	checks := []automationcheck.Check{{
		Descriptor: automationcheck.Descriptor{Name: "fixture"},
		Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
			return true, "fixture unavailable", nil
		},
	}}
	var output bytes.Buffer
	if err := runPreflight(t.Context(), checks, &output); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"fixture SKIP", "fixture unavailable", "1 skipped", "no acceptance credit"} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("missing %q: %s", want, output.String())
		}
	}
	if strings.Contains(output.String(), "fixture ok") {
		t.Fatal("skipped fixture reported as passing")
	}
	checks[0].Run = func(context.Context, automationcheck.Invocation) (bool, string, error) {
		return true, "", errors.New("fixture corrupt")
	}
	output.Reset()
	if err := runPreflight(t.Context(), checks, &output); err == nil || !strings.Contains(output.String(), "fixture FAIL") {
		t.Fatalf("skip masked failure: %v; %s", err, output.String())
	}
}

func TestPreflightNeverRepairsStore(t *testing.T) {
	root, path := magicGateFixture(t, true)
	writeMagicSource(t, path, "package p\nconst ExistingLimit = 9\n")
	g := &gateContext{
		repo: root, paths: []string{"internal/p/p.go"}, storePath: "store", preflight: true,
		runCommand: func(string, string, ...string) (string, error) {
			t.Fatal("preflight attempted a store repair")
			return "", nil
		},
	}
	if _, err := g.stepMagics(); err == nil || !strings.Contains(err.Error(), "outside preflight") || !strings.Contains(err.Error(), "stale active binding") {
		t.Fatalf("stale authority was not diagnosed: %v", err)
	}
}

func TestPreflightRejectsCombinedInspection(t *testing.T) {
	if err := Run(Options{Preflight: true, InspectPlan: true}); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("combined modes accepted: %v", err)
	}
}
