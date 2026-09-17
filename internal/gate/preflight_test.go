package gate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

func TestPreflightRejectsUnplannedDocsBeforeStore(t *testing.T) {
	t.Parallel()
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
	for _, check := range g.pipelineChecks() {
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
	t.Parallel()
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
	var callsMutex sync.Mutex
	fake := func(name string, err error, calls *[]string) automationcheck.Check {
		return automationcheck.Check{
			Descriptor: automationcheck.Descriptor{Name: name, Phase: runrecord.PhaseValidate, Always: true},
			Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
				callsMutex.Lock()
				*calls = append(*calls, name)
				callsMutex.Unlock()
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
	slices.Sort(calls)
	if !slices.Equal(calls, []string{"first", "fourth", "second", "third"}) {
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
	t.Parallel()
	checks := []automationcheck.Check{{
		Descriptor: automationcheck.Descriptor{Name: "fixture", Phase: runrecord.PhaseValidate, Always: true},
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
	t.Parallel()
	root, path := magicGateFixture(t, true)
	writeMagicSource(t, path, "package p\nconst ExistingLimit = 9\n")
	g := &gateContext{
		repo: root, paths: []string{"internal/p/p.go"}, storePath: "store", preflight: true,
		runCommand: func(_ string, _ string, args ...string) (string, error) {
			if !strings.Contains(strings.Join(args, " "), "-inventory-unclassified-policy") {
				t.Fatalf("preflight attempted a store repair: %v", args)
			}
			return `{"candidates":[]}`, nil
		},
	}
	if _, err := g.stepMagics(); err == nil || !strings.Contains(err.Error(), "-import-store") || !strings.Contains(err.Error(), "stale active binding") {
		t.Fatalf("stale authority was not diagnosed: %v", err)
	}
}

func TestPreflightRejectsCombinedInspection(t *testing.T) {
	t.Parallel()
	if err := Run(Options{Preflight: true, InspectPlan: true}); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("combined modes accepted: %v", err)
	}
}

// TestGateEquivalentStaticPreflight proves preflight and the gate select the
// same static checks by declared requirements, plan the same invocation
// identities, report the same seeded findings, and never start the expensive
// work behind a failed check.
func TestGateEquivalentStaticPreflight(t *testing.T) {
	t.Parallel()
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/gate/preflight.go"}}
	pipeline := g.pipelineChecks()
	var wantStatic []string
	for _, check := range pipeline {
		requirements := check.Descriptor.Requirements
		expensive := requirements.Process == automationcheck.ProcessSuite || requirements.Process == automationcheck.ProcessBrowser || requirements.Process == automationcheck.ProcessDevice
		if (check.Descriptor.Phase == runrecord.PhaseTest || check.Descriptor.Phase == runrecord.PhasePackage) && requirements.Static() {
			t.Fatalf("%s: %s phase declared static", check.Descriptor.Name, check.Descriptor.Phase)
		}
		if expensive && requirements.Static() {
			t.Fatalf("%s: %s process declared static", check.Descriptor.Name, requirements.Process)
		}
		if requirements.Static() {
			wantStatic = append(wantStatic, check.Descriptor.Name)
		}
	}
	static, satisfied, err := preflightInvocations(pipeline)
	if err != nil {
		t.Fatal(err)
	}
	planned, err := automationcheck.Plan(pipeline, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	gateIDs := map[string]artifact.ID{}
	for _, invocation := range planned {
		gateIDs[invocation.Check.Name] = invocation.ID
	}
	var names []string
	for _, invocation := range static {
		names = append(names, invocation.Check.Name)
		if invocation.ID != gateIDs[invocation.Check.Name] {
			t.Fatalf("%s: preflight identity differs from the gate plan", invocation.Check.Name)
		}
		if satisfied[invocation.Check.Name] {
			t.Fatalf("%s: static check counted as satisfied", invocation.Check.Name)
		}
	}
	if !slices.Equal(names, wantStatic) {
		t.Fatalf("preflight invocations = %v, want %v", names, wantStatic)
	}
	for _, required := range []string{"vet", "build", "fmt", "magics", "modern-go", "architecture", "docs"} {
		if !slices.Contains(names, required) {
			t.Fatalf("preflight omits %s", required)
		}
	}
	for _, excluded := range []string{modernCensusCheckName, "acceptance", testPlanCheckName, testOwnersCheckName, testDeviceCheckName, testRestCheckName, "device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName, "commit"} {
		if !satisfied[excluded] || slices.Contains(names, excluded) {
			t.Fatalf("preflight would run %s", excluded)
		}
	}

	// Seeded findings: two independent cheap failures report together; the
	// suite behind them never starts on either path.
	var mutex sync.Mutex
	started := map[string]int{}
	fake := func(name string, requirements automationcheck.Requirements, dependencies []string, err error) automationcheck.Check {
		return automationcheck.Check{
			Descriptor: automationcheck.Descriptor{Name: name, Phase: runrecord.PhaseValidate, Always: true, Dependencies: dependencies, Requirements: requirements},
			Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
				mutex.Lock()
				started[name]++
				mutex.Unlock()
				return false, "", err
			},
		}
	}
	// The census intermediate is skipped, so style inherits its scope
	// dependency; build follows style and still runs after the two failures
	// beside it, while vet waits on a failed check and the suite behind them
	// never starts.
	seeded := []automationcheck.Check{
		fake("scope", automationcheck.Requirements{}, nil, nil),
		fake("census", automationcheck.Requirements{Intermediate: true}, []string{"scope"}, nil),
		fake("baseline", automationcheck.Requirements{}, []string{"scope"}, errors.New("changed baseline: ceiling 1 exceeded by 3")),
		fake("assertion", automationcheck.Requirements{Process: automationcheck.ProcessToolchain}, []string{"scope"}, errors.New("zero tests matched -run")),
		fake("style", automationcheck.Requirements{}, []string{"census"}, nil),
		fake("vet", automationcheck.Requirements{Process: automationcheck.ProcessToolchain}, []string{"baseline", "style"}, nil),
		fake("build", automationcheck.Requirements{Process: automationcheck.ProcessToolchain}, []string{"style"}, nil),
		fake("suite", automationcheck.Requirements{Candidate: true, Process: automationcheck.ProcessSuite}, []string{"vet", "build"}, nil),
		fake("commit", automationcheck.Requirements{Candidate: true}, []string{"suite"}, nil),
	}
	preflightStatic, _, err := preflightInvocations(seeded)
	if err != nil {
		t.Fatal(err)
	}
	styleIndex := slices.IndexFunc(preflightStatic, func(invocation automationcheck.Invocation) bool { return invocation.Check.Name == "style" })
	if styleIndex < 0 || !slices.Equal(preflightStatic[styleIndex].Check.Dependencies, []string{"scope"}) {
		t.Fatalf("style dependencies through the skipped census = %v", preflightStatic)
	}
	var output bytes.Buffer
	err = runPreflight(t.Context(), seeded, &output)
	if err == nil || !strings.Contains(err.Error(), "2 finding(s)") {
		t.Fatalf("preflight = %v:\n%s", err, output.String())
	}
	for _, want := range []string{"baseline FAIL", "changed baseline: ceiling 1 exceeded by 3", "assertion FAIL", "zero tests matched -run", "build ok", "vet not run", "1 not run", "first finding after", "total "} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("preflight report lacks %q:\n%s", want, output.String())
		}
	}
	preflightStarted := map[string]int{}
	mutex.Lock()
	maps.Copy(preflightStarted, started)
	clear(started)
	mutex.Unlock()
	if preflightStarted["suite"] != 0 || preflightStarted["commit"] != 0 || preflightStarted["census"] != 0 || preflightStarted["vet"] != 0 ||
		preflightStarted["baseline"] != 1 || preflightStarted["assertion"] != 1 || preflightStarted["style"] != 1 || preflightStarted["build"] != 1 {
		t.Fatalf("preflight started %v", preflightStarted)
	}
	gatePlanned, err := automationcheck.Plan(seeded, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	results, err := automationcheck.ExecuteDAG(t.Context(), gatePlanned, nil, func(ctx context.Context, invocation automationcheck.Invocation) (automationcheck.Evidence, error) {
		return automationcheck.Run(ctx, invocation)
	})
	if err != nil {
		t.Fatal(err)
	}
	preflightStatic, _, err = preflightInvocations(seeded)
	if err != nil {
		t.Fatal(err)
	}
	for position, result := range results {
		name := gatePlanned[position].Check.Name
		executed := result.Err != nil || result.Evidence.ID.Valid()
		index := slices.IndexFunc(preflightStatic, func(invocation automationcheck.Invocation) bool { return invocation.Check.Name == name })
		if !gatePlanned[position].Check.Requirements.Static() {
			if index >= 0 || (name != "census" && (executed || started[name] != 0)) {
				t.Fatalf("gate started %s behind a failed check: %+v", name, result)
			}
			continue
		}
		if index < 0 || preflightStatic[index].ID != gatePlanned[position].ID {
			t.Fatalf("%s: gate and preflight identities differ", name)
		}
		switch {
		case executed && result.Err != nil:
			if !strings.Contains(output.String(), name+" FAIL") || !strings.Contains(output.String(), result.Err.Error()) {
				t.Fatalf("%s: gate finding %v is not the preflight finding:\n%s", name, result.Err, output.String())
			}
		case executed:
			if !strings.Contains(output.String(), name+" ok") {
				t.Fatalf("%s: gate pass is not the preflight pass:\n%s", name, output.String())
			}
		case name == "vet":
			if started[name] != 0 || preflightStarted[name] != 0 {
				t.Fatalf("%s started behind a failed check: gate %v, preflight %v", name, started, preflightStarted)
			}
		default:
			// The gate stops at the failed wave; the diagnostic still runs
			// the independent work behind it.
			if started[name] != 0 || preflightStarted[name] != 1 {
				t.Fatalf("%s: gate %v, preflight %v", name, started, preflightStarted)
			}
		}
	}
}
