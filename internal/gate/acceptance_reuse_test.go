package gate

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestAcceptanceReuseGrammar holds parseReusableAcceptance to the exact shape an
// owner receipt can discharge: one go test of a single package, selected by
// -run and run once, behind at most one file-existence guard. Every command
// outside that shape reports not-reusable so its verifier still runs.
func TestAcceptanceReuseGrammar(t *testing.T) {
	for _, tc := range []struct {
		name   string
		verify string
		ok     bool
		pkg    string
		short  bool
	}{
		{name: "single full", verify: "go test ./internal/gate -run '^TestX$' -count=1", ok: true, pkg: "./internal/gate"},
		{name: "single short", verify: "go test ./internal/gate -run '^TestX$' -short -count=1", ok: true, pkg: "./internal/gate", short: true},
		{name: "guarded", verify: "test -f internal/gate/x_test.go && go test ./internal/gate -run '^TestX$' -count=1", ok: true, pkg: "./internal/gate"},
		{name: "flag order", verify: "go test ./internal/gate -count=1 -run '^TestX$'", ok: true, pkg: "./internal/gate"},
		{name: "double quoted run", verify: `go test ./internal/gate -run "^TestX$" -count=1`, ok: true, pkg: "./internal/gate"},
		{name: "compound", verify: "go test ./a -run '^TestX$' -count=1 && go test ./b -run '^TestY$' -count=1"},
		{name: "env prefix", verify: "OVERGO_DATA_ROOT=x go test ./a -run '^TestX$' -count=1"},
		{name: "repeat count", verify: "go test ./a -run '^TestX$' -count=2"},
		{name: "no count", verify: "go test ./a -run '^TestX$'"},
		{name: "multi package", verify: "go test ./a ./b -run '^TestX$' -count=1"},
		{name: "no run", verify: "go test ./a -count=1"},
		{name: "extra flag", verify: "go test ./a -run '^TestX$' -count=1 -v"},
		{name: "bench flag", verify: "go test ./a -run '^TestX$' -bench=. -count=1"},
		{name: "second command not go test", verify: "test -f x && echo done"},
		{name: "two guards", verify: "test -f a && test -f b && go test ./a -run '^TestX$' -count=1"},
		{name: "pipe", verify: "go test ./a -run '^TestX$' -count=1 | cat"},
		{name: "guard with space", verify: `test -f "a b.go" && go test ./a -run '^TestX$' -count=1`},
		{name: "empty", verify: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			acceptance, ok := parseReusableAcceptance(tc.verify)
			if ok != tc.ok {
				t.Fatalf("parse reusable = %v, want %v", ok, tc.ok)
			}
			if !ok {
				return
			}
			if acceptance.packagePath != tc.pkg || acceptance.short != tc.short {
				t.Fatalf("parsed pkg=%q short=%v, want pkg=%q short=%v", acceptance.packagePath, acceptance.short, tc.pkg, tc.short)
			}
			if acceptance.target == nil || !acceptance.target.MatchString("TestX") {
				t.Fatalf("parsed target does not select its declared test")
			}
		})
	}
}

// TestAcceptanceReuseReceiptMatching holds receiptDischarges to receipts that
// actually prove the target: a matched test must have passed, a matched skip or
// an empty verdict set discharges nothing, and unmatched verdicts never credit.
func TestAcceptanceReuseReceiptMatching(t *testing.T) {
	target, err := runrecord.GoTestTargets("go test -run '^TestX$'", true)
	if err != nil || len(target) != 1 {
		t.Fatalf("compile target: %v", err)
	}
	for _, tc := range []struct {
		name  string
		tests map[string]string
		want  bool
	}{
		{name: "matched pass", tests: map[string]string{"TestX": "pass"}, want: true},
		{name: "matched skip", tests: map[string]string{"TestX": "skip"}},
		{name: "matched subtest skip", tests: map[string]string{"TestX": "pass", "TestX/slow": "skip"}},
		{name: "unmatched only", tests: map[string]string{"TestY": "pass"}},
		{name: "pass beside unmatched", tests: map[string]string{"TestX": "pass", "TestOther": "pass"}, want: true},
		{name: "empty", tests: map[string]string{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := receiptDischarges(target[0], tc.tests); got != tc.want {
				t.Fatalf("receiptDischarges = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestAcceptanceReuseSkipsProvenExecution proves the coalescing end to end: an
// owner package run recorded a passing short receipt, so a matching acceptance
// verifier is discharged from that receipt without a second execution, while
// every context mismatch keeps its own execution.
func TestAcceptanceReuseSkipsProvenExecution(t *testing.T) {
	root := t.TempDir()
	graph := packageInputGraph{
		root:  root,
		byID:  map[string][]int{"example/app": {0}},
		nodes: []goPackageInput{{ImportPath: "example/app", Dir: filepath.Join(root, "app"), Match: []string{"./..."}}},
	}
	g := &gateContext{repo: root, storePath: StorePath, environment: lifecycleTestEnvironment(t), packageGraph: &graph}
	t.Cleanup(func() { _ = g.closeStore() })
	ledger, err := g.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]artifact.ID{"example/app": testutil.ArtifactID(t, artifact.KindEvidence, "app source")}
	if err := ledger.prepare(t.Context(), []string{"example/app"}, "short", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	if err := ledger.record(t.Context(), "example/app", true, map[string]string{"TestWorks": "pass", "TestSlow": "skip"}); err != nil {
		t.Fatal(err)
	}
	g.testPlan = &testGroups{ledger: ledger, directInputs: inputs}

	const reusable = "go test ./app -run '^TestWorks$' -short -count=1"
	verdict, reused, err := g.reuseAcceptanceVerdict(t.Context(), reusable)
	if err != nil || !reused || verdict != runrecord.ClassifyVerifyCommand(reusable) {
		t.Fatalf("proven execution not reused: verdict=%q reused=%v err=%v", verdict, reused, err)
	}
	for _, tc := range []struct {
		name   string
		verify string
	}{
		{name: "matched skip runs", verify: "go test ./app -run '^TestSlow$' -short -count=1"},
		{name: "unrecorded test runs", verify: "go test ./app -run '^TestAbsent$' -short -count=1"},
		{name: "mode mismatch runs", verify: "go test ./app -run '^TestWorks$' -count=1"},
		{name: "unprepared package runs", verify: "go test ./dep -run '^TestWorks$' -short -count=1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, reused, err := g.reuseAcceptanceVerdict(t.Context(), tc.verify); err != nil || reused {
				t.Fatalf("reused=%v err=%v, want no reuse", reused, err)
			}
		})
	}
}
