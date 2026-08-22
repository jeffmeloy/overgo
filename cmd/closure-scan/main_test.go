package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
	"overgo/internal/repodb"
)

func TestConfigurationClosureGates(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	checks := []struct {
		scope        string
		requirements closureRequirements
	}{
		{
			"cmd/train,cmd/generate,internal/clioptions,internal/optimizer,internal/trainingprogram,internal/trainingworkflow,internal/modelrecipe",
			closureRequirements{classified: true, noStale: true, noUncatalogued: true},
		},
		{
			"cmd/server,internal/inference,internal/server",
			closureRequirements{classified: true, noStale: true, noUncatalogued: true, noModelFacts: true},
		},
	}
	for _, check := range checks {
		if err := checkProductionClosures(root, "repodb-store", check.scope, false, check.requirements); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCensusReportIsDeterministicAndComplete(t *testing.T) {
	snapshot := censusSnapshot(t)
	first, err := closurescan.BuildCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	second, err := closurescan.BuildCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var firstJSON, secondJSON, firstText, secondText bytes.Buffer
	if err := clioptions.WritePrettyJSON(&firstJSON, first); err != nil {
		t.Fatal(err)
	}
	if err := clioptions.WritePrettyJSON(&secondJSON, second); err != nil {
		t.Fatal(err)
	}
	if err := writeCensusText(&firstText, first, len(first.Owners)); err != nil {
		t.Fatal(err)
	}
	if err := writeCensusText(&secondText, second, len(second.Owners)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstJSON.Bytes(), secondJSON.Bytes()) || !bytes.Equal(firstText.Bytes(), secondText.Bytes()) {
		t.Fatal("census rendering is not deterministic")
	}
	if first.Schema != closurescan.CensusSchema || first.Source != snapshot.Identity() || len(first.Owners) != 1 || len(first.Files) != 2 || len(first.Repeated) != 1 {
		t.Fatalf("census = %+v", first)
	}
	if first.Files[0].File != "internal/policy.go" || first.Files[0].Test || first.Files[1].File != "internal/policy_test.go" || !first.Files[1].Test {
		t.Fatalf("file queue = %+v", first.Files)
	}
}

func TestCensusCountsReconcile(t *testing.T) {
	report, err := closurescan.BuildCensus(censusSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	var named, inline, assumptions, policy, groups, sites int
	for _, owner := range report.Owners {
		named += owner.NamedConstants
		inline += owner.InlineLiterals
		assumptions += owner.AssumptionHints
		policy += owner.TestPolicyCopies
		groups += owner.RepeatedGroups
		sites += owner.RepeatedSites
	}
	want := report.Counts
	if named != want.NamedConstants || inline != want.InlineLiterals || assumptions != want.AssumptionHints ||
		policy != want.TestPolicyCopies || groups != want.RepeatedGroups || sites != want.RepeatedSites ||
		want.TestLiterals != want.TestFixtures+want.TestAssertions+want.TestPolicyCopies {
		t.Fatalf("owner totals=(%d,%d,%d,%d,%d,%d), counts=%+v", named, inline, assumptions, policy, groups, sites, want)
	}
}

func TestCensusFileQueueReconcilesEverySource(t *testing.T) {
	report, err := closurescan.BuildCensus(censusSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	var production, tests, named, inline, assumptions, testLiterals, fixtures, assertions, policy int
	seen := map[string]bool{}
	for _, file := range report.Files {
		if seen[file.File] || file.DecisionSurfaces != file.NamedConstants+file.InlineLiterals+file.AssumptionHints+file.TestPolicyCopies {
			t.Fatalf("invalid file row: %+v", file)
		}
		seen[file.File] = true
		if file.Test {
			tests++
		} else {
			production++
		}
		named += file.NamedConstants
		inline += file.InlineLiterals
		assumptions += file.AssumptionHints
		testLiterals += file.TestLiterals
		fixtures += file.TestFixtures
		assertions += file.TestAssertions
		policy += file.TestPolicyCopies
	}
	want := report.Counts
	if production != want.ProductionFiles || tests != want.TestFiles || named != want.NamedConstants || inline != want.InlineLiterals ||
		assumptions != want.AssumptionHints || testLiterals != want.TestLiterals || fixtures != want.TestFixtures ||
		assertions != want.TestAssertions || policy != want.TestPolicyCopies {
		t.Fatalf("file totals=(%d,%d,%d,%d,%d,%d,%d,%d,%d), counts=%+v", production, tests, named, inline, assumptions, testLiterals, fixtures, assertions, policy, want)
	}
}

func TestCensusEvidenceRoundTrip(t *testing.T) {
	store, snapshot, _ := censusEvidenceFixture(t)
	report, err := closurescan.BuildCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := publishCensusEvidence(context.Background(), store, snapshot, report)
	if err != nil {
		t.Fatal(err)
	}
	loaded, found, err := closurescan.ReadCensusEvidence(context.Background(), store, evidence.ID)
	if err != nil || !found || loaded.ID != evidence.ID || loaded.Counts != report.Counts {
		t.Fatalf("loaded = (%+v, %t, %v)", loaded, found, err)
	}
}

func TestCensusEvidenceBindsCatalogAndSourceFingerprint(t *testing.T) {
	store, snapshot, head := censusEvidenceFixture(t)
	report, err := closurescan.BuildCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := publishCensusEvidence(context.Background(), store, snapshot, report)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Source != snapshot.Identity() || evidence.CatalogHead != head.String() || evidence.CatalogSequence == 0 {
		t.Fatalf("evidence = %+v", evidence)
	}
}

func censusEvidenceFixture(t *testing.T) (*repodb.Store, repoanalysis.SourceSnapshot, artifact.CommitID) {
	t.Helper()
	store, err := repodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	seed, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("catalog seed"))
	if err != nil {
		t.Fatal(err)
	}
	head, err := store.Commit(context.Background(), artifact.Batch{
		Key: "catalog/seed", Artifacts: []artifact.Descriptor{{ID: seed}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return store, censusSnapshot(t), head
}

func censusSnapshot(t *testing.T) repoanalysis.SourceSnapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"internal/policy.go": `package policy
const Limit = 3
func first(n int) bool { return n > 7 }
func second(n int) bool { return n < 7 }
func shaped(rows int) bool { return rows > Limit }
`,
		"internal/policy_test.go": `package policy
func test() { if first(3) != true { panic("fixture") } }
`,
	}
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func TestTriagePublishesAndRetiresExactBindings(t *testing.T) {
	root := t.TempDir()
	relative := "internal/sample/policy.go"
	source := []byte("package sample\n\nconst PolicyWindow = 3 * 8\n")
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, source, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	var candidate closurescan.Candidate
	for _, scanned := range candidates {
		if scanned.Name == "PolicyWindow" {
			candidate = scanned
			break
		}
	}
	if candidate.Name == "" {
		t.Fatal("fixture constant not scanned")
	}
	triage := triageFile{Rows: []triageRow{{
		Kind: candidate.Kind, Name: candidate.Name, File: candidate.File, Scope: candidate.Scope, Line: candidate.Line,
		Tier: string(closureledger.TierImplementation), Status: string(closureledger.StatusClosed),
		Understanding: "Fixed fixture policy.", ClosurePath: "Replace when the fixture contract changes.",
		RerankTrigger: "Fixture contract change.",
	}}}
	encoded, err := json.Marshal(triage)
	if err != nil {
		t.Fatal(err)
	}
	triagePath := filepath.Join(root, "triage.json")
	if err := os.WriteFile(triagePath, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := emit(root, "store", triagePath, candidates); err != nil {
		t.Fatal(err)
	}
	store, err := repodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	owner, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := candidate.Binding()
	if err != nil || binding.Owner != owner {
		t.Fatalf("binding owner = (%s, %v), want %s", binding.Owner, err, owner)
	}
	document, active, err := closureledger.ResolveActiveBinding(
		context.Background(), store, binding, candidate.ValueJSON(),
	)
	if err != nil || !active || len(document.Bindings) != 1 || document.Bindings[0] != binding {
		t.Fatalf("active document = (%+v, %t, %v)", document, active, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	moved := append([]byte("// moved declaration\n"), source...)
	if err := os.WriteFile(path, moved, 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if count, err := rebindUnchangedClosures(root, "store", snapshot); err != nil || count != len(triage.Rows) {
		t.Fatalf("rebound documents = (%d, %v)", count, err)
	}
	candidates, err = closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	previous := binding
	for _, current := range candidates {
		if current.Name == candidate.Name {
			binding, err = current.Binding()
			candidate = current
			break
		}
	}
	if err != nil || binding == previous {
		t.Fatalf("rebound binding = (%+v, %v)", binding, err)
	}
	store, err = repodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, active, err := closureledger.ResolveActiveBinding(context.Background(), store, binding, candidate.ValueJSON()); err != nil || !active {
		t.Fatalf("rebound document = (%t, %v)", active, err)
	}
	if _, found, err := artifact.ResolveAlias(context.Background(), store, mustActiveAlias(t, previous)); err != nil || found {
		t.Fatalf("previous binding resolves = (%t, %v)", found, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if retained, err := retireOrphanAliases(root, "store", snapshot); err != nil || retained != nil {
		t.Fatalf("retired live bindings = (%d, %v)", len(retained), err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	snapshot, err = repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if retired, err := retireOrphanAliases(root, "store", snapshot); err != nil || len(retired) != 1 {
		t.Fatalf("retired bindings = (%d, %v)", len(retired), err)
	}
	store, err = repodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := artifact.ResolveAlias(context.Background(), store, mustActiveAlias(t, binding)); err != nil || found {
		t.Fatalf("retired binding resolves = (%t, %v)", found, err)
	}
}

func TestScopedClosureCheckRequiresExactActiveEvidence(t *testing.T) {
	root := t.TempDir()
	relative := "internal/policy/policy.go"
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(value string) {
		t.Helper()
		if err := os.WriteFile(path, []byte("package policy\nconst Window = "+value+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("8")
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	requirements := closureRequirements{classified: true, noStale: true}
	if err := checkProductionClosures(root, "store", "internal/policy", false, requirements); err == nil {
		t.Fatal("unclassified constant passed scoped closure check")
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = (%+v, %v)", candidates, err)
	}
	triage := triageFile{Rows: []triageRow{{
		Kind: candidates[0].Kind, Name: candidates[0].Name, File: candidates[0].File, Scope: candidates[0].Scope, Line: candidates[0].Line,
		Tier: string(closureledger.TierImplementation), Status: string(closureledger.StatusClosed),
		Understanding: "Fixture window.", ClosurePath: "Replace with fixture authority.", RerankTrigger: "Fixture contract change.",
	}}}
	encoded, err := json.Marshal(triage)
	if err != nil {
		t.Fatal(err)
	}
	triagePath := filepath.Join(root, "triage.json")
	if err := os.WriteFile(triagePath, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := emit(root, "store", triagePath, candidates); err != nil {
		t.Fatal(err)
	}
	if err := checkProductionClosures(root, "store", "internal/policy", false, requirements); err != nil {
		t.Fatal(err)
	}
	write("9")
	if err := checkProductionClosures(root, "store", "internal/policy", false, requirements); err == nil {
		t.Fatal("stale constant passed scoped closure check")
	}
}

func TestAuthorityEnforcement(t *testing.T) {
	root := t.TempDir()
	relative := "internal/model/policy.go"
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package model\nconst PolicyLimit = 7\nfunc allowed(n int) bool { return n < PolicyLimit }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	requirements := closureRequirements{classified: true, noStale: true, noUncatalogued: true, noModelFacts: true}
	if err := checkProductionClosures(root, "store", "internal/model", false, requirements); err == nil {
		t.Fatal("uncatalogued model policy passed")
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateAll)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = (%+v, %v)", candidates, err)
	}
	row := triageRow{
		Kind: candidates[0].Kind, Name: candidates[0].Name, File: candidates[0].File,
		Scope: candidates[0].Scope, Line: candidates[0].Line,
		Tier: string(closureledger.TierImplementation), Status: string(closureledger.StatusClosed),
		Understanding: "Fixture model policy.", ClosurePath: "Move to fixture recipe.", RerankTrigger: "Fixture recipe change.",
	}
	emitTriage := func() {
		t.Helper()
		encoded, err := json.Marshal(triageFile{Rows: []triageRow{row}})
		if err != nil {
			t.Fatal(err)
		}
		triagePath := filepath.Join(root, "triage.json")
		if err := os.WriteFile(triagePath, encoded, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := emit(root, "store", triagePath, candidates); err != nil {
			t.Fatal(err)
		}
	}
	emitTriage()
	if err := checkProductionClosures(root, "store", "internal/model", false, requirements); err != nil {
		t.Fatal(err)
	}
	row.Tier, row.Status = string(closureledger.TierDerivationBlocked), string(closureledger.StatusOpen)
	emitTriage()
	if err := checkProductionClosures(root, "store", "internal/model", false, requirements); err == nil {
		t.Fatal("open model fact passed")
	}
	if err := checkProductionClosures(root, "store", "internal/model", false, closureRequirements{zeroOpen: true}); err == nil {
		t.Fatal("open closure row passed")
	}
	testPath := filepath.Join(root, "internal", "model", "policy_test.go")
	if err := os.WriteFile(testPath, []byte("package model\nfunc verify() { if allowed(1) == 7 { panic(\"policy\") } }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkTestAuthority(snapshot, testRequirements{noPolicyCopies: true, classifiedFixtures: true}); err == nil {
		t.Fatal("copied test policy passed")
	}
}

func mustActiveAlias(t *testing.T, binding closureledger.SourceBinding) string {
	t.Helper()
	alias, err := closureledger.ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	return alias
}
