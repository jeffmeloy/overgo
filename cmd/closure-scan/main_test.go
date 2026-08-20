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
	if first.Schema != closurescan.CensusSchema || first.Source != snapshot.Identity() || len(first.Owners) != 1 || len(first.Repeated) != 1 {
		t.Fatalf("census = %+v", first)
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
	candidates, err := closurescan.ScanRoot(root)
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
		Name: candidate.Name, File: candidate.File, Scope: candidate.Scope, Line: candidate.Line,
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
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
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

func mustActiveAlias(t *testing.T, binding closureledger.SourceBinding) string {
	t.Helper()
	alias, err := closureledger.ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	return alias
}
