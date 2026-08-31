package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
	"overgo/internal/testevidence"
)

const closureAuthorityTestEnv = "OVERGO_CLOSURE_AUTHORITY_TEST"

func requireRepositoryClosureAuthority(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(closureAuthorityTestEnv) != "1" {
		t.Skip("set " + closureAuthorityTestEnv + "=1 to verify the current worktree against shared closure authority")
	}
}

func TestConfigurationClosureGates(t *testing.T) {
	requireRepositoryClosureAuthority(t)
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
		if err := checkProductionClosures(root, "overgodb-store", check.scope, false, check.requirements); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPermanentMagicGateRepositoryZeroDebt(t *testing.T) {
	requireRepositoryClosureAuthority(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	requirements := closureRequirements{
		classified: true, noStale: true, noUncatalogued: true, noModelFacts: true, zeroOpen: true,
	}
	if err := checkProductionClosures(root, "overgodb-store", "", true, requirements); err != nil {
		t.Fatal(err)
	}
	if err := checkTestAuthority(mustSnapshot(root), testRequirements{noPolicyCopies: true}); err != nil {
		t.Fatal(err)
	}
}

func TestRepositoryGoStyleTests(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	sites, err := closurescan.CensusTestLiterals(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, site := range sites {
		if site.File == "cmd/benchmark/main_test.go" && site.Class == closurescan.TestPolicyCopy {
			t.Errorf("%s:%d copies production policy %s", site.File, site.Line, site.Value)
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
	if err := writeCensusText(&firstText, first); err != nil {
		t.Fatal(err)
	}
	if err := writeCensusText(&secondText, second); err != nil {
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
	classes := want.Structural + want.Mathematical + want.Format + want.Capacity + want.Policy + want.ModelFact + want.Unknown
	if named != want.NamedConstants || inline != want.InlineLiterals || assumptions != want.AssumptionHints ||
		policy != want.TestPolicyCopies || groups != want.RepeatedGroups || sites != want.RepeatedSites ||
		classes != want.InlineLiterals || want.TestLiterals != want.TestFixtures+want.TestAssertions+want.TestPolicyCopies {
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
	evidence, err := publishCensusEvidence(t.Context(), store, snapshot, report)
	if err != nil {
		t.Fatal(err)
	}
	loaded, found, err := closurescan.ReadCensusEvidence(t.Context(), store, evidence.ID)
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
	evidence, err := publishCensusEvidence(t.Context(), store, snapshot, report)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Source != snapshot.Identity() || evidence.CatalogHead != head.String() || evidence.CatalogSequence == 0 {
		t.Fatalf("evidence = %+v", evidence)
	}
}

func censusEvidenceFixture(t *testing.T) (*overgodb.Store, repoanalysis.SourceSnapshot, artifact.CommitID) {
	t.Helper()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	seed, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("catalog seed"))
	if err != nil {
		t.Fatal(err)
	}
	head, err := store.Commit(t.Context(), artifact.Batch{
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

func TestTriagePublishesAndRebindsExactBindings(t *testing.T) {
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
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "store"))
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
		t.Context(), store, binding, candidate.ValueJSON(),
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
	if count, unmatched, first, _, err := importClosureDocuments(root, "store", filepath.Join(root, "store"), snapshot, false, false); err != nil || count != len(triage.Rows) || unmatched != 0 || first != "" {
		t.Fatalf("rebound documents = (%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
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
	store, err = overgodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, active, err := closureledger.ResolveActiveBinding(t.Context(), store, binding, candidate.ValueJSON()); err != nil || !active {
		t.Fatalf("rebound document = (%t, %v)", active, err)
	}
	previousAlias, currentAlias := mustActiveAlias(t, previous), mustActiveAlias(t, binding)
	if previousAlias != currentAlias {
		t.Fatalf("structural alias moved: %s != %s", previousAlias, currentAlias)
	}
	if _, found, err := artifact.ResolveAlias(t.Context(), store, previousAlias); err != nil || !found {
		t.Fatalf("stable binding resolves = (%t, %v)", found, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := importClosureDocuments(root, "store", filepath.Join(root, "store"), snapshot, false, false); err != nil {
		t.Fatalf("retain live bindings: %v", err)
	}
	if err := os.WriteFile(path, source, 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err = repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := importClosureDocuments(root, "store", filepath.Join(root, "store"), snapshot, false, false); err != nil {
		t.Fatalf("restore historical binding: %v", err)
	}
	candidates, err = closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	for _, current := range candidates {
		if current.Name == candidate.Name {
			binding, err = current.Binding()
			break
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	snapshot, err = repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, review, err := importClosureDocuments(root, "store", filepath.Join(root, "store"), snapshot, false, false); err != nil {
		t.Fatalf("preserve orphan bindings: %v", err)
	} else if review != (closureAliasReview{Reviewed: 1, Preserved: 1}) {
		t.Fatalf("default orphan review = %+v", review)
	}
	store, err = overgodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, found, err := artifact.ResolveAlias(t.Context(), store, mustActiveAlias(t, binding)); err != nil || !found {
		t.Fatalf("preserved binding resolves = (%t, %v)", found, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	count, unmatched, first, review, err := importClosureDocuments(
		root, "store", filepath.Join(root, "store"), snapshot, false, true,
	)
	if err != nil || count != 0 || unmatched != 1 || first != "PolicyWindow:declaration" {
		t.Fatalf("retire orphan binding = (%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
	}
	if review != (closureAliasReview{Reviewed: 1, Retired: 1}) {
		t.Fatalf("explicit orphan review = %+v", review)
	}
	store, err = overgodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, found, err := artifact.ResolveAlias(t.Context(), store, mustActiveAlias(t, binding)); err != nil || found {
		t.Fatalf("retired binding resolves = (%t, %v)", found, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = nil
	if count, unmatched, first, _, err := importClosureDocuments(
		root, "store", filepath.Join(root, "store"), snapshot, false, false,
	); err != nil || count != 0 || unmatched != 0 || first != "" {
		t.Fatalf("explicit retirement recovery = (%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
	}
	store, err = overgodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if _, found, err := artifact.ResolveAlias(t.Context(), store, mustActiveAlias(t, binding)); err != nil || found {
		t.Fatalf("explicitly retired binding recovered = (%t, %v)", found, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSameStoreImportLeavesUnprovenMatchingRetirementInactive(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"internal/required/policy.go":  "package required\nconst Required = 7\n",
		"internal/retired/policy.go":   "package retired\nconst Retired = 9\n",
		"internal/unrelated/policy.go": "package unrelated\nconst Unrelated = 11\n",
	}
	for relative, content := range files {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != len(files) {
		t.Fatalf("candidates = (%d, %v)", len(candidates), err)
	}
	documents := make(map[string]closureledger.Document, len(candidates))
	bindings := make(map[string]closureledger.SourceBinding, len(candidates))
	for _, candidate := range candidates {
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, err := closureledger.New(
			candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			"Fixture policy remains explicit.", []closureledger.SourceBinding{binding},
			"Replace with fixture authority.", "Fixture contract change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		documents[candidate.Name] = document
		bindings[candidate.Name] = binding
	}
	ordered := []closureledger.Document{documents["Required"], documents["Retired"], documents["Unrelated"]}
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(root, storePath, closurePublishOperation, ordered, nil, nil); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	var retirements []artifact.AliasBinding
	for _, name := range []string{"Required", "Retired"} {
		alias := mustActiveAlias(t, bindings[name])
		retirements = append(retirements, artifact.AliasBinding{
			Name: alias, Target: documents[name].ID, Previous: artifact.IDPointer(documents[name].ID), Remove: true,
		})
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/closure/same-store-retire", Aliases: retirements}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for _, relative := range []string{"internal/retired/policy.go", "internal/unrelated/policy.go"} {
		if err := os.Remove(filepath.Join(root, filepath.FromSlash(relative))); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	count, unmatched, first, _, err := importClosureDocuments(root, "store", storePath, snapshot, false, false)
	if err != nil || count != 0 || unmatched != 1 || first != "Unrelated:declaration" {
		t.Fatalf("same-store unproven retirement = (%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
	}
	store, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, check := range []struct {
		name string
		want bool
	}{{"Required", false}, {"Retired", false}, {"Unrelated", true}} {
		_, found, err := artifact.ResolveAlias(t.Context(), store, mustActiveAlias(t, bindings[check.name]))
		if err != nil || found != check.want {
			t.Fatalf("%s active = (%t, %v), want %t", check.name, found, err, check.want)
		}
	}
}

func TestSameStoreImportRecoversAutomaticTechnicalRebind(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldPath := filepath.Join(root, "internal", "policy", "policy.go")
	newPath := filepath.Join(root, "internal", "policy2", "policy.go")
	writePolicy := func(path, pkg string) closurescan.Candidate {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("package "+pkg+"\nconst Policy = 7\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
		if err != nil || len(candidates) != 1 {
			t.Fatalf("candidates = (%d, %v)", len(candidates), err)
		}
		return candidates[0]
	}
	oldCandidate := writePolicy(oldPath, "policy")
	oldBinding, err := oldCandidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	oldDocument, err := closureledger.New(
		oldCandidate.Name, oldCandidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
		"Fixture policy remains explicit.", []closureledger.SourceBinding{oldBinding},
		"Replace with fixture authority.", "Fixture contract change.", oldBinding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation, []closureledger.Document{oldDocument}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(oldPath); err != nil {
		t.Fatal(err)
	}
	newCandidate := writePolicy(newPath, "policy2")
	newBinding, err := newCandidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	successor, err := closureledger.New(
		oldDocument.Name, oldDocument.Value, oldDocument.Tier, oldDocument.Status, oldDocument.Understanding,
		[]closureledger.SourceBinding{newBinding}, oldDocument.ClosurePath, oldDocument.RerankTrigger, newBinding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	oldAlias, newAlias := mustActiveAlias(t, oldBinding), mustActiveAlias(t, newBinding)
	if oldAlias == newAlias {
		t.Fatal("fixture move did not change its active alias")
	}
	retirement := artifact.AliasBinding{
		Name: oldAlias, Target: oldDocument.ID, Previous: artifact.IDPointer(oldDocument.ID), Remove: true,
	}
	if _, _, err := commitClosureDocuments(
		root, storePath, closureRebindOperation, []closureledger.Document{successor}, []artifact.AliasBinding{retirement}, nil,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(newPath); err != nil {
		t.Fatal(err)
	}
	writePolicy(oldPath, "policy")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	count, unmatched, first, _, err := importClosureDocuments(root, "store", storePath, snapshot, false, false)
	if err != nil || count != 1 || unmatched != 1 || first != "Policy:declaration" {
		t.Fatalf("technical recovery = (%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	for alias, want := range map[string]bool{oldAlias: true, newAlias: true} {
		if _, found, err := artifact.ResolveAlias(t.Context(), store, alias); err != nil || found != want {
			t.Fatalf("alias %s active = (%t, %v), want %t", alias, found, err, want)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	writer, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	malformedAlias := closureledger.ActiveAliasPrefix + strings.Repeat("0", sha256.Size*2)
	if _, err := writer.Commit(t.Context(), artifact.Batch{
		Key: string(closureRebindOperation) + strings.Repeat("a", sha256.Size*2),
		Aliases: []artifact.AliasBinding{
			{Name: oldAlias, Target: oldDocument.ID, Previous: artifact.IDPointer(oldDocument.ID), Remove: true},
			{Name: malformedAlias, Target: successor.ID},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	count, unmatched, first, _, err = importClosureDocuments(root, "store", storePath, snapshot, false, false)
	if err != nil || count != 0 || unmatched != 1 || first != "Policy:declaration" {
		t.Fatalf("malformed successor recovery = (%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
	}
	store, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := artifact.ResolveAlias(t.Context(), store, oldAlias); err != nil || found {
		t.Fatalf("malformed successor recovered old alias = (%t, %v)", found, err)
	}
}

type closureChainFixture struct {
	root, storePath       string
	first, successor      closureledger.Document
	firstAlias, nextAlias string
	nextBinding           closureledger.SourceBinding
}

func setupClosureChainFixture(t *testing.T) closureChainFixture {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"internal/first/policy.go": "package first\nconst Policy = 7\n",
		"internal/next/policy.go":  "package next\nconst Policy = 7\n",
	}
	for relative, content := range files {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 2 {
		t.Fatalf("candidates = (%d, %v)", len(candidates), err)
	}
	bindings := map[string]closureledger.SourceBinding{}
	for _, candidate := range candidates {
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		bindings[candidate.Package] = binding
	}
	decision := func(binding closureledger.SourceBinding) closureledger.Document {
		document, err := closureledger.New(
			"Policy", candidates[0].ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			"Shared chain fixture policy.", []closureledger.SourceBinding{binding},
			"Replace with fixture authority.", "Fixture contract change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		return document
	}
	first := decision(bindings["internal/first"])
	successor := decision(bindings["internal/next"])
	firstAlias := mustActiveAlias(t, bindings["internal/first"])
	nextAlias := mustActiveAlias(t, bindings["internal/next"])
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation, []closureledger.Document{first}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	if _, _, err := commitClosureDocuments(
		root, storePath, closureRebindOperation, []closureledger.Document{successor},
		[]artifact.AliasBinding{{
			Name: firstAlias, Target: first.ID, Previous: artifact.IDPointer(first.ID), Remove: true,
		}}, nil,
	); err != nil {
		t.Fatal(err)
	}
	return closureChainFixture{
		root: root, storePath: storePath, first: first, successor: successor,
		firstAlias: firstAlias, nextAlias: nextAlias, nextBinding: bindings["internal/next"],
	}
}

func analyzeClosureChainFixture(t *testing.T, fixture closureChainFixture) closureRecoveryAnalysis {
	t.Helper()
	store, err := overgodb.OpenReadOnly(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	documents, err := allClosureDocuments(t.Context(), store)
	if err != nil {
		t.Fatal(err)
	}
	analysis, err := analyzeClosureRecovery(t.Context(), store, documents)
	closeErr := store.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("analysis = (%v, close=%v)", err, closeErr)
	}
	return analysis
}

func TestRecoveryAuthorityRequiresLiveSuccessorChain(t *testing.T) {
	t.Run("explicit successor retirement", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRetireUnmatchedOperation, nil,
			[]artifact.AliasBinding{{
				Name: fixture.nextAlias, Target: fixture.successor.ID,
				Previous: artifact.IDPointer(fixture.successor.ID), Remove: true,
			}}, nil,
		); err != nil {
			t.Fatal(err)
		}
		analysis := analyzeClosureChainFixture(t, fixture)
		if _, found := analysis.Authorized[fixture.firstAlias]; found ||
			analysis.Dispositions[fixture.firstAlias].Blocker != "successor-chain-not-live" ||
			analysis.Dispositions[fixture.nextAlias].Blocker != "later-explicit-retirement" {
			t.Fatalf("retired successor analysis = %+v", analysis)
		}
	})
	t.Run("conflicting successor supersession", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		conflict, err := closureledger.New(
			fixture.successor.Name, fixture.successor.Value, fixture.successor.Tier, fixture.successor.Status,
			"Conflicting successor policy.", []closureledger.SourceBinding{fixture.nextBinding},
			fixture.successor.ClosurePath, fixture.successor.RerankTrigger, fixture.nextBinding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closurePublishOperation, []closureledger.Document{conflict}, nil, nil,
		); err != nil {
			t.Fatal(err)
		}
		analysis := analyzeClosureChainFixture(t, fixture)
		if _, found := analysis.Authorized[fixture.firstAlias]; found ||
			analysis.Dispositions[fixture.firstAlias].Blocker != "successor-chain-not-live" {
			t.Fatalf("superseded successor analysis = %+v", analysis)
		}
	})
	t.Run("reactivation audit follows retired successor", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRebindOperation,
			[]closureledger.Document{fixture.first}, nil, nil,
		); err != nil {
			t.Fatal(err)
		}
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRetireUnmatchedOperation, nil,
			[]artifact.AliasBinding{{
				Name: fixture.nextAlias, Target: fixture.successor.ID,
				Previous: artifact.IDPointer(fixture.successor.ID), Remove: true,
			}}, nil,
		); err != nil {
			t.Fatal(err)
		}
		analysis := analyzeClosureChainFixture(t, fixture)
		if len(analysis.Reactivations) != 1 || analysis.Reactivations[0].Alias != fixture.firstAlias ||
			analysis.Reactivations[0].Provenance != "rebind-operation-unverified" ||
			analysis.Reactivations[0].Eligible || analysis.Reactivations[0].Blocker != "successor-chain-not-live" {
			t.Fatalf("reactivation audit = %+v", analysis.Reactivations)
		}
	})
	t.Run("canonical move back is not recovery", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRebindOperation,
			[]closureledger.Document{fixture.first}, []artifact.AliasBinding{{
				Name: fixture.nextAlias, Target: fixture.successor.ID,
				Previous: artifact.IDPointer(fixture.successor.ID), Remove: true,
			}}, nil,
		); err != nil {
			t.Fatal(err)
		}
		analysis := analyzeClosureChainFixture(t, fixture)
		if len(analysis.Reactivations) != 0 {
			t.Fatalf("canonical move reported as recovery = %+v", analysis.Reactivations)
		}
	})
	t.Run("unrelated same-decision removal does not hide reactivation", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		thirdPath := filepath.Join(fixture.root, "internal", "third", "policy.go")
		if err := os.MkdirAll(filepath.Dir(thirdPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(thirdPath, []byte("package third\nconst Policy = 7\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		candidates, err := closurescan.ScanRoot(fixture.root, closurescan.CandidateConstants)
		if err != nil {
			t.Fatal(err)
		}
		var thirdBinding closureledger.SourceBinding
		for _, candidate := range candidates {
			if candidate.Package == "internal/third" {
				thirdBinding, err = candidate.Binding()
				break
			}
		}
		if err != nil || !thirdBinding.Owner.Valid() {
			t.Fatalf("third binding = (%+v, %v)", thirdBinding, err)
		}
		third, err := closureledger.New(
			fixture.first.Name, fixture.first.Value, fixture.first.Tier, fixture.first.Status,
			fixture.first.Understanding, []closureledger.SourceBinding{thirdBinding},
			fixture.first.ClosurePath, fixture.first.RerankTrigger, thirdBinding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closurePublishOperation, []closureledger.Document{third}, nil, nil,
		); err != nil {
			t.Fatal(err)
		}
		thirdAlias := mustActiveAlias(t, thirdBinding)
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRebindOperation, []closureledger.Document{fixture.first},
			[]artifact.AliasBinding{{
				Name: thirdAlias, Target: third.ID, Previous: artifact.IDPointer(third.ID), Remove: true,
			}}, nil,
		); err != nil {
			t.Fatal(err)
		}
		analysis := analyzeClosureChainFixture(t, fixture)
		if len(analysis.Reactivations) != 1 || analysis.Reactivations[0].Alias != fixture.firstAlias ||
			analysis.Reactivations[0].Provenance != "rebind-operation-unverified" {
			t.Fatalf("unrelated predecessor audit = %+v", analysis.Reactivations)
		}
	})
}

func TestReviewedClosureCommitRefusesMovedStoreHead(t *testing.T) {
	fixture := setupClosureChainFixture(t)
	expectedHead, _ := storeCoordinates(t, fixture.storePath)
	appendUnrelatedClosureCommit(t, fixture.storePath)
	if _, _, err := commitClosureDocumentsAtHead(
		fixture.root, fixture.storePath, closureRebindOperation,
		[]closureledger.Document{fixture.first}, nil, nil, &expectedHead,
	); err == nil || !errors.Is(err, overgodb.ErrHeadConflict) {
		t.Fatalf("stale reviewed closure commit error = %v", err)
	}
	store, err := overgodb.OpenReadOnly(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := artifact.ResolveAlias(t.Context(), store, fixture.firstAlias); err != nil || found {
		t.Fatalf("stale reviewed commit resurrected alias = (%t, %v)", found, err)
	}
}

func TestReboundClosureAliasIsExcludedFromFollowingRetirement(t *testing.T) {
	fixture := setupClosureChainFixture(t)
	retirements := []artifact.AliasBinding{
		{
			Name: fixture.nextAlias, Target: fixture.successor.ID,
			Previous: artifact.IDPointer(fixture.successor.ID), Remove: true,
		},
		{
			Name: fixture.firstAlias, Target: fixture.first.ID,
			Previous: artifact.IDPointer(fixture.first.ID), Remove: true,
		},
	}

	filtered, excluded, err := excludeReboundClosureRetirements(
		retirements, []closureledger.Document{fixture.successor},
	)
	if err != nil {
		t.Fatal(err)
	}
	if excluded != 1 || len(filtered) != 1 || filtered[0].Name != fixture.firstAlias {
		t.Fatalf("filtered retirements = (%+v, excluded=%d)", filtered, excluded)
	}
}

func TestClosureImportRefusesConflictingDecisionClaims(t *testing.T) {
	fixture := setupClosureChainFixture(t)
	conflict, err := closureledger.New(
		fixture.successor.Name, fixture.successor.Value,
		fixture.successor.Tier, fixture.successor.Status,
		"Conflicting reviewed fixture policy.", []closureledger.SourceBinding{fixture.nextBinding},
		fixture.successor.ClosurePath, fixture.successor.RerankTrigger, fixture.nextBinding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := storeCoordinates(t, fixture.storePath)
	_, _, err = commitClosureDocuments(
		fixture.root, fixture.storePath, closureRebindOperation,
		[]closureledger.Document{fixture.successor, conflict}, nil, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "conflicting reviewed decisions") {
		t.Fatalf("conflicting alias claim error = %v", err)
	}
	if currentHead, currentSequence := storeCoordinates(t, fixture.storePath); currentHead != head || currentSequence != sequence {
		t.Fatalf("conflicting alias claim moved store: %s@%d -> %s@%d", head, sequence, currentHead, currentSequence)
	}
}

func TestClosureRetirementRequiresSettledRebind(t *testing.T) {
	fixture := setupClosureChainFixture(t)
	path := filepath.Join(fixture.root, "internal", "next", "policy.go")
	if err := os.WriteFile(path, []byte("// reviewed source movement\npackage next\nconst Policy = 7\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(fixture.root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := storeCoordinates(t, fixture.storePath)
	_, _, _, _, err = importClosureDocuments(
		fixture.root, "store", fixture.storePath, snapshot, false, true,
	)
	if err == nil || !strings.Contains(err.Error(), "retirement requires a settled rebind") {
		t.Fatalf("unsettled retirement error = %v", err)
	}
	if currentHead, currentSequence := storeCoordinates(t, fixture.storePath); currentHead != head || currentSequence != sequence {
		t.Fatalf("unsettled retirement moved store: %s@%d -> %s@%d", head, sequence, currentHead, currentSequence)
	}
}

func TestSameStoreImportDoesNotResurrectSupersededDecision(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(value string) closurescan.Candidate {
		t.Helper()
		if err := os.WriteFile(path, []byte("package policy\nconst Policy = "+value+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
		if err != nil || len(candidates) != 1 {
			t.Fatalf("candidates = (%d, %v)", len(candidates), err)
		}
		return candidates[0]
	}
	newDocument := func(candidate closurescan.Candidate) (closureledger.Document, closureledger.SourceBinding) {
		t.Helper()
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, err := closureledger.New(
			candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			"Latest reviewed fixture policy.", []closureledger.SourceBinding{binding},
			"Replace with fixture authority.", "Fixture contract change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		return document, binding
	}
	first, firstBinding := newDocument(write("7"))
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(root, storePath, closurePublishOperation, []closureledger.Document{first}, nil, nil); err != nil {
		t.Fatal(err)
	}
	latest, latestBinding := newDocument(write("9"))
	if mustActiveAlias(t, latestBinding) != mustActiveAlias(t, firstBinding) {
		t.Fatal("fixture declaration identity changed with its reviewed value")
	}
	if _, _, err := commitClosureDocuments(root, storePath, closurePublishOperation, []closureledger.Document{latest}, nil, nil); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	alias := mustActiveAlias(t, latestBinding)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/closure/retire-latest",
		Aliases: []artifact.AliasBinding{{
			Name: alias, Target: latest.ID, Previous: artifact.IDPointer(latest.ID), Remove: true,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	write("7")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if count, unmatched, firstReason, _, err := importClosureDocuments(root, "store", storePath, snapshot, false, false); err != nil || count != 0 || unmatched != 0 || firstReason != "" {
		t.Fatalf("same-store superseded recovery = (%d, unmatched=%d first=%s, %v)", count, unmatched, firstReason, err)
	}
	store, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := artifact.ResolveAlias(t.Context(), store, alias); err != nil || found {
		t.Fatalf("superseded decision active = (%t, %v)", found, err)
	}
}

func TestSameStoreImportRejectsConflictingHistoricalAuthority(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package policy\nconst Policy = 7\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = (%d, %v)", len(candidates), err)
	}
	binding, err := candidates[0].Binding()
	if err != nil {
		t.Fatal(err)
	}
	decision := func(understanding string) closureledger.Document {
		t.Helper()
		document, err := closureledger.New(
			candidates[0].Name, candidates[0].ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			understanding, []closureledger.SourceBinding{binding},
			"Replace with fixture authority.", "Fixture contract change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		return document
	}
	first := decision("First reviewed policy interpretation.")
	second := decision("Conflicting reviewed policy interpretation.")
	storePath := filepath.Join(root, "store")
	for _, document := range []closureledger.Document{first, second} {
		if _, _, err := commitClosureDocuments(root, storePath, closurePublishOperation, []closureledger.Document{document}, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	alias := mustActiveAlias(t, binding)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/closure/retire-conflict",
		Aliases: []artifact.AliasBinding{{
			Name: alias, Target: second.ID, Previous: artifact.IDPointer(second.ID), Remove: true,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if count, unmatched, firstReason, _, err := importClosureDocuments(root, "store", storePath, snapshot, false, false); err != nil || count != 0 || unmatched != 0 || firstReason != "" {
		t.Fatalf("same-store conflicting recovery = (%d, unmatched=%d first=%s, %v)", count, unmatched, firstReason, err)
	}
	store, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := artifact.ResolveAlias(t.Context(), store, alias); err != nil || found {
		t.Fatalf("conflicting historical authority active = (%t, %v)", found, err)
	}
}

func TestClosurePublicationDeltaCopiesFixtureAndSkipsRepeat(t *testing.T) {
	root := t.TempDir()
	relative := "internal/sample/policy.go"
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package sample\nconst RequestLimit = 7\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates=%d err=%v", len(candidates), err)
	}
	candidate := candidates[0]
	binding, err := candidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	fixtureID, err := artifact.IdentifyBytes(artifact.KindFile, []byte("external fixture"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
		"Request admission bound.", []closureledger.SourceBinding{binding}, "Typed request policy.", "Request admission contract change.", fixtureID)
	if err != nil {
		t.Fatal(err)
	}
	fixture := artifact.Descriptor{ID: fixtureID, Size: 16}
	if _, _, err := commitClosureDocuments(root, filepath.Join(root, "source"), closurePublishOperation, []closureledger.Document{document}, nil, []artifact.Descriptor{fixture}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := commitClosureDocuments(root, filepath.Join(root, "source"), closurePublishOperation, []closureledger.Document{document}, nil, []artifact.Descriptor{fixture}); err != nil {
		t.Fatal(err)
	}
	source, err := overgodb.OpenReadOnly(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	if _, sequence := source.Head(); sequence != 1 {
		t.Fatalf("closure publication sequence = %d, want 1", sequence)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := importClosureDocuments(root, "target", filepath.Join(root, "source"), snapshot, false, true); err == nil || !strings.Contains(err.Error(), "same store") {
		t.Fatalf("cross-store retirement error = %v", err)
	}
	count, unmatched, first, _, err := importClosureDocuments(root, "target", filepath.Join(root, "source"), snapshot, false, false)
	if err != nil || count != 1 || unmatched != 0 || first != "" {
		t.Fatalf("import=(%d, unmatched=%d first=%s, %v)", count, unmatched, first, err)
	}
	target, err := overgodb.OpenReadOnly(filepath.Join(root, "target"))
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	if _, found, err := closureledger.ResolveActiveBinding(t.Context(), target, binding, candidate.ValueJSON()); err != nil || !found {
		t.Fatalf("imported binding=(%t, %v)", found, err)
	}
	result, err := target.Query(t.Context(), overgodb.Query{
		Artifact: &fixtureID, MaxResults: 1, Projection: overgodb.ProjectArtifacts,
	})
	if err != nil || len(result.Artifacts) != 1 || result.Artifacts[0] != fixture {
		t.Fatalf("imported fixture=(%+v, %v)", result.Artifacts, err)
	}
}

func TestClosurePublicationDeduplicatesContentWithinBatch(t *testing.T) {
	root := t.TempDir()
	relative := "internal/sample/policy.go"
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package sample\nconst RequestLimit = 7\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates=%d err=%v", len(candidates), err)
	}
	candidate := candidates[0]
	binding, err := candidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
		"Request admission bound.", []closureledger.SourceBinding{binding},
		"Typed request policy.", "Request admission contract change.", binding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := commitClosureDocuments(
		root, filepath.Join(root, "store"), closurePublishOperation,
		[]closureledger.Document{document, document}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if found, err := store.HasContent(t.Context(), document.ID); err != nil || !found {
		t.Fatalf("deduplicated content=(%t, %v)", found, err)
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
	store, err := overgodb.Open(filepath.Join(root, "store"))
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

func TestPermanentMagicGateAuthorityEnforcement(t *testing.T) {
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
	store, err := overgodb.Open(filepath.Join(root, "store"))
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
	if err := os.WriteFile(testPath, []byte("package model\nconst expectedPolicyLimit = 7\nfunc verify() { if allowed(1) == expectedPolicyLimit { panic(\"policy\") } }\n"), 0o644); err != nil {
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
