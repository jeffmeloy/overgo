package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

func TestUnclassifiedReportSeparatesPriorNewAndConflictingDecisions(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`package policy
const Active = 7
const Prior = 9
const Fresh = 11
const Conflict = 13
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 4 {
		t.Fatalf("candidates = (%d, %v)", len(candidates), err)
	}
	byName := map[string]closurescan.Candidate{}
	for _, candidate := range candidates {
		byName[candidate.Name] = candidate
	}
	newDocument := func(name, understanding string, reviewedCallsite bool) closureledger.Document {
		t.Helper()
		candidate := byName[name]
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		if reviewedCallsite {
			binding.CallsiteID = strings.Repeat("0", sha256.Size*2)
		}
		document, err := closureledger.New(
			candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			understanding, []closureledger.SourceBinding{binding},
			"Replace with typed fixture authority.", "Fixture contract change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		return document
	}
	active := newDocument("Active", "Active fixture policy.", false)
	priorOlder := newDocument("Prior", "Superseded prior fixture policy.", true)
	prior := newDocument("Prior", "Reviewed prior fixture policy.", true)
	priorSameDecisionBinding := prior.Bindings[0]
	priorSameDecisionBinding.CallsiteID = strings.Repeat("1", sha256.Size*2)
	priorSameDecision, err := closureledger.New(
		prior.Name, prior.Value, prior.Tier, prior.Status, prior.Understanding,
		[]closureledger.SourceBinding{priorSameDecisionBinding}, prior.ClosurePath, prior.RerankTrigger, prior.Fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	conflictFirst := newDocument("Conflict", "First conflicting policy.", false)
	conflictLatest := newDocument("Conflict", "Second conflicting policy.", false)
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation,
		[]closureledger.Document{active, priorOlder, conflictFirst}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation,
		[]closureledger.Document{priorSameDecision}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation,
		[]closureledger.Document{prior, conflictLatest}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	retire := func(name string, document closureledger.Document) artifact.AliasBinding {
		t.Helper()
		binding, err := byName[name].Binding()
		if err != nil {
			t.Fatal(err)
		}
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.AliasBinding{
			Name: alias, Target: document.ID, Previous: artifact.IDPointer(document.ID), Remove: true,
		}
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:     "fixture/manual-retirement",
		Aliases: []artifact.AliasBinding{retire("Conflict", conflictLatest)},
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
	store, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	report, err := buildUnclassifiedReport(t.Context(), snapshot, store)
	closeErr := store.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("report = (%v, close=%v)", err, closeErr)
	}
	wantCounts := unclassifiedReportCounts{
		Constants: 4, Classified: 1, Unclassified: 3, RecoverablePrior: 1,
		ReviewedCallsite: 1, GenuinelyNew: 1, ConflictingPrior: 1, TriageProposalRows: 1,
	}
	if report.Counts != wantCounts {
		t.Fatalf("counts = %+v, want %+v", report.Counts, wantCounts)
	}
	entries := map[string]unclassifiedCandidate{}
	for _, entry := range report.Candidates {
		entries[entry.Current.Name] = entry
	}
	priorEntry := entries["Prior"]
	selectedPrior := false
	if len(priorEntry.History) == 2 {
		for _, provenance := range priorEntry.History[1].Provenance {
			selectedPrior = selectedPrior || provenance.SelectedByAlias && provenance.Match == historyMatchReviewed &&
				provenance.RecoveryBlocker == "alias-currently-bound"
		}
	}
	if priorEntry.Category != unclassifiedRecoverable || len(priorEntry.History) != 2 ||
		len(priorEntry.History[1].Provenance) != 2 || !selectedPrior ||
		priorEntry.Proposed == nil || priorEntry.Proposed.Understanding != "Reviewed prior fixture policy." {
		t.Fatalf("prior entry = %+v", priorEntry)
	}
	if fresh := entries["Fresh"]; fresh.Category != unclassifiedNew || len(fresh.History) != 0 || fresh.Proposed != nil {
		t.Fatalf("fresh entry = %+v", fresh)
	}
	if conflict := entries["Conflict"]; conflict.Category != unclassifiedConflicting || len(conflict.History) != 2 || conflict.Proposed != nil {
		t.Fatalf("conflict entry = %+v", conflict)
	}
	if len(report.TriageProposal.Rows) != 1 || report.TriageProposal.Rows[0].Name != "Prior" {
		t.Fatalf("triage proposal = %+v", report.TriageProposal)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	repeated, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, repeated) {
		t.Fatal("inventory JSON is not deterministic")
	}
	text := string(encoded)
	if strings.Contains(text, `"commit":[`) || !strings.Contains(text, `"commit":"`) {
		t.Fatalf("commit identity is not a canonical string: %s", text)
	}
	historyJSON, err := json.Marshal(priorEntry.History)
	if err != nil {
		t.Fatal(err)
	}
	if count := strings.Count(string(historyJSON), `"understanding":"Reviewed prior fixture policy."`); count != 1 {
		t.Fatalf("identical decision fields encoded %d times", count)
	}
}

func TestUnclassifiedPolicyReportIncludesEveryPolicyKindReadOnly(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`package policy
import "example.org/statistics"
const Active = 7
const Missing = 9
func decide(samples []float64, count int) bool {
	_ = samples[0]
	_ = statistics.Quantile(samples)
	independent := localIID(samples)
	return count > 3 && independent
}
func localIID([]float64) bool { return true }
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateAll)
	if err != nil {
		t.Fatal(err)
	}
	policyByKind := map[closureledger.BindingKind][]closurescan.Candidate{}
	for _, candidate := range candidates {
		if candidate.Policy {
			policyByKind[candidate.Kind] = append(policyByKind[candidate.Kind], candidate)
		}
	}
	if got := len(policyByKind[closureledger.BindingConstant]); got != 2 {
		t.Fatalf("policy constants = %d, want 2; candidates=%+v", got, candidates)
	}
	if got := len(policyByKind[closureledger.BindingLiteral]); got != 1 {
		t.Fatalf("policy literals = %d, want 1; candidates=%+v", got, candidates)
	}
	if got := len(policyByKind[closureledger.BindingAssumption]); got != 1 {
		t.Fatalf("policy assumptions = %d, want 1; candidates=%+v", got, candidates)
	}
	byName := map[string]closurescan.Candidate{}
	for _, candidate := range policyByKind[closureledger.BindingConstant] {
		byName[candidate.Name] = candidate
	}
	newDocument := func(candidate closurescan.Candidate, understanding string) closureledger.Document {
		t.Helper()
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, err := closureledger.New(
			candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			understanding, []closureledger.SourceBinding{binding},
			"Replace with typed fixture authority.", "Fixture contract change.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		return document
	}
	active := newDocument(byName["Active"], "Active fixture policy.")
	literal := policyByKind[closureledger.BindingLiteral][0]
	priorLiteral := newDocument(literal, "Reviewed literal fixture policy.")
	storePath := filepath.Join(root, "store")
	if _, _, err := commitClosureDocuments(
		root, storePath, closurePublishOperation, []closureledger.Document{active, priorLiteral}, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	literalBinding, err := literal.Binding()
	if err != nil {
		t.Fatal(err)
	}
	literalAlias, err := closureledger.ActiveAlias(literalBinding)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/manual-retirement",
		Aliases: []artifact.AliasBinding{{
			Name: literalAlias, Target: priorLiteral.ID, Previous: artifact.IDPointer(priorLiteral.ID), Remove: true,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	headBefore, sequenceBefore := store.Head()
	report, reportErr := buildUnclassifiedPolicyReport(t.Context(), snapshot, store)
	headAfter, sequenceAfter := store.Head()
	closeErr := store.Close()
	if reportErr != nil || closeErr != nil {
		t.Fatalf("report = (%v, close=%v)", reportErr, closeErr)
	}
	if headAfter != headBefore || sequenceAfter != sequenceBefore {
		t.Fatalf("read-only inventory changed store head: %s@%d -> %s@%d", headBefore, sequenceBefore, headAfter, sequenceAfter)
	}
	wantCounts := unclassifiedReportCounts{
		PolicyCandidates: 4, Constants: 2, Literals: 1, Assumptions: 1,
		Classified: 1, Unclassified: 3, RecoverablePrior: 1, GenuinelyNew: 2, TriageProposalRows: 1,
	}
	if report.Selection != "production-policy" || report.Counts != wantCounts {
		t.Fatalf("report selection/counts = %q %+v, want production-policy %+v", report.Selection, report.Counts, wantCounts)
	}
	byKind := map[closureledger.BindingKind][]unclassifiedCandidate{}
	for _, candidate := range report.Candidates {
		if !candidate.Current.Policy {
			t.Fatalf("non-policy candidate reported: %+v", candidate.Current)
		}
		byKind[candidate.Current.Kind] = append(byKind[candidate.Current.Kind], candidate)
	}
	if len(byKind[closureledger.BindingConstant]) != 1 || len(byKind[closureledger.BindingLiteral]) != 1 ||
		len(byKind[closureledger.BindingAssumption]) != 1 {
		t.Fatalf("unclassified kinds = %+v", byKind)
	}
	literalEntry := byKind[closureledger.BindingLiteral][0]
	if literalEntry.Category != unclassifiedRecoverable || len(literalEntry.History) != 1 ||
		literalEntry.Proposed == nil || literalEntry.Proposed.Kind != closureledger.BindingLiteral {
		t.Fatalf("literal history = %+v", literalEntry)
	}
	proposalJSON, err := json.Marshal(report.TriageProposal)
	if err != nil {
		t.Fatal(err)
	}
	var proposal triageFile
	if err := json.Unmarshal(proposalJSON, &proposal); err != nil {
		t.Fatalf("triage proposal is not reusable by -triage: %v", err)
	}
	if len(proposal.Rows) != 1 || proposal.Rows[0] != *literalEntry.Proposed {
		t.Fatalf("triage proposal = %+v, want %+v", proposal, literalEntry.Proposed)
	}
	var firstText, secondText bytes.Buffer
	if err := writeUnclassifiedPolicyReport(&firstText, report); err != nil {
		t.Fatal(err)
	}
	if err := writeUnclassifiedPolicyReport(&secondText, report); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstText.Bytes(), secondText.Bytes()) {
		t.Fatal("policy inventory text is not deterministic")
	}
	if text := firstText.String(); !strings.Contains(text,
		"policy_candidates=4 constants=2 literals=1 assumptions=1 classified=1 unclassified=3") ||
		!strings.Contains(text, "recoverable-prior-decision exact literal history=1") {
		t.Fatalf("policy inventory text = %s", text)
	}
	firstJSON, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatal("policy inventory JSON is not deterministic")
	}
}
