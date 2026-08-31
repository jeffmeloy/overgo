package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/overgodb"
)

func TestReactivationRemediationRequiresExactReviewedAuthority(t *testing.T) {
	t.Run("unverified rebind recovery", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		commit, sequence := reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, true)
		result, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector)
		if err != nil || result.Reviewed != 1 || result.Retired != 1 ||
			!result.Record.Valid() || result.Commit == "" {
			t.Fatalf("remediation = (%+v, %v)", result, err)
		}
		store, err := overgodb.OpenReadOnly(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		if _, found, err := artifact.ResolveAlias(t.Context(), store, fixture.firstAlias); err != nil || found {
			t.Fatalf("remediated alias active = (%t, %v)", found, err)
		}
		head, headSequence := store.Head()
		history, historyErr := readClosureAliasEventHistory(t.Context(), store, headSequence)
		record, found, recordErr := artifact.ReadContent(t.Context(), store, result.Record)
		introduction, introduced, introductionErr := store.ArtifactIntroduction(t.Context(), result.Record)
		closeErr := store.Close()
		latest := history.latest[fixture.firstAlias]
		if historyErr != nil || closeErr != nil || !latest.Binding.Remove ||
			!closureOperationKey(latest.Commit.Key, string(closureRemediateRecoveryOperation)) {
			t.Fatalf("remediation history = (%+v, history=%v close=%v)", latest, historyErr, closeErr)
		}
		if recordErr != nil || !found || record.Descriptor.Schema != closureReactivationRemediationSchema ||
			record.Descriptor.MediaType != artifact.JSONMediaType || record.Descriptor.ID.Kind() != artifact.KindEvidence {
			t.Fatalf("remediation evidence = (%+v, found=%t, %v)", record.Descriptor, found, recordErr)
		}
		var evidence closureRemediationEvidence
		if err := json.Unmarshal(record.Data, &evidence); err != nil {
			t.Fatal(err)
		}
		if evidence.Version != artifact.InitialDocumentVersion || evidence.ReactivationCommit != commit.String() ||
			evidence.ReactivationSequence != sequence || evidence.ReplacedHead != selector.ExpectedHead.String() ||
			evidence.ReplacedSequence != selector.ExpectedSequence || evidence.ReviewedCount != 1 ||
			!evidence.ConfirmUnverifiedRecovery || len(evidence.ReviewedAliases) != 1 {
			t.Fatalf("remediation evidence body = %+v", evidence)
		}
		reviewed := evidence.ReviewedAliases[0]
		if reviewed != (closureRemediationReviewedAlias{
			Alias:           fixture.firstAlias,
			RetiredDocument: fixture.first.ID, ReactivatedDocument: fixture.first.ID,
			CurrentDocument:    fixture.first.ID,
			ActivationSequence: sequence, RetirementSequence: sequence - 1, CurrentSequence: sequence,
			Blocker: "successor-chain-not-live", Provenance: "rebind-operation-unverified",
		}) {
			t.Fatalf("remediation reviewed alias = %+v", reviewed)
		}
		if !introduced || introductionErr != nil || introduction.Commit.String() != result.Commit ||
			introduction.Sequence != selector.ExpectedSequence+1 || latest.Commit.ID.String() != result.Commit ||
			latest.Commit.Sequence != introduction.Sequence || head.String() != result.Commit || headSequence != introduction.Sequence {
			t.Fatalf(
				"remediation atomic authority = (introduced=%t, intro=%+v, introErr=%v, alias=%+v, head=%s@%d)",
				introduced, introduction, introductionErr, latest.Commit, head, headSequence,
			)
		}
	})

	t.Run("stale reviewed head", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		commit, sequence := reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, true)
		appendUnrelatedClosureCommit(t, fixture.storePath)
		if _, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector); err == nil ||
			!strings.Contains(err.Error(), "head is stale") {
			t.Fatalf("stale-head remediation error = %v", err)
		}
		assertClosureAliasTarget(t, fixture.storePath, fixture.firstAlias, fixture.first.ID)
	})

	t.Run("changed reviewed count", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		commit, sequence := reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, false)
		selector.ExpectedCount = 2
		head, headSequence := storeCoordinates(t, fixture.storePath)
		if _, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector); err == nil ||
			!strings.Contains(err.Error(), "count changed") {
			t.Fatalf("changed-count remediation error = %v", err)
		}
		if currentHead, currentSequence := storeCoordinates(t, fixture.storePath); currentHead != head || currentSequence != headSequence {
			t.Fatalf("changed count mutated store: %s@%d -> %s@%d", head, headSequence, currentHead, currentSequence)
		}
		assertClosureAliasTarget(t, fixture.storePath, fixture.firstAlias, fixture.first.ID)
	})

	t.Run("reviewed later active target", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		commit, sequence := reactivateClosureChain(t, fixture, false)
		retireClosureChainSuccessor(t, fixture)
		store, err := overgodb.Open(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), artifact.Batch{
			Key: "fixture/changed-reactivation-target",
			Aliases: []artifact.AliasBinding{{
				Name: fixture.firstAlias, Target: fixture.successor.ID,
				Previous: artifact.IDPointer(fixture.first.ID),
			}},
		}); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, true)
		result, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector)
		if err != nil || result.Reviewed != 1 || result.Retired != 1 {
			t.Fatalf("later-target remediation = (%+v, %v)", result, err)
		}
		store, err = overgodb.OpenReadOnly(fixture.storePath)
		if err != nil {
			t.Fatal(err)
		}
		if _, found, resolveErr := artifact.ResolveAlias(t.Context(), store, fixture.firstAlias); resolveErr != nil || found {
			t.Fatalf("later active target remained after remediation = (found=%t, %v)", found, resolveErr)
		}
		record, found, recordErr := artifact.ReadContent(t.Context(), store, result.Record)
		var evidence closureRemediationEvidence
		decodeErr := json.Unmarshal(record.Data, &evidence)
		if recordErr != nil || !found || decodeErr != nil || len(evidence.ReviewedAliases) != 1 {
			t.Fatalf("later-target evidence = (found=%t, read=%v decode=%v body=%+v)", found, recordErr, decodeErr, evidence)
		}
		reviewed := evidence.ReviewedAliases[0]
		if reviewed.ReactivatedDocument != fixture.first.ID || reviewed.CurrentDocument != fixture.successor.ID ||
			reviewed.ActivationSequence != sequence || reviewed.CurrentSequence != selector.ExpectedSequence {
			t.Fatalf("later-target reviewed tuple = %+v", reviewed)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("legacy recovery needs explicit attestation", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closurePublishOperation,
			[]closureledger.Document{fixture.first}, nil, nil,
		); err != nil {
			t.Fatal(err)
		}
		if _, _, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRetireUnmatchedOperation, nil,
			[]artifact.AliasBinding{{
				Name: fixture.firstAlias, Target: fixture.first.ID,
				Previous: artifact.IDPointer(fixture.first.ID), Remove: true,
			}}, nil,
		); err != nil {
			t.Fatal(err)
		}
		commit, sequence := reactivateClosureChain(t, fixture, true)
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, false)
		if _, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector); err == nil ||
			!strings.Contains(err.Error(), "unverified operation provenance") {
			t.Fatalf("unconfirmed legacy remediation error = %v", err)
		}
		assertClosureAliasTarget(t, fixture.storePath, fixture.firstAlias, fixture.first.ID)
		selector.ConfirmUnverifiedRecovery = true
		result, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector)
		if err != nil || result.Reviewed != 1 || result.Retired != 1 {
			t.Fatalf("confirmed legacy remediation = (%+v, %v)", result, err)
		}
	})

	t.Run("proven eligible recovery", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		commit, sequence := reactivateClosureChain(t, fixture, false)
		selector := remediationSelectorAtHead(t, fixture.storePath, commit, sequence, false)
		if _, err := remediateClosureReactivations(t.Context(), fixture.storePath, selector); err == nil ||
			!strings.Contains(err.Error(), "proven eligible") {
			t.Fatalf("eligible remediation error = %v", err)
		}
		assertClosureAliasTarget(t, fixture.storePath, fixture.firstAlias, fixture.first.ID)
	})

	t.Run("missing selector", func(t *testing.T) {
		fixture := setupClosureChainFixture(t)
		head, sequence := storeCoordinates(t, fixture.storePath)
		if _, err := remediateClosureReactivations(t.Context(), fixture.storePath, closureRemediationSelector{}); err == nil {
			t.Fatal("empty remediation selector was accepted")
		}
		if currentHead, currentSequence := storeCoordinates(t, fixture.storePath); currentHead != head || currentSequence != sequence {
			t.Fatalf("empty selector mutated store: %s@%d -> %s@%d", head, sequence, currentHead, currentSequence)
		}
	})
}

func TestClosureRemediationEvidenceAliasesAreReplayableAndSorted(t *testing.T) {
	ids := make([]artifact.ID, 3)
	for index, value := range []string{"retired", "reactivated", "current"} {
		id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(value))
		if err != nil {
			t.Fatal(err)
		}
		ids[index] = id
	}
	selected := []closureRecoveryReactivation{
		{
			Alias: "closure/active/z", RetiredDocument: ids[0], ReactivatedDocument: ids[1], CurrentDocument: ids[2],
			Commit: overgodb.CommitView{Sequence: 7}, RetirementSequence: 3, CurrentSequence: 9,
			Blocker: "z-blocker", Provenance: "z-provenance",
		},
		{
			Alias: "closure/active/a", RetiredDocument: ids[2], ReactivatedDocument: ids[1], CurrentDocument: ids[0],
			Commit: overgodb.CommitView{Sequence: 8}, RetirementSequence: 4, CurrentSequence: 10,
			Blocker: "a-blocker", Provenance: "a-provenance",
		},
	}
	reviewed := closureRemediationReviewedAliases(selected)
	if len(reviewed) != 2 || reviewed[0] != (closureRemediationReviewedAlias{
		Alias: "closure/active/a", RetiredDocument: ids[2], ReactivatedDocument: ids[1], CurrentDocument: ids[0],
		ActivationSequence: 8, RetirementSequence: 4, CurrentSequence: 10,
		Blocker: "a-blocker", Provenance: "a-provenance",
	}) || reviewed[1].Alias != "closure/active/z" {
		t.Fatalf("reviewed aliases = %+v", reviewed)
	}
}

func TestClosureRemediationResultPrettyJSONContract(t *testing.T) {
	record, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("remediation record"))
	if err != nil {
		t.Fatal(err)
	}
	commit := strings.Repeat("a", 64)
	result := closureRemediationResult{Reviewed: 2, Retired: 2, Record: record, Commit: commit}
	var output bytes.Buffer
	if err := clioptions.WritePrettyJSON(&output, result); err != nil {
		t.Fatal(err)
	}
	want := "{\n" +
		"  \"reviewed\": 2,\n" +
		"  \"retired\": 2,\n" +
		"  \"record\": \"" + record.String() + "\",\n" +
		"  \"commit\": \"" + commit + "\"\n" +
		"}\n"
	if output.String() != want {
		t.Fatalf("remediation JSON = %q, want %q", output.String(), want)
	}
}

func TestClosureRemediationPostCommitErrorsCannotReverseSuccess(t *testing.T) {
	operationErr := errors.New("post-commit operation error")
	closeErr := errors.New("post-commit close error")
	if err := closureRemediationFinalError(true, operationErr, closeErr); err != nil {
		t.Fatalf("committed remediation returned error: %v", err)
	}
	if err := closureRemediationFinalError(false, operationErr, closeErr); !errors.Is(err, operationErr) || !errors.Is(err, closeErr) {
		t.Fatalf("pre-commit errors were lost: %v", err)
	}
}

func reactivateClosureChain(t *testing.T, fixture closureChainFixture, legacy bool) (artifact.CommitID, uint64) {
	t.Helper()
	if !legacy {
		_, commit, err := commitClosureDocuments(
			fixture.root, fixture.storePath, closureRebindOperation,
			[]closureledger.Document{fixture.first}, nil, nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		head, sequence := storeCoordinates(t, fixture.storePath)
		if head != commit {
			t.Fatalf("reactivation head = %s, want %s", head, commit)
		}
		return commit, sequence
	}
	store, err := overgodb.Open(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	commit, err := store.Commit(t.Context(), artifact.Batch{
		Key:     "closure-scan/" + strings.Repeat("a", 64),
		Aliases: []artifact.AliasBinding{{Name: fixture.firstAlias, Target: fixture.first.ID}},
	})
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if closeErr := store.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if head != commit {
		t.Fatalf("legacy reactivation head = %s, want %s", head, commit)
	}
	return commit, sequence
}

func retireClosureChainSuccessor(t *testing.T, fixture closureChainFixture) {
	t.Helper()
	if _, _, err := commitClosureDocuments(
		fixture.root, fixture.storePath, closureRetireUnmatchedOperation, nil,
		[]artifact.AliasBinding{{
			Name: fixture.nextAlias, Target: fixture.successor.ID,
			Previous: artifact.IDPointer(fixture.successor.ID), Remove: true,
		}}, nil,
	); err != nil {
		t.Fatal(err)
	}
}

func remediationSelectorAtHead(
	t *testing.T,
	storePath string,
	commit artifact.CommitID,
	sequence uint64,
	confirmLegacy bool,
) closureRemediationSelector {
	t.Helper()
	head, headSequence := storeCoordinates(t, storePath)
	return closureRemediationSelector{
		ReactivationCommit: commit, ReactivationSequence: sequence,
		ExpectedHead: head, ExpectedSequence: headSequence, ExpectedCount: 1,
		ConfirmUnverifiedRecovery: confirmLegacy,
	}
}

func storeCoordinates(t *testing.T, storePath string) (artifact.CommitID, uint64) {
	t.Helper()
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return head, sequence
}

func appendUnrelatedClosureCommit(t *testing.T, storePath string) {
	t.Helper()
	payload := []byte("unrelated remediation head movement")
	id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	_, commitErr := store.Commit(t.Context(), artifact.Batch{
		Key:       "fixture/unrelated-remediation-head",
		Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(payload))}},
	})
	closeErr := store.Close()
	if commitErr != nil || closeErr != nil {
		t.Fatalf("append unrelated commit = (%v, close=%v)", commitErr, closeErr)
	}
}

func assertClosureAliasTarget(t *testing.T, storePath, alias string, want artifact.ID) {
	t.Helper()
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	got, found, resolveErr := artifact.ResolveAlias(t.Context(), store, alias)
	closeErr := store.Close()
	if resolveErr != nil || closeErr != nil || !found || got != want {
		t.Fatalf("alias %s = (%s, found=%t, err=%v, close=%v), want %s", alias, got, found, resolveErr, closeErr, want)
	}
}
