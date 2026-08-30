package main

import (
	"context"
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

const emptyRestoreAuthorityDigest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

type closureAliasRestoreCorruption uint8

const (
	restoreAliasesValid closureAliasRestoreCorruption = iota
	restoreAliasTargetsNonClosure
	restoreAliasTargetsMismatchedClosure
	restoreSourceAliasesEmpty
)

type closureAliasRestoreFixture struct {
	root, storePath         string
	snapshot                repoanalysis.SourceSnapshot
	sourceHead, currentHead artifact.CommitID
	sourceSequence          uint64
	currentSequence         uint64
	one, two                artifact.Content
	oneAlias, twoAlias      string
}

func TestReviewedClosureAliasRestoreDryRunAndCommit(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
	selector := fixture.selector()

	dryRun, err := restoreClosureAliasesAtReviewedHead(
		t.Context(), fixture.storePath, fixture.snapshot, selector,
	)
	if err != nil {
		t.Fatal(err)
	}
	if dryRun.Confirmed || dryRun.DesiredAliases != 2 || dryRun.CurrentAliases != 2 ||
		dryRun.Changes != 3 || dryRun.PredictedStale != 0 || dryRun.AuthorityDigest != emptyRestoreAuthorityDigest ||
		dryRun.Commit != "" {
		t.Fatalf("alias restore dry run = %+v", dryRun)
	}
	if head, sequence := storeCoordinates(t, fixture.storePath); head != fixture.currentHead || sequence != fixture.currentSequence {
		t.Fatalf("dry run moved store to %s@%d", head, sequence)
	}

	selector.Confirm = true
	selector.ExpectedAliases = dryRun.DesiredAliases
	selector.ExpectedChanges = dryRun.Changes
	selector.ExpectedStale = dryRun.PredictedStale
	selector.ExpectedAuthorityDigest = dryRun.AuthorityDigest
	committed, err := restoreClosureAliasesAtReviewedHead(
		t.Context(), fixture.storePath, fixture.snapshot, selector,
	)
	if err != nil || !committed.Confirmed || committed.Commit == "" {
		t.Fatalf("confirmed alias restore = (%+v, %v)", committed, err)
	}
	store, err := overgodb.OpenReadOnly(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	assertAlias := func(name string, want artifact.ID, foundWant bool) {
		t.Helper()
		got, found, err := artifact.ResolveAlias(t.Context(), store, name)
		if err != nil || found != foundWant || found && got != want {
			t.Fatalf("alias %s = (%s, found=%t, %v), want (%s, %t)", name, got, found, err, want, foundWant)
		}
	}
	assertAlias(fixture.oneAlias, fixture.one.Descriptor.ID, true)
	assertAlias(fixture.twoAlias, fixture.two.Descriptor.ID, true)
	assertAlias(closureledger.ActiveAliasPrefix+"c", artifact.ID{}, false)
	head, sequence := store.Head()
	if head.String() != committed.Commit || sequence != fixture.currentSequence+1 {
		t.Fatalf("restored head = %s@%d, want %s@%d", head, sequence, committed.Commit, fixture.currentSequence+1)
	}
	record, found, err := artifact.ReadContent(t.Context(), store, committed.Record)
	if err != nil || !found || record.Descriptor.Schema != closureAliasRestoreSchema {
		t.Fatalf("restoration review record = (%+v, found=%t, %v)", record.Descriptor, found, err)
	}
}

func TestReviewedClosureAliasRestoreRefusesReviewMismatch(t *testing.T) {
	for _, test := range []struct {
		name       string
		corruption closureAliasRestoreCorruption
		changes    int
		stale      int
		wantError  string
	}{
		{"changed review", restoreAliasesValid, 4, 0, "review changed"},
		{"nonzero reviewed stale count", restoreAliasesValid, 3, 1, "zero stale bindings"},
		{"mismatched closure binding", restoreAliasTargetsMismatchedClosure, 1, 0, "has 0 canonical bindings for that alias"},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newClosureAliasRestoreFixture(t, test.corruption)
			selector := fixture.selector()
			selector.Confirm = true
			selector.ExpectedAliases = 2
			selector.ExpectedChanges = test.changes
			selector.ExpectedStale = test.stale
			selector.ExpectedAuthorityDigest = emptyRestoreAuthorityDigest
			_, err := restoreClosureAliasesAtReviewedHead(t.Context(), fixture.storePath, fixture.snapshot, selector)
			if err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("%s error = %v", test.name, err)
			}
			if head, sequence := storeCoordinates(t, fixture.storePath); head != fixture.currentHead || sequence != fixture.currentSequence {
				t.Fatalf("%s moved store to %s@%d", test.name, head, sequence)
			}
		})
	}
}

func TestReviewedClosureAliasRestoreRefusesWrongSourceCoordinate(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
	selector := fixture.selector()
	selector.SourceHead = fixture.currentHead
	_, err := restoreClosureAliasesAtReviewedHead(t.Context(), fixture.storePath, fixture.snapshot, selector)
	if err == nil || !strings.Contains(err.Error(), "source coordinate") {
		t.Fatalf("wrong source error = %v", err)
	}
}

func TestReviewedClosureAliasRestoreRefusesNonClosureTarget(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreAliasTargetsNonClosure)
	_, err := restoreClosureAliasesAtReviewedHead(
		t.Context(), fixture.storePath, fixture.snapshot, fixture.selector(),
	)
	if err == nil || !strings.Contains(err.Error(), "target is not a closure document") {
		t.Fatalf("non-closure target error = %v", err)
	}
	if head, sequence := storeCoordinates(t, fixture.storePath); head != fixture.currentHead || sequence != fixture.currentSequence {
		t.Fatalf("non-closure target moved store to %s@%d", head, sequence)
	}
}

func TestReviewedClosureAliasRestorePurgesToEmptyReviewedSource(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreSourceAliasesEmpty)
	selector := fixture.selector()
	selector.Confirm = true
	selector.ExpectedAliases = 0
	selector.ExpectedChanges = 2
	selector.ExpectedStale = 0
	selector.ExpectedAuthorityDigest = emptyRestoreAuthorityDigest
	result, err := restoreClosureAliasesAtReviewedHead(
		t.Context(), fixture.storePath, fixture.snapshot, selector,
	)
	if err != nil || result.Commit == "" || !result.Record.Valid() || result.DesiredAliases != 0 || result.Changes != 2 {
		t.Fatalf("empty-source restore = (%+v, %v)", result, err)
	}
	store, err := overgodb.OpenReadOnly(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	for _, alias := range []string{fixture.oneAlias, fixture.twoAlias} {
		if _, found, resolveErr := artifact.ResolveAlias(t.Context(), store, alias); resolveErr != nil || found {
			t.Fatalf("purged alias %s = (found=%t, %v)", alias, found, resolveErr)
		}
	}
	if closeErr := store.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
}

func TestReviewedClosureAliasRestoreRefusesSourceOnlyAuthorityChange(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
	dryRun, err := restoreClosureAliasesAtReviewedHead(
		t.Context(), fixture.storePath, fixture.snapshot, fixture.selector(),
	)
	if err != nil || dryRun.AuthorityError != "" || dryRun.PredictedStale != 0 {
		t.Fatalf("clean dry run = (%+v, %v)", dryRun, err)
	}
	changed := addUnreviewedRestorePolicy(t, fixture.root)
	selector := fixture.selector()
	selector.Confirm = true
	selector.ExpectedAliases = dryRun.DesiredAliases
	selector.ExpectedChanges = dryRun.Changes
	selector.ExpectedStale = 0
	selector.ExpectedAuthorityDigest = dryRun.AuthorityDigest
	_, err = restoreClosureAliasesAtReviewedHead(t.Context(), fixture.storePath, changed, selector)
	if err == nil || !strings.Contains(err.Error(), "authority digest changed") {
		t.Fatalf("source-only authority change error = %v", err)
	}
	if head, sequence := storeCoordinates(t, fixture.storePath); head != fixture.currentHead || sequence != fixture.currentSequence {
		t.Fatalf("source-only authority change moved store to %s@%d", head, sequence)
	}
}

func TestReviewedClosureAliasRestoreConfirmsReviewedNonemptyAuthorityResult(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
	changed := addUnreviewedRestorePolicy(t, fixture.root)
	dryRun, err := restoreClosureAliasesAtReviewedHead(
		t.Context(), fixture.storePath, changed, fixture.selector(),
	)
	if err != nil || dryRun.AuthorityError == "" || !validClosureAuthorityDigest(dryRun.AuthorityDigest) {
		t.Fatalf("nonempty-authority dry run = (%+v, %v)", dryRun, err)
	}
	selector := fixture.selector()
	selector.Confirm = true
	selector.ExpectedAliases = dryRun.DesiredAliases
	selector.ExpectedChanges = dryRun.Changes
	selector.ExpectedStale = 0
	selector.ExpectedAuthorityDigest = dryRun.AuthorityDigest
	result, err := restoreClosureAliasesAtReviewedHead(t.Context(), fixture.storePath, changed, selector)
	if err != nil || result.Commit == "" || !result.Record.Valid() || result.AuthorityDigest != dryRun.AuthorityDigest {
		t.Fatalf("reviewed nonempty-authority restore = (%+v, %v)", result, err)
	}
	store, err := overgodb.OpenReadOnly(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	content, found, readErr := artifact.ReadContent(t.Context(), store, result.Record)
	closeErr := store.Close()
	if readErr != nil || closeErr != nil || !found {
		t.Fatalf("read authority-bound restore record = (found=%t, read=%v, close=%v)", found, readErr, closeErr)
	}
	var record closureAliasRestoreRecord
	if err := json.Unmarshal(content.Data, &record); err != nil || record.AuthorityDigest != dryRun.AuthorityDigest {
		t.Fatalf("authority-bound restore record = (%+v, %v)", record, err)
	}
}

func TestReviewedClosureAliasRestoreRejectsInvalidAuthorityDigest(t *testing.T) {
	fixture := newClosureAliasRestoreFixture(t, restoreAliasesValid)
	valid := emptyRestoreAuthorityDigest
	for _, test := range []struct {
		name, digest string
	}{
		{name: "absent"},
		{name: "short", digest: valid[:len(valid)-1]},
		{name: "non-hex", digest: strings.Repeat("g", len(valid))},
		{name: "uppercase", digest: strings.ToUpper(valid)},
	} {
		t.Run(test.name, func(t *testing.T) {
			selector := fixture.selector()
			selector.Confirm = true
			selector.ExpectedAliases = 2
			selector.ExpectedChanges = 3
			selector.ExpectedStale = 0
			selector.ExpectedAuthorityDigest = test.digest
			if err := selector.validate(); err == nil || !strings.Contains(err.Error(), "authority digest") {
				t.Fatalf("invalid authority digest error = %v", err)
			}
		})
	}
}

func addUnreviewedRestorePolicy(t *testing.T, root string) repoanalysis.SourceSnapshot {
	t.Helper()
	path := filepath.Join(root, "internal", "unreviewed", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package unreviewed\nconst NewPolicy = 11\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func newClosureAliasRestoreFixture(t *testing.T, corruption closureAliasRestoreCorruption) closureAliasRestoreFixture {
	t.Helper()
	root := t.TempDir()
	path := filepath.Join(root, "internal", "p", "p.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package p\nconst (\n\tA = 7\n\tB = 9\n)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 2 {
		t.Fatalf("closure candidates = (%d, %v)", len(candidates), err)
	}
	documents := make([]closureledger.Document, len(candidates))
	aliases := make([]string, len(candidates))
	contents := make([]artifact.Content, len(candidates))
	for index, candidate := range candidates {
		binding, err := candidate.Binding()
		if err != nil {
			t.Fatal(err)
		}
		documents[index], err = closureledger.New(
			candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed,
			"Reviewed restore fixture policy.", []closureledger.SourceBinding{binding},
			"Replace with fixture authority.", "Fixture policy changes.", binding.Owner,
		)
		if err != nil {
			t.Fatal(err)
		}
		aliases[index], err = closureledger.ActiveAlias(binding)
		if err != nil {
			t.Fatal(err)
		}
		contents[index], err = documents[index].Content()
		if err != nil {
			t.Fatal(err)
		}
	}
	contract := artifact.JSONContract(artifact.KindEvidence, "fixture/alias-restore/v1")
	nonClosure, err := artifact.JSONContent(contract, map[string]string{"value": "not-a-closure"})
	if err != nil {
		t.Fatal(err)
	}
	one, two := contents[0], contents[1]
	oneTarget := one.Descriptor.ID
	switch corruption {
	case restoreAliasTargetsNonClosure:
		oneTarget = nonClosure.Descriptor.ID
	case restoreAliasTargetsMismatchedClosure:
		oneTarget = two.Descriptor.ID
	}
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	sourceAliases := []artifact.AliasBinding{
		{Name: aliases[0], Target: oneTarget},
		{Name: aliases[1], Target: two.Descriptor.ID},
	}
	if corruption == restoreSourceAliasesEmpty {
		sourceAliases = nil
	}
	sourceHead, err := store.Commit(context.Background(), artifact.Batch{
		Key:      "fixture/alias-restore/source",
		Contents: []artifact.Content{one, two, nonClosure},
		Aliases:  sourceAliases,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, sourceSequence := store.Head()
	currentAliases := []artifact.AliasBinding{{Name: closureledger.ActiveAliasPrefix + "c", Target: one.Descriptor.ID}}
	switch corruption {
	case restoreAliasesValid:
		currentAliases = append(currentAliases,
			artifact.AliasBinding{
				Name: aliases[0], Target: two.Descriptor.ID,
				Previous: artifact.IDPointer(one.Descriptor.ID),
			},
			artifact.AliasBinding{
				Name: aliases[1], Target: two.Descriptor.ID,
				Previous: artifact.IDPointer(two.Descriptor.ID), Remove: true,
			},
		)
	case restoreSourceAliasesEmpty:
		currentAliases = []artifact.AliasBinding{
			{Name: aliases[0], Target: one.Descriptor.ID},
			{Name: aliases[1], Target: two.Descriptor.ID},
		}
	}
	currentHead, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/alias-restore/current", Aliases: currentAliases,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, currentSequence := store.Head()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if corruption == restoreSourceAliasesEmpty {
		if err := os.WriteFile(path, []byte("package p\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		snapshot, err = repoanalysis.DiscoverGo(root, "internal", "cmd")
		if err != nil {
			t.Fatal(err)
		}
	}
	return closureAliasRestoreFixture{
		root: root, storePath: storePath, snapshot: snapshot,
		sourceHead: sourceHead, currentHead: currentHead,
		sourceSequence: sourceSequence, currentSequence: currentSequence,
		one: one, two: two, oneAlias: aliases[0], twoAlias: aliases[1],
	}
}

func (fixture closureAliasRestoreFixture) selector() closureAliasRestoreSelector {
	return closureAliasRestoreSelector{
		SourceHead: fixture.sourceHead, SourceSequence: fixture.sourceSequence,
		ExpectedHead: fixture.currentHead, ExpectedSequence: fixture.currentSequence,
	}
}
