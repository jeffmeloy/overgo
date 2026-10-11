package evaluation

import (
	"maps"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestConfirmationProtocolAndExposureSurviveReplacement holds a pool's
// acceptance to what it fixed and its documents to what they have answered,
// however the acceptance is replaced.
//
// The window is part of the acceptance: a pool that names none is not built,
// one with another window is another acceptance, and an acceptance published
// before the window was part of it confirms nothing, with nothing charged and
// nothing prepared.
//
// A replacement over the same documents, in its gain, its basis, its window,
// its base or the weights of its partition, starts with what those documents
// have already answered spent: none of them obtains a fresh budget. A verdict
// recorded under one acceptance answers no claim under another, so the same
// candidate against another base is read and charged again. A charge stays
// with the grant it was made against. A copy of a charged document, another
// record of the same content, is that document: a part that holds it is
// charged for it. Only a part that shares no document with a charged one, by
// record or by content, starts whole.
//
// The reading, which is where models and documents are prepared, runs once
// for a confirmation the budget pays for and has not read, and not at all for
// one recorded before or one the budget cannot pay for.
func TestConfirmationProtocolAndExposureSurviveReplacement(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, objective, _, _ := poolObjective(t)
	first := HeldoutPoolRequest{
		Objective: objective, Base: testutil.ArtifactID(t, artifact.KindModel, "pool fixture base"), Sequence: 8,
		Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "pool fixture verdict")}, Weights: [poolPartCount]uint64{1, 2, 4, 4},
	}
	unwindowed := first
	unwindowed.Sequence, unwindowed.Gain = 0, 1
	if _, err := BuildHeldoutPool(ctx, store, unwindowed); err == nil || !strings.Contains(err.Error(), "read as far as") {
		t.Fatalf("a pool that fixes no window = %v, want it refused", err)
	}

	// prepared counts the readings, and so what was prepared for them.
	prepared := 0
	reading := func(documents []string) ([]float64, error) {
		prepared++
		differences := make([]float64, len(documents))
		for index := range differences {
			differences[index] = 1 + float64(index%2)
		}
		return differences, nil
	}
	candidate := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, "candidate "+name) }
	// confirm asks one acceptance for one candidate and holds what the reading ran and what is left.
	confirm := func(step string, acceptance PoolAcceptance, name string, readings int, left uint64) PoolConfirmation {
		t.Helper()
		before := prepared
		confirmation, remaining, err := ConfirmOnPool(ctx, store, acceptance.ID, candidate(name), reading)
		if err != nil || prepared-before != readings || remaining != left || confirmation.Acceptance != acceptance.ID {
			t.Fatalf("%s: %d readings, %d left, under %s (%v); want %d readings and %d left under %s", step, prepared-before, remaining, confirmation.Acceptance, err, readings, left, acceptance.ID)
		}
		return confirmation
	}
	// refused asks for a confirmation the documents cannot pay for: nothing is read and nothing is charged.
	refused := func(step string, pool HeldoutPool, name string) {
		t.Helper()
		before := prepared
		granted, err := runrecord.NewBudget(poolUnit, pool.Partition.Promotion, pool.Acceptance.Confirmations, pool.Acceptance.ID)
		if err != nil {
			t.Fatal(err)
		}
		_, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID)
		if err != nil {
			t.Fatal(err)
		}
		_, _, err = ConfirmOnPool(ctx, store, pool.Acceptance.ID, candidate(name), reading)
		_, after, chargesErr := runrecord.BudgetCharges(ctx, store, granted.ID)
		if err == nil || !strings.Contains(err.Error(), "fresh partition") || prepared != before || chargesErr != nil || len(after) != len(charges) {
			t.Fatalf("%s = %v with %d readings and %d charges added (%v), want it refused with nothing read or charged", step, err, prepared-before, len(after)-len(charges), chargesErr)
		}
	}

	original := publishedPool(t, store, first, 2)
	kept := confirm("the first confirmation of a pool no grant has charged", original.Acceptance, "one", 1, 1)

	// The same documents under a gain that sizes the part for one more confirmation.
	regained := publishedPool(t, store, first, 3)
	if regained.Partition.Promotion != original.Partition.Promotion || regained.Acceptance.ID == original.Acceptance.ID {
		t.Fatalf("the pool rebuilt for another gain holds part %s under acceptance %s, want the first pool's part under another acceptance", regained.Partition.Promotion, regained.Acceptance.ID)
	}
	other := confirm("the same candidate under the replaced gain", regained.Acceptance, "one", 1, 1)
	if other.ID == kept.ID || other.Charge == kept.Charge {
		t.Fatalf("a verdict recorded under one acceptance answered a claim under another: %s", other.ID)
	}
	if again := confirm("the first confirmation asked again", original.Acceptance, "one", 0, 0); again.ID != kept.ID {
		t.Fatalf("the first confirmation asked again = %s, want the recorded %s", again.ID, kept.ID)
	}

	// The same documents and gain as the first, the gain stated from other verdicts.
	rebased := first
	rebased.Basis = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "another pool fixture verdict")}
	refused("a replacement that changes only the basis, on documents that have answered as often as it allows", publishedPool(t, store, rebased, 2), "two")

	// Another window is another acceptance of the same documents.
	windowed := first
	windowed.Sequence = 16
	rewindowed := publishedPool(t, store, windowed, 3)
	if rewindowed.Acceptance.ID == regained.Acceptance.ID || rewindowed.Acceptance.Sequence != 16 {
		t.Fatalf("a pool with another window is acceptance %s reading %d tokens, want another acceptance reading 16", rewindowed.Acceptance.ID, rewindowed.Acceptance.Sequence)
	}
	confirm("a candidate under the replaced window", rewindowed.Acceptance, "two", 1, 0)
	refused("a further candidate under the replaced window", rewindowed, "three")

	// Another base is another comparison: the candidate the first pool confirmed is read and charged again.
	based := first
	based.Base = testutil.ArtifactID(t, artifact.KindModel, "another pool fixture base")
	confirm("the first candidate against another base", publishedPool(t, store, based, 4).Acceptance, "one", 1, 0)

	// Other weights divide the documents otherwise. A promotion part that
	// holds a document the charged part holds is charged for it; one that
	// holds none starts whole.
	weighed := first
	weighed.Weights = [poolPartCount]uint64{4, 2, 3, 2}
	repartitioned := publishedPool(t, store, weighed, 6)
	charged, err := partDocuments(ctx, store, original.Partition.Promotion)
	if err != nil {
		t.Fatal(err)
	}
	moved, err := partDocuments(ctx, store, repartitioned.Partition.Promotion)
	if err != nil {
		t.Fatal(err)
	}
	const spent = 4
	left := uint64(6 - 1)
	if slices.ContainsFunc(moved, func(document string) bool { return slices.Contains(charged, document) }) {
		left -= spent
	} else if repartitioned.Partition.Promotion == original.Partition.Promotion {
		t.Fatal("other weights left the promotion part what it was")
	}
	confirm("a candidate on the documents divided by other weights", repartitioned.Acceptance, "four", 1, left)

	// A copy: another record of the dataset with the content of a document
	// the charged part holds. A part that held it is charged what that content
	// has answered, by content alone; a part of documents no charged part
	// holds, by record or by content, starts whole.
	contents, err := recordContents(ctx, store, objective.Dataset)
	if err != nil {
		t.Fatal(err)
	}
	holds := func(part []string, record string) bool {
		return slices.ContainsFunc(part, func(held string) bool { return contents[held] == contents[record] })
	}
	var copied, fresh string
	for _, record := range slices.Sorted(maps.Keys(contents)) {
		switch {
		case copied == "" && holds(charged, record) && !slices.Contains(charged, record):
			copied = record
		case fresh == "" && !holds(charged, record) && !holds(moved, record):
			fresh = record
		}
	}
	if copied == "" || fresh == "" {
		t.Fatalf("the fixture has copy %q of a charged document and fresh document %q, and the test needs both", copied, fresh)
	}
	owed := uint64(spent)
	if holds(moved, copied) {
		owed++
	}
	if answered, err := exposure(ctx, store, original.Partition, artifact.ID{}, []string{copied}); err != nil || answered != owed {
		t.Fatalf("a part holding a copy of a charged document has answered %d confirmations (%v), want the %d its content has", answered, err, owed)
	}
	if answered, err := exposure(ctx, store, original.Partition, artifact.ID{}, []string{fresh}); err != nil || answered != 0 {
		t.Fatalf("a part of a document no charged part holds has answered %d confirmations (%v), want none", answered, err)
	}

	// Each charge is still the grant's it was made against.
	for name, pool := range map[string]HeldoutPool{"first": original, "regained": regained, "rewindowed": rewindowed} {
		granted, err := runrecord.NewBudget(poolUnit, pool.Partition.Promotion, pool.Acceptance.Confirmations, pool.Acceptance.ID)
		if err != nil {
			t.Fatal(err)
		}
		if _, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID); err != nil || len(charges) != 1 {
			t.Fatalf("the %s pool's grant holds %d charges (%v), want the one made against it", name, len(charges), err)
		}
	}

	// An acceptance as one was published before the window was part of it.
	historical := original.Acceptance
	historical.Sequence, historical.ID = 0, artifact.ID{}
	if historical, err = poolAcceptanceCodec.NewInitial(historical); err != nil {
		t.Fatal(err)
	}
	batch, err := poolAcceptanceCodec.Batch("pool-fixture/historical", historical, artifact.DependencyLineage(historical.ID, original.Partition.ID, objective.ID), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if read, err := RequireHeldoutPool(ctx, store, historical.ID); err != nil || read.Acceptance.Sequence != 0 {
		t.Fatalf("an acceptance without a window reads back %+v (%v), want it read as it was published", read.Acceptance, err)
	}
	before := prepared
	if _, _, err := ConfirmOnPool(ctx, store, historical.ID, candidate("six"), reading); err == nil || !strings.Contains(err.Error(), "fixed no tokens") || prepared != before {
		t.Fatalf("a confirmation on an acceptance without a window = %v with %d readings, want it refused before anything is prepared", err, prepared-before)
	}
}
