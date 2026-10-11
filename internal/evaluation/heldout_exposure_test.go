package evaluation

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// chargedBetween is a store in which another confirmation is charged between
// one confirmation's reading of what is left and its own charge: the first
// charge committed through it runs before, once, ahead of itself.
type chargedBetween struct {
	artifact.Repository
	before func()
}

func (store *chargedBetween) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if run := store.before; run != nil && strings.Contains(batch.Key, "/confirmation/") {
		store.before = nil
		run()
	}
	return store.Repository.Commit(ctx, batch)
}

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
// The same bytes registered again as another dataset are the same documents:
// a pool over that dataset starts with what their contents have answered
// spent, though no record and no partition of it is the first dataset's. And
// two parts that share documents are charged one at a time: when a second
// confirmation is charged between one's reading of what is left and its own
// charge, the store refuses the one that comes second, which reads nothing,
// and asked again it is charged as the next on those documents. That holds
// when the two parts are of two datasets the same bytes are registered under,
// which share no record and no partition: their charges are numbered
// together, so the second is refused there too, is charged as the next when
// asked again, returns its verdict unread when asked a third time, and leaves
// neither part a further confirmation.
//
// A part published without the names of what it holds, as parts were before
// contents were named, cannot be found from another registration of its
// documents: it is refused a confirmation, with nothing charged or read.
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
	if answered, _, _, err := exposure(ctx, store, original.Partition, artifact.ID{}, dataset.Membership{Records: []dataset.Record{{ID: copied, Group: copied}}}); err != nil || answered != owed {
		t.Fatalf("a part holding a copy of a charged document has answered %d confirmations (%v), want the %d its content has", answered, err, owed)
	}
	answered, all, scope, err := exposure(ctx, store, original.Partition, artifact.ID{}, dataset.Membership{Records: []dataset.Record{{ID: fresh, Group: fresh}}})
	if err != nil || answered != 0 || all != spent+1 || scope != objective.Dataset {
		t.Fatalf("a part of a document no charged part holds has answered %d confirmations of the %d numbered under %s (%v), want none of the dataset's own %d", answered, all, scope, err, spent+1)
	}

	// The same bytes registered again as another dataset, with a document more:
	// a pool over it starts with what the charged contents have answered spent.
	// And two parts that share documents, charged at once: one is charged, and
	// the other is refused by the store and reads nothing.
	again, _, _ := poolObjectiveOver(t, store, "pool-corpus-again", poolFixtureDocuments+1)
	if again.Dataset == objective.Dataset {
		t.Fatal("the corpus registered with a document more is the same dataset")
	}
	rebound := first
	rebound.Objective = again
	refused("a pool over the same bytes registered as another dataset", publishedPool(t, store, rebound, 2), "seven")

	raced, contested, _, _ := poolObjective(t)
	one, two := first, first
	one.Objective, two.Objective, two.Weights = contested, contested, [poolPartCount]uint64{4, 2, 3, 2}
	lost, won := publishedPool(t, raced, one, 2), publishedPool(t, raced, two, 6)
	shared, err := partDocuments(ctx, raced, won.Partition.Promotion)
	if err != nil {
		t.Fatal(err)
	}
	if mine, err := partDocuments(ctx, raced, lost.Partition.Promotion); err != nil || lost.Partition.Promotion == won.Partition.Promotion || !slices.ContainsFunc(mine, func(document string) bool { return slices.Contains(shared, document) }) {
		t.Fatalf("the two pools' promotion parts are one part or share no document (%v); the test needs two that overlap", err)
	}
	prepared = 0
	racing := &chargedBetween{Repository: raced}
	racing.before = func() {
		if _, _, err := ConfirmOnPool(ctx, raced, won.Acceptance.ID, candidate("eight"), reading); err != nil {
			t.Errorf("the confirmation that was charged first: %v", err)
		}
	}
	if _, _, err := ConfirmOnPool(ctx, racing, lost.Acceptance.ID, candidate("eight"), reading); err == nil || !strings.Contains(err.Error(), "was not charged") || prepared != 1 {
		t.Fatalf("of two confirmations on parts that share documents, charged at once, the second = %v with %d readings in all; want it refused uncharged and one reading", err, prepared)
	}
	if _, remaining, err := ConfirmOnPool(ctx, raced, lost.Acceptance.ID, candidate("eight"), reading); err != nil || remaining != 0 || prepared != 2 {
		t.Fatalf("asked again, the confirmation that lost = %d left after %d readings (%v); want it charged as the second on its documents, which leaves none of its two", remaining, prepared, err)
	}
	if _, _, err := ConfirmOnPool(ctx, raced, lost.Acceptance.ID, candidate("nine"), reading); err == nil || !strings.Contains(err.Error(), "fresh partition") || prepared != 2 {
		t.Fatalf("a further confirmation on documents that have answered twice = %v with %d readings; want it refused", err, prepared)
	}

	// The two together: the same bytes under two datasets, a part of each
	// charged at once. No record and no partition is shared, so only the
	// contents say the parts are one material; the charges are numbered across
	// both datasets, and the store refuses the one that comes second.
	twice, under, _, _ := poolObjective(t)
	beside, _, _ := poolObjectiveOver(t, twice, "pool-corpus-again", poolFixtureDocuments+1)
	here, there := first, first
	here.Objective, there.Objective = under, beside
	behind, ahead := publishedPool(t, twice, here, 2), publishedPool(t, twice, there, 2)
	// contentsOf are the contents a pool's promotion part holds.
	contentsOf := func(pool HeldoutPool) map[string]bool {
		t.Helper()
		named, err := recordContents(ctx, twice, pool.Partition.Dataset)
		if err != nil {
			t.Fatal(err)
		}
		documents, err := partDocuments(ctx, twice, pool.Partition.Promotion)
		if err != nil {
			t.Fatal(err)
		}
		held := map[string]bool{}
		for _, document := range documents {
			held[named[document]] = true
		}
		return held
	}
	mine, theirs := contentsOf(behind), contentsOf(ahead)
	if behind.Partition.Dataset == ahead.Partition.Dataset || !slices.ContainsFunc(slices.Collect(maps.Keys(mine)), func(content string) bool { return theirs[content] }) {
		t.Fatal("the two registrations are one dataset, or their promotion parts share no content; the test needs two datasets whose parts do")
	}
	prepared = 0
	crossing := &chargedBetween{Repository: twice}
	crossing.before = func() {
		if _, _, err := ConfirmOnPool(ctx, twice, ahead.Acceptance.ID, candidate("ten"), reading); err != nil {
			t.Errorf("the confirmation under the other dataset that was charged first: %v", err)
		}
	}
	if _, _, err := ConfirmOnPool(ctx, crossing, behind.Acceptance.ID, candidate("ten"), reading); err == nil || !strings.Contains(err.Error(), "was not charged") || prepared != 1 {
		t.Fatalf("of two confirmations on parts of two datasets that share contents, charged at once, the second = %v with %d readings in all; want it refused uncharged and one reading", err, prepared)
	}
	late, remaining, err := ConfirmOnPool(ctx, twice, behind.Acceptance.ID, candidate("ten"), reading)
	if err != nil || remaining != 0 || prepared != 2 {
		t.Fatalf("asked again, the confirmation that lost across datasets = %d left after %d readings (%v); want it charged as the second on those contents, which leaves none of its two", remaining, prepared, err)
	}
	if retried, _, err := ConfirmOnPool(ctx, twice, behind.Acceptance.ID, candidate("ten"), reading); err != nil || retried.ID != late.ID || prepared != 2 {
		t.Fatalf("asked a third time it is %s after %d readings (%v), want the recorded %s unread", retried.ID, prepared, err, late.ID)
	}
	for name, pool := range map[string]HeldoutPool{"the first dataset's": behind, "the second dataset's": ahead} {
		if _, _, err := ConfirmOnPool(ctx, twice, pool.Acceptance.ID, candidate("eleven"), reading); err == nil || !strings.Contains(err.Error(), "fresh partition") || prepared != 2 {
			t.Fatalf("a further confirmation on %s part, whose contents have answered twice = %v with %d readings; want it refused", name, err, prepared)
		}
	}

	// A part published without the names of what it holds, as parts were before
	// contents were named, takes no confirmation: nothing is charged or read.
	bare, unnamed, _, _ := poolObjective(t)
	old := first
	old.Objective = unnamed
	nameless := sizedPool(t, bare, old, 2)
	nameless.Batch.Lineage = slices.DeleteFunc(nameless.Batch.Lineage, func(edge artifact.Lineage) bool { return edge.Relation == artifact.RelationContains })
	if _, err := artifact.Publish(ctx, bare, nameless.Batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ConfirmOnPool(ctx, bare, nameless.Acceptance.ID, candidate("twelve"), reading); err == nil || !strings.Contains(err.Error(), "published without the names") || prepared != 2 {
		t.Fatalf("a confirmation on a part published without its contents' names = %v with %d readings; want it refused unread", err, prepared)
	}
	if granted, err := runrecord.NewBudget(poolUnit, nameless.Partition.Promotion, nameless.Acceptance.Confirmations, nameless.Acceptance.ID); err != nil {
		t.Fatal(err)
	} else if _, charges, err := runrecord.BudgetCharges(ctx, bare, granted.ID); err != nil || len(charges) != 0 {
		t.Fatalf("the refused confirmation left %d charges (%v), want none", len(charges), err)
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
