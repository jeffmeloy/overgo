package evaluation

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// overtaken is a store in which something is committed between one
// confirmation's reading of its own grant's charges and its reading of what
// else its documents have answered: when the grant's dependents are listed
// through it, between runs after the listing is taken, each time it is set.
// skip is how many listings pass first: an instruction's comparison lists its
// grant once before the listing its balance is counted from.
type overtaken struct {
	artifact.Repository
	grant   artifact.ID
	skip    int
	between func()
}

func (store *overtaken) Children(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	edges, err := store.Repository.Children(ctx, id)
	if run := store.between; run != nil && id == store.grant {
		if store.skip > 0 {
			store.skip--
		} else {
			run()
		}
	}
	return edges, err
}

// TestConfirmationCountsAChargeThatLandsBetweenItsReadings holds a
// confirmation's readings and its charge to one decision: what it charges
// against is the store it read, and a store that has moved by then is read
// again before anything is charged.
//
// With one confirmation left on a part, a second candidate is charged after
// a first has read its grant's charges and before it reads what else its
// documents have answered. The first counted one left and would have charged
// past the grant; it is refused instead, with nothing read, and the grant
// holds the charges it issued and no more. With a confirmation left for each,
// the one overtaken is charged as the next in the same asking and reads once,
// and asked again it returns its recorded verdict unread. And under a store
// that is committed to during every reading, a confirmation gives up after
// its bounded askings with nothing charged and nothing read.
func TestConfirmationCountsAChargeThatLandsBetweenItsReadings(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
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
	// pooled is a published pool of so many confirmations in a store of its own, with its grant and what counts the grant's charges.
	pooled := func(confirmations uint64) (artifact.Repository, HeldoutPool, artifact.ID, func() int) {
		t.Helper()
		store, objective, _, _ := poolObjective(t)
		pool := publishedPool(t, store, HeldoutPoolRequest{
			Objective: objective, Base: testutil.ArtifactID(t, artifact.KindModel, "pool fixture base"), Sequence: 8,
			Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "pool fixture verdict")}, Weights: [poolPartCount]uint64{1, 2, 4, 4},
		}, confirmations)
		granted, err := runrecord.NewBudget(poolUnit, pool.Partition.Promotion, pool.Acceptance.Confirmations, pool.Acceptance.ID)
		if err != nil {
			t.Fatal(err)
		}
		return store, pool, granted.ID, func() int {
			t.Helper()
			budget, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID)
			if err == nil {
				_, err = runrecord.BudgetBalance(budget, charges)
			}
			if err != nil {
				t.Fatalf("the grant's charges do not balance: %v", err)
			}
			return len(charges)
		}
	}
	// charged asks a store for a candidate's confirmation and wants it charged and read once, with so many left.
	charged := func(step string, store artifact.Repository, pool HeldoutPool, name string, left uint64) PoolConfirmation {
		t.Helper()
		before := prepared
		confirmation, remaining, err := ConfirmOnPool(ctx, store, pool.Acceptance.ID, candidate(name), reading)
		if err != nil || remaining != left || prepared != before+1 {
			t.Fatalf("%s = %d left after %d readings (%v), want it charged and read once with %d left", step, remaining, prepared-before, err, left)
		}
		return confirmation
	}

	// One confirmation left, and a second candidate charged between the first's readings.
	store, pool, grant, charges := pooled(2)
	charged("the first confirmation of two", store, pool, "one", 1)
	skewed := &overtaken{Repository: store, grant: grant}
	skewed.between = func() {
		skewed.between = nil
		charged("the candidate charged between the other's readings", store, pool, "two", 0)
	}
	before := prepared
	if _, _, err := ConfirmOnPool(ctx, skewed, pool.Acceptance.ID, candidate("three"), reading); err == nil || !strings.Contains(err.Error(), "fresh partition") || prepared != before+1 || charges() != 2 {
		t.Fatalf("with one confirmation left and another candidate charged between its readings, the confirmation = %v, with %d readings in all and the grant holding %d charges; want it refused unread, the other's one reading, and the grant's 2 charges", err, prepared-before, charges())
	}

	// A confirmation left for each: the one overtaken is charged as the next, in the same asking.
	store, pool, grant, charges = pooled(3)
	charged("the first confirmation of three", store, pool, "four", 2)
	skewed = &overtaken{Repository: store, grant: grant}
	skewed.between = func() {
		skewed.between = nil
		charged("the candidate charged between the other's readings", store, pool, "five", 1)
	}
	before = prepared
	late, remaining, err := ConfirmOnPool(ctx, skewed, pool.Acceptance.ID, candidate("six"), reading)
	if err != nil || remaining != 0 || prepared != before+2 || charges() != 3 {
		t.Fatalf("the confirmation overtaken with one left for it = %d left (%v), with %d readings in all and the grant holding %d charges; want it charged as the third, its one reading beside the other's, and none left", remaining, err, prepared-before, charges())
	}
	if again, _, err := ConfirmOnPool(ctx, store, pool.Acceptance.ID, candidate("six"), reading); err != nil || again.ID != late.ID || prepared != before+2 || charges() != 3 {
		t.Fatalf("asked again it is %s after %d readings with the grant holding %d charges (%v), want the recorded %s unread and 3 charges", again.ID, prepared-before, charges(), err, late.ID)
	}

	// A store committed to during every reading.
	store, pool, grant, charges = pooled(2)
	moves := 0
	restless := &overtaken{Repository: store, grant: grant}
	restless.between = func() {
		moves++
		if _, err := artifact.Publish(ctx, store, artifact.Batch{
			Key:       fmt.Sprintf("test/a-commit-beside-the-confirmation/%d", moves),
			Artifacts: []artifact.Descriptor{{ID: testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("a commit beside the confirmation %d", moves))}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	before = prepared
	if _, _, err := ConfirmOnPool(ctx, restless, pool.Acceptance.ID, candidate("seven"), reading); err == nil || !strings.Contains(err.Error(), "moved under") || prepared != before || charges() != 0 || moves < chargeAttempts {
		t.Fatalf("under a store committed to during each of %d readings, the confirmation = %v, with %d readings and %d charges; want it to give up after %d askings with nothing charged or read", moves, err, prepared-before, charges(), chargeAttempts)
	}
}
