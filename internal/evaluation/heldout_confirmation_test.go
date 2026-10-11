package evaluation

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// publishedPool builds and publishes the pool of the fixture's objective
// whose promotion part answers exactly confirmations of them: its gain is set
// between the one that count needs and the one the next would.
func publishedPool(t *testing.T, store *overgodb.Store, request HeldoutPoolRequest, confirmations uint64) HeldoutPool {
	t.Helper()
	ctx := t.Context()
	request.Gain = 1
	sized, err := BuildHeldoutPool(ctx, store, request)
	if err != nil {
		t.Fatal(err)
	}
	missed := normalQuantile(1 - missedGain)
	request.Gain = (poolCritical(confirmations) + poolCritical(confirmations+1) + missed + missed) / 2 / math.Sqrt(float64(sized.Documents[poolPromotion]))
	pool, err := BuildHeldoutPool(ctx, store, request)
	if err != nil || pool.Acceptance.Confirmations != confirmations {
		t.Fatalf("a pool sized for %d confirmations answers %d (%v)", confirmations, pool.Acceptance.Confirmations, err)
	}
	if _, err := artifact.Publish(ctx, store, pool.Batch); err != nil {
		t.Fatal(err)
	}
	return pool
}

// TestConfirmationIsChargedBeforeThePromotionPartIsRead holds a pool's
// confirmations to its budget. A confirmation is charged before the promotion
// part's documents are handed to the reading, and those are the documents it
// is handed; its verdict is read from the differences by the acceptance's
// statistic. A reading that fails leaves its charge and owes its verdict: the
// candidate asked again is read and not charged again, and one whose verdict
// is recorded is neither read nor charged. Past the budget nothing is charged
// and nothing is read, and a pool whose promotion part the proposer has
// charged answers nothing.
func TestConfirmationIsChargedBeforeThePromotionPartIsRead(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, objective, _, _ := poolObjective(t)
	request := HeldoutPoolRequest{
		Objective: objective, Base: testutil.ArtifactID(t, artifact.KindModel, "pool fixture base"), Sequence: 8,
		Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "pool fixture verdict")}, Weights: [poolPartCount]uint64{1, 2, 4, 4},
	}
	const issued = 2
	pool := publishedPool(t, store, request, issued)
	acceptance := pool.Acceptance
	membership, found, err := dataset.LoadMembership(ctx, store, pool.Partition.Promotion)
	if err != nil || !found {
		t.Fatalf("the promotion part is committed %t (%v)", found, err)
	}
	var promotion []string
	for _, record := range membership.Records {
		promotion = append(promotion, record.ID)
	}
	budget, err := runrecord.NewBudget(poolUnit, pool.Partition.Promotion, acceptance.Confirmations, acceptance.ID)
	if err != nil {
		t.Fatal(err)
	}
	charged := func() int {
		_, charges, err := runrecord.BudgetCharges(ctx, store, budget.ID)
		if err != nil {
			t.Fatal(err)
		}
		return len(charges)
	}
	// reading answers with one difference a document, and holds that it was
	// handed the promotion part's documents after want charges were committed.
	reading := func(want int, difference func(document int) float64) func([]string) ([]float64, error) {
		return func(documents []string) ([]float64, error) {
			if !slices.Equal(documents, promotion) || charged() != want {
				t.Errorf("a reading was handed %d documents with %d charges committed, want the promotion part's %d after %d", len(documents), charged(), len(promotion), want)
			}
			differences := make([]float64, len(documents))
			for index := range differences {
				differences[index] = difference(index)
			}
			return differences, nil
		}
	}
	unread := func([]string) ([]float64, error) {
		t.Error("the promotion part was read")
		return nil, nil
	}
	candidate := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, "candidate "+name) }

	gain, left, err := ConfirmOnPool(ctx, store, acceptance.ID, candidate("better"), reading(1, func(document int) float64 { return 1 + float64(document%2) }))
	if err != nil || gain.Verdict != poolGain || left != issued-1 {
		t.Fatalf("a candidate better on every document = (%q, %d left, %v), want a gain and %d left", gain.Verdict, left, err, issued-1)
	}
	failed := errors.New("the candidate could not be read")
	if _, _, err := ConfirmOnPool(ctx, store, acceptance.ID, candidate("worse"), func([]string) ([]float64, error) { return nil, failed }); !errors.Is(err, failed) || charged() != issued {
		t.Fatalf("a reading that fails = %v with %d charges, want its error and its charge kept", err, charged())
	}
	loss, left, err := ConfirmOnPool(ctx, store, acceptance.ID, candidate("worse"), reading(issued, func(document int) float64 { return -1 - float64(document%2) }))
	if err != nil || loss.Verdict != poolLoss || left != 0 || charged() != issued {
		t.Fatalf("the candidate asked again = (%q, %d left, %v) with %d charges, want its owed verdict, a loss, and no second charge", loss.Verdict, left, err, charged())
	}
	recorded, left, err := ConfirmOnPool(ctx, store, acceptance.ID, candidate("better"), unread)
	if err != nil || recorded.ID != gain.ID || left != 0 {
		t.Fatalf("a confirmed candidate asked again = (%s, %d left, %v), want its recorded confirmation %s", recorded.ID, left, err, gain.ID)
	}
	if _, _, err := ConfirmOnPool(ctx, store, acceptance.ID, candidate("third"), unread); err == nil || !strings.Contains(err.Error(), "fresh partition") || charged() != issued {
		t.Fatalf("a confirmation past the budget = %v with %d charges, want it refused and nothing charged", err, charged())
	}
	committed, err := poolConfirmationCodec.Require(ctx, store, loss.ID)
	if err != nil || committed.Candidate != candidate("worse") || committed.Acceptance != acceptance.ID {
		t.Fatalf("the committed confirmation = %+v (%v)", committed, err)
	}

	// Differences that settle nothing, and ones that do not vary.
	for name, verdict := range map[string]struct {
		differences []float64
		want        string
	}{
		"as often better as worse": {[]float64{1, -1, 1, -1, 1, -1}, poolUndecided},
		"better by too little":     {[]float64{3, -2, 3, -2, 3, -2}, poolUndecided},
		"the same everywhere":      {[]float64{0, 0, 0}, poolUndecided},
		"better alike everywhere":  {[]float64{2, 2, 2}, poolGain},
		"worse alike everywhere":   {[]float64{-2, -2, -2}, poolLoss},
	} {
		if got := poolVerdict(verdict.differences, acceptance.Critical); got != verdict.want {
			t.Errorf("differences %s read %q, want %q", name, got, verdict.want)
		}
	}

	// A second pool of the same documents, whose promotion part the proposer charges.
	blinded := publishedPool(t, store, request, issued+1)
	proposer, err := changeProposer(objective.ID)
	if err != nil {
		t.Fatal(err)
	}
	granted, err := runrecord.NewBudget(poolUnit, blinded.Partition.Promotion, blinded.Acceptance.Confirmations, blinded.Acceptance.ID)
	if err != nil {
		t.Fatal(err)
	}
	peek, err := runrecord.NewBudgetCharge(granted.ID, 1, proposer, "a reading by the proposer")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := peek.Batch("pool-fixture/proposer-charge")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ConfirmOnPool(ctx, store, blinded.Acceptance.ID, candidate("better"), unread); err == nil || !strings.Contains(err.Error(), "blinded") {
		t.Fatalf("a pool whose promotion part the proposer charged = %v, want its confirmations refused", err)
	}
}
