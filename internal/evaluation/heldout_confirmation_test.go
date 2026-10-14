package evaluation

import (
	"context"
	"errors"
	"fmt"
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
	pool := sizedPool(t, store, request, confirmations)
	if _, err := artifact.Publish(t.Context(), store, pool.Batch); err != nil {
		t.Fatal(err)
	}
	return pool
}

// sizedPool is a pool built to answer so many confirmations, not yet
// published: publishedPool's, for a test that publishes it otherwise.
func sizedPool(t *testing.T, store *overgodb.Store, request HeldoutPoolRequest, confirmations uint64) HeldoutPool {
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

// poolChargePurpose is what a confirmation's charge is for, before the
// candidate it names.
const poolChargePurpose = "confirmation of "

// CommittedPool is a held-out pool as the store holds it, for whoever fits
// and ranks candidates on it: the acceptance its confirmations are judged by,
// the partition that binds its parts, and the documents of its development
// and selection parts, each in its membership's order. The promotion part's
// documents are handed out by ConfirmOnPool alone.
type CommittedPool struct {
	Acceptance  PoolAcceptance
	Partition   runrecord.SplitPartition
	Development []string
	Selection   []string
}

// RequireHeldoutPool loads the pool an acceptance was fixed for.
func RequireHeldoutPool(ctx context.Context, reader artifact.Reader, acceptance artifact.ID) (CommittedPool, error) {
	var pool CommittedPool
	var err error
	if pool.Acceptance, err = poolAcceptanceCodec.Require(ctx, reader, acceptance); err != nil {
		return CommittedPool{}, err
	}
	content, found, err := artifact.ReadContent(ctx, reader, pool.Acceptance.Split)
	if err != nil || !found {
		return CommittedPool{}, errors.Join(fmt.Errorf("evaluation: the partition of pool %s is not committed", acceptance), err)
	}
	if pool.Partition, err = runrecord.ParseSplitPartition(content.Data); err != nil {
		return CommittedPool{}, err
	}
	if pool.Development, err = partDocuments(ctx, reader, pool.Partition.Development); err != nil {
		return CommittedPool{}, err
	}
	if pool.Selection, err = partDocuments(ctx, reader, pool.Partition.Selection); err != nil {
		return CommittedPool{}, err
	}
	return pool, nil
}

// ConfirmOnPool has the promotion part of a pool answer for one candidate. The
// candidate and the acceptance are named by identity before anything is read,
// and without both the confirmation is refused. A confirmation is charged to
// the promotion part's budget before its documents are handed to read, which
// returns the pool's measure for each of them, the candidate less the base; the
// verdict is read from those by the acceptance's rule and published, and it
// returns with what the budget has left and nothing else of the reading.
//
// What is left is counted from the charges the store holds, so it is the same
// in any process. A candidate charged for before is not charged again: its
// recorded verdict is returned, or read now when the reading that owed it did
// not finish. When nothing is left a confirmation is refused, and the pool
// answers again only as a fresh partition.
func ConfirmOnPool(ctx context.Context, store artifact.Repository, acceptanceID, candidate artifact.ID, read func(documents []string) ([]float64, error)) (PoolConfirmation, uint64, error) {
	return oneDecision(store, func(head artifact.CommitID) (PoolConfirmation, uint64, error) {
		pool, err := RequireHeldoutPool(ctx, store, acceptanceID)
		if err != nil || !candidate.Valid() {
			return PoolConfirmation{}, 0, errors.Join(errors.New("evaluation: a confirmation names its candidate and the acceptance it is judged by before it reads anything"), err)
		}
		acceptance := pool.Acceptance
		if acceptance.Sequence == 0 {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: pool %s fixed no tokens to read each document as far as, so nothing is confirmed on it; build a pool that does", acceptance.ID)
		}
		return confirmOnPart(ctx, store, head, pool.Partition, acceptance.ID, acceptance.Objective, candidate, acceptance.Documents, acceptance.Confirmations, poolChargePurpose+candidate.String(),
			func(promotion []string) (string, error) {
				differences, err := read(promotion)
				if err != nil || len(differences) != len(promotion) {
					return "", errors.Join(fmt.Errorf("evaluation: the candidate was read on %d of the promotion part's %d documents; its confirmation is charged and its verdict is owed", len(differences), len(promotion)), err)
				}
				return poolVerdict(differences, acceptance.Critical), nil
			})
	})
}

// poolVerdict reads a confirmation from the measure's paired differences, one
// a document of the promotion part: a gain when their mean over its standard
// error passes the acceptance's statistic, a loss when it passes it downward,
// and otherwise not enough evidence. Differences that do not vary have no
// error to divide by: the statistic is then without bound on the mean's side,
// and undefined, passing neither way, when the mean is zero.
func poolVerdict(differences []float64, critical float64) string {
	scores := make([]RecordScore, len(differences))
	var mean float64
	for index, difference := range differences {
		scores[index].Value = difference
		mean += difference / float64(len(differences))
	}
	switch statistic := mean / meanError(scores); {
	case statistic > critical:
		return poolGain
	case statistic < -critical:
		return poolLoss
	}
	return poolUndecided
}
