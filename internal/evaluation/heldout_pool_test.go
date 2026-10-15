package evaluation

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// Sizes of the pool fixture: the documents of its corpus, of which some are
// copies of others under another name, and the held-out records its objective
// has scored before.
const (
	poolFixtureDocuments = 420
	poolFixtureCopies    = 40
	poolFixtureScored    = 12
)

// poolObjective registers a corpus of documents, some of them copies of
// others, splits it one part training to three held out, and publishes a
// token-prediction objective over it that scores the first few held-out
// records. It returns the store, the objective and both memberships.
func poolObjective(t *testing.T) (*overgodb.Store, trainingprogram.ObjectiveDocument, dataset.Membership, dataset.Membership) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	objective, training, heldout := poolObjectiveOver(t, store, "pool-corpus", poolFixtureDocuments)
	return store, objective, training, heldout
}

// poolObjectiveOver is poolObjective in a store given, over a corpus of the
// first so many documents registered under a name: a corpus of more documents
// is another dataset that holds the same bytes.
func poolObjectiveOver(t *testing.T, store *overgodb.Store, name string, documents int) (trainingprogram.ObjectiveDocument, dataset.Membership, dataset.Membership) {
	t.Helper()
	ctx := t.Context()
	corpus := t.TempDir()
	for index := range documents {
		text := fmt.Sprintf("document %d of the pool's corpus\n", index%(poolFixtureDocuments-poolFixtureCopies))
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("document-%03d.txt", index)), []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, name, corpus)
	if err != nil {
		t.Fatal(err)
	}
	document, _, err := dataset.Load(ctx, store, registered.Dataset)
	if err != nil {
		t.Fatal(err)
	}
	var records []dataset.Record
	for _, asset := range document.Assets {
		for ordinal := range asset.Records {
			id := dataset.RecordID(registered.Dataset, asset.Name, ordinal)
			records = append(records, dataset.Record{ID: id, Group: id})
		}
	}
	split, err := dataset.BuildGroupSplit(registered.Dataset, records, 1, []dataset.SplitPartition{{Name: "train", Weight: 1}, {Name: "heldout", Weight: 3}})
	if err != nil {
		t.Fatal(err)
	}
	training, heldout := split.Memberships[1], split.Memberships[0]
	profile := func(role string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindProfile, "pool fixture "+role)
	}
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "pool fixture evidence")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "pool fixture token prediction", Kind: trainingprogram.ObjectiveTokenPrediction,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{"text"}, Outputs: []recipecontract.Modality{"text"}},
		Dataset:   registered.Dataset, Split: training.ID, Processors: []artifact.ID{profile("processor")},
		Loss: profile("loss"), Evaluation: profile("evaluation"), Metric: trainingprogram.MetricTokenAccuracy,
		HeldoutRecords: poolFixtureScored, Evidence: []artifact.ID{evidence}, Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := split.PublicationBatch("pool-fixture/"+objective.ID.String(), nil)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: profile("processor")}, artifact.Descriptor{ID: profile("loss")},
		artifact.Descriptor{ID: profile("evaluation")}, artifact.Descriptor{ID: evidence})
	batch.Contents = append(batch.Contents, content)
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return objective, training, heldout
}

// TestHeldOutPoolIsSizedForItsDeclaredGain holds a held-out pool to what it
// states. The pool divides an objective's held-out documents into the four
// parts of a split partition: no document of the training membership is in
// it, no two of its documents have the same content, each is in one part, and
// the documents a verdict has already read are excluded from every part. Its
// acceptance is fixed with it: the promotion part answers as many
// confirmations as its documents can show the stated gain for at the two
// error rates, and one more would need more documents than it has; the
// statistic a confirmation must pass divides the false-gain rate over every
// sequence of verdicts a proposer could have been told; the budgets the
// promotion and audit parts grant are that count and one, and the proposer is
// blinded from both. A gain the pool cannot show once is refused with the
// documents it would need, and the same request builds the same pool.
func TestHeldOutPoolIsSizedForItsDeclaredGain(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, objective, training, heldout := poolObjective(t)
	request := HeldoutPoolRequest{
		Objective: objective, Base: testutil.ArtifactID(t, artifact.KindModel, "pool fixture base"), Gain: 0.5, Sequence: 8,
		Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "pool fixture verdict")}, Weights: [poolPartCount]uint64{1, 2, 6, 2},
	}
	pool, err := BuildHeldoutPool(ctx, store, request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, pool.Batch); err != nil {
		t.Fatal(err)
	}

	// The parts hold held-out documents only, each once, none twice by content.
	contents, err := recordContents(ctx, store, objective.Dataset)
	if err != nil {
		t.Fatal(err)
	}
	trained := map[string]bool{}
	for _, record := range training.Records {
		trained[contents[record.ID]] = true
	}
	held, placed := map[string]bool{}, map[string]string{}
	for _, record := range heldout.Records {
		held[record.ID] = true
	}
	shards := [poolPartCount]artifact.ID{pool.Partition.Development, pool.Partition.Selection, pool.Partition.Promotion, pool.Partition.Audit}
	documents := 0
	for part, shard := range shards {
		membership, found, err := dataset.LoadMembership(ctx, store, shard)
		if err != nil || !found || len(membership.Records) != pool.Documents[part] || membership.Partition != PoolParts[part] {
			t.Fatalf("the %s part is not the committed membership the pool counts: %+v, %t, %v", PoolParts[part], membership.Partition, found, err)
		}
		for _, record := range membership.Records {
			content := contents[record.ID]
			if !held[record.ID] || trained[content] || placed[content] != "" {
				t.Fatalf("%s is in the %s part: held out %t, content the training membership holds %t, content already in the %s part",
					record.ID, PoolParts[part], held[record.ID], trained[content], placed[content])
			}
			placed[content] = PoolParts[part]
			documents++
		}
	}
	if dropped := len(heldout.Records) - poolFixtureScored - documents; dropped <= 0 {
		t.Fatalf("the pool holds all %d unread held-out records of a corpus with copies in it; want the copies left out", len(heldout.Records)-poolFixtureScored)
	}
	// The records the objective's verdicts have read are in no part, by record or by content.
	for _, record := range heldout.Records[:poolFixtureScored] {
		if part := placed[contents[record.ID]]; part != "" {
			t.Fatalf("%s, which the objective's verdicts have read, has its content in the %s part", record.ID, part)
		}
	}

	// The acceptance states the count the promotion part's size allows, and no more.
	acceptance, err := poolAcceptanceCodec.Require(ctx, store, pool.Acceptance.ID)
	if err != nil {
		t.Fatal(err)
	}
	promotion := uint64(pool.Documents[poolPromotion])
	if acceptance.Split != pool.Partition.ID || acceptance.Documents != promotion || acceptance.Confirmations == 0 ||
		poolDocuments(request.Gain, acceptance.Confirmations) > promotion || poolDocuments(request.Gain, acceptance.Confirmations+1) <= promotion ||
		acceptance.SmallestGain() > request.Gain {
		t.Fatalf("the acceptance %+v over %d promotion documents does not state the most confirmations they can show a gain of %.2f for (%d need %d, one more %d)",
			acceptance, promotion, request.Gain, acceptance.Confirmations, poolDocuments(request.Gain, acceptance.Confirmations), poolDocuments(request.Gain, acceptance.Confirmations+1))
	}
	// One confirmation is read at the false-gain rate itself; by the third, a
	// proposer told two verdicts could be confirming any of 1 + 3 + 9 candidates.
	for confirmations, reachable := range map[uint64]float64{1: 1, 2: 4, 3: 13} {
		if got, want := poolCritical(confirmations), normalQuantile(1-falseGain/reachable); math.Abs(got-want) > 1e-12 {
			t.Errorf("a pool answering %d confirmations holds each to %.6f, want the false-gain rate over %v candidates, %.6f", confirmations, got, reachable, want)
		}
	}
	if once := poolDocuments(request.Gain, 1); once != 25 || poolConfirmations(once, request.Gain) != 1 || poolConfirmations(once-1, request.Gain) != 0 {
		t.Errorf("a gain of half a standard deviation needs %d documents to be shown once, want 25: (1.645 + 0.842) / 0.5, squared", once)
	}

	// The budgets are the acceptance's, and the proposer is blinded from both parts.
	proposer, err := changeProposer(objective.ID)
	if err != nil {
		t.Fatal(err)
	}
	issued := map[artifact.ID]uint64{}
	for _, content := range pool.Batch.Contents {
		if content.Descriptor.MediaType != runrecord.BudgetMediaType {
			continue
		}
		budget, err := runrecord.ParseBudget(content.Data)
		if err != nil || budget.Authority != acceptance.ID {
			t.Fatalf("a budget of the pool: %+v, %v; want the acceptance as its authority", budget, err)
		}
		issued[budget.Split] = budget.Issued
		charge, err := runrecord.NewBudgetCharge(budget.ID, 1, proposer, "a reading by the proposer")
		if err != nil {
			t.Fatal(err)
		}
		if err := runrecord.ValidateBlinding(pool.Partition, budget, []runrecord.BudgetCharge{charge}); err == nil {
			t.Errorf("the proposer may charge the budget of %s", budget.Split)
		}
	}
	if len(issued) != 2 || issued[pool.Partition.Promotion] != acceptance.Confirmations || issued[pool.Partition.Audit] != 1 {
		t.Fatalf("the pool grants %v, want %d confirmations of the promotion part and one reading of the audit part", issued, acceptance.Confirmations)
	}

	// A gain the pool cannot show once is refused; the same request builds the same pool.
	small := request
	small.Gain = 0.05
	if _, err := BuildHeldoutPool(ctx, store, small); err == nil || !strings.Contains(err.Error(), fmt.Sprint(poolDocuments(small.Gain, 1))) {
		t.Fatalf("a pool sized for a gain it cannot show: %v, want a refusal naming the documents one showing needs", err)
	}
	again, err := BuildHeldoutPool(ctx, store, request)
	if err != nil || again.Acceptance.ID != pool.Acceptance.ID || again.Partition.ID != pool.Partition.ID || !slices.Equal(again.Documents[:], pool.Documents[:]) {
		t.Fatalf("the same request built another pool: %v", err)
	}
}

// TestRecordedGainsReadTheVerdictsOnRecord holds the reading a pool's gain is
// stated from to the committed verdicts of the objective: a verdict whose
// trained model halved the error on every other held-out record shows that
// gain as a multiple of the spread of its record differences, counted as a
// gain because the metric is one that falls, with the base model the verdict
// judged and the verdict itself as the basis; an objective with no verdict on
// record gives nothing to state a gain from.
func TestRecordedGainsReadTheVerdictsOnRecord(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	commit := func(key string, stage func(string) (artifact.Batch, error), declared ...artifact.ID) {
		t.Helper()
		batch, err := stage(key)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range declared {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	commit("test-split", func(key string) (artifact.Batch, error) { return forecast.split.PublicationBatch(key, nil) }, forecast.objective.Dataset)
	commit("test-objective", func(key string) (artifact.Batch, error) {
		content, err := forecast.objective.Content()
		return artifact.Batch{Key: key, Contents: []artifact.Content{content}}, err
	})
	if _, _, _, err := RecordedGains(ctx, store, forecast.objective); err == nil {
		t.Fatal("an objective with no verdict on record stated a gain")
	}
	session := heldoutTraining(t, store, forecast.objective.Dataset, forecast.view.TrainingMembership)
	base := forecast.score(t, 2, "falling")
	var observations []HeldoutObservation
	var mean, squares float64
	for index, target := range forecast.cases {
		offset := float64(2 - index%2)
		observations = append(observations, HeldoutObservation{Record: target.Record, Values: []float64{target.Values[0] + offset, target.Values[1] + offset}, Label: "falling"})
		mean += (2 - offset) / float64(len(forecast.cases))
	}
	for index := range forecast.cases {
		squares += (float64(index%2) - mean) * (float64(index%2) - mean)
	}
	trained, err := forecast.plan.Score(observations)
	if err != nil {
		t.Fatal(err)
	}
	commit("test-view", forecast.view.Batch, forecast.objective.Processors...)
	commit("test-plan", forecast.plan.Batch)
	commit("test-base", base.Batch)
	commit("test-trained", trained.Batch)
	verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, trained, session)
	if err != nil {
		t.Fatal(err)
	}
	commit("test-verdict", verdict.Batch)
	gains, model, basis, err := RecordedGains(ctx, store, forecast.objective)
	want := mean / math.Sqrt(squares/float64(len(forecast.cases)-1))
	if err != nil || len(gains) != 1 || math.Abs(gains[0]-want) > 1e-9 || model != session.Model || !slices.Equal(basis, []artifact.ID{verdict.ID}) {
		t.Fatalf("the gains on record = %v of base %s on %v (%v); want one of %.6f standard deviations, of base %s, on the committed verdict", gains, model, basis, err, want, session.Model)
	}
}
