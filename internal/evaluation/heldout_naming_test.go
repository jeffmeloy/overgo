package evaluation

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestMaterialChargedUnnamedIsNotFreshWhenImportedAgain holds how material
// that answered confirmations before a part named what it holds is admitted
// when it is registered again: its earlier parts are named first, as a part
// published today is, and what they answered is then found from the new
// registration.
//
// A pool is published as pools were then, with no names, and charged once as
// a confirmation was charged then. The same bytes are registered as another
// dataset and a pool is built over it. Until the earlier part is named the
// new pool counts nothing its documents have answered, which is the fault.
// Named, the earlier part contains what a pool published today contains, the
// new pool starts with that confirmation spent, its last is refused, and the
// earlier part answers again, counted like any other. Naming a second time
// names nothing.
//
// An instruction acceptance published then is named by its cases' contents,
// as one built today is, when the store's cases are records of its dataset.
// When they are another dataset's nothing names its prompts: uncharged it is
// passed by, and charged the naming is refused, and with it whatever asked to
// admit material.
func TestMaterialChargedUnnamedIsNotFreshWhenImportedAgain(t *testing.T) {
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
	model := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindModel, "naming fixture "+name)
	}
	// unnamed publishes a batch as it was published before parts named what they hold, and gives the names a part of it is published with today.
	unnamed := func(store *overgodb.Store, batch artifact.Batch, part artifact.ID) (today []artifact.ID) {
		t.Helper()
		for _, edge := range batch.Lineage {
			if edge.Relation == artifact.RelationContains && edge.Parent == part {
				today = append(today, edge.Child)
			}
		}
		batch.Lineage = slices.DeleteFunc(slices.Clone(batch.Lineage), func(edge artifact.Lineage) bool { return edge.Relation == artifact.RelationContains })
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		if len(today) == 0 {
			t.Fatal("the fixture's part is published with no names today either; the test needs one that is")
		}
		return today
	}
	// chargedThen charges a grant directly, as a confirmation was charged before a part without names was refused one.
	chargedThen := func(store *overgodb.Store, unit string, part artifact.ID, issued uint64, authority, owner artifact.ID, purpose string) {
		t.Helper()
		granted, err := runrecord.NewBudget(unit, part, issued, authority)
		if err != nil {
			t.Fatal(err)
		}
		// The grant is published here when the fixture has not published it; one it has is no change.
		grant, err := granted.Batch("test/granted-before-names/" + granted.ID.String())
		if err == nil {
			_, err = artifact.Publish(ctx, store, grant)
		}
		if err != nil {
			t.Fatal(err)
		}
		confirmer, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(poolConfirmer+owner.String()))
		if err != nil {
			t.Fatal(err)
		}
		charge, err := runrecord.NewBudgetCharge(granted.ID, 1, confirmer, purpose)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := charge.Batch("test/charged-before-names/" + charge.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: confirmer})
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	// namedAs reports whether a part now contains exactly the names given.
	namedAs := func(store *overgodb.Store, part artifact.ID, today []artifact.ID) bool {
		t.Helper()
		published, err := partContents(ctx, store, part)
		if err != nil {
			t.Fatal(err)
		}
		order := func(left, right artifact.ID) int { return strings.Compare(left.String(), right.String()) }
		slices.SortFunc(published, order)
		today = slices.Clone(today)
		slices.SortFunc(today, order)
		return slices.Equal(slices.Compact(published), slices.Compact(today))
	}

	// A pool.
	store, objective, _, _ := poolObjective(t)
	request := HeldoutPoolRequest{
		Objective: objective, Base: model("base"), Sequence: 8,
		Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "naming fixture verdict")}, Weights: [poolPartCount]uint64{1, 2, 4, 4},
	}
	earlier := sizedPool(t, store, request, 2)
	today := unnamed(store, earlier.Batch, earlier.Partition.Promotion)
	chargedThen(store, poolUnit, earlier.Partition.Promotion, earlier.Acceptance.Confirmations, earlier.Acceptance.ID, earlier.Acceptance.Objective, poolChargePurpose+model("judged then").String())
	again, _, _ := poolObjectiveOver(t, store, "pool-corpus-again", poolFixtureDocuments+1)
	renewed := request
	renewed.Objective = again
	later := publishedPool(t, store, renewed, 2)
	membership, found, err := dataset.LoadMembership(ctx, store, later.Partition.Promotion)
	if err != nil || !found || later.Partition.Dataset == earlier.Partition.Dataset {
		t.Fatalf("the pool over the bytes registered again is of the first dataset, or its part is not committed (%t, %v)", found, err)
	}
	answered := func() uint64 {
		t.Helper()
		spent, _, _, err := exposure(ctx, store, later.Partition, artifact.ID{}, membership)
		if err != nil {
			t.Fatal(err)
		}
		return spent
	}
	if answered() != 0 {
		t.Fatalf("before the earlier part is named, the pool over its bytes registered again counts %d confirmations answered; the test needs the fault it is about, which counts none", answered())
	}
	named, err := NameEarlierParts(ctx, store)
	if err != nil || named == 0 || !namedAs(store, earlier.Partition.Promotion, today) {
		t.Fatalf("naming the earlier parts named %d (%v), and the charged part contains what it is published with today: %t", named, err, namedAs(store, earlier.Partition.Promotion, today))
	}
	if answered() != 1 {
		t.Fatalf("with the earlier part named, the pool over its bytes registered again counts %d confirmations answered, want the one charged then", answered())
	}
	if _, left, err := ConfirmOnPool(ctx, store, later.Acceptance.ID, model("judged now"), reading); err != nil || left != 0 || prepared != 1 {
		t.Fatalf("the new pool's first confirmation leaves %d after %d readings (%v), want none of its two: one was answered before the bytes were registered again", left, prepared, err)
	}
	for name, acceptance := range map[string]artifact.ID{"the new pool": later.Acceptance.ID, "the earlier pool, named": earlier.Acceptance.ID} {
		if _, _, err := ConfirmOnPool(ctx, store, acceptance, model("judged last"), reading); err == nil || !strings.Contains(err.Error(), "fresh partition") || prepared != 1 {
			t.Fatalf("a further confirmation on %s = %v after %d readings, want it refused: its documents have answered what they allow", name, err, prepared)
		}
	}
	if more, err := NameEarlierParts(ctx, store); err != nil || more != 0 {
		t.Fatalf("naming a second time named %d (%v), want nothing left to name", more, err)
	}

	// An instruction acceptance, over a dataset of its own store.
	prompts, held, training, heldout := poolObjective(t)
	contents := map[string]string{}
	var cases []InstructionCase
	const combinations = 140
	for index, record := range slices.Concat(training.Records, heldout.Records) {
		asked := InstructionCase{Record: record.ID, Rules: []string{fmt.Sprintf("length:%d", index%combinations), fmt.Sprintf("keywords:%d", index%combinations)}, Content: fmt.Sprintf("naming fixture case %d", index)}
		cases, contents[asked.Record] = append(cases, asked), asked.Content
	}
	fixed := InstructionAcceptanceRequest{
		Dataset: held.Dataset, Base: model("instruction base"), Training: testutil.ArtifactID(t, artifact.KindProfile, "naming fixture training"), Cases: cases,
		Protocol: InstructionProtocol{Scorer: InstructionScorer, Template: "chat-template", Thinking: "off", Sampling: "greedy", Stop: "end of turn", Window: 4096, MaxTokens: 1280},
		Seed:     7, Weights: [poolPartCount]uint64{1, 1, 4, 1}, Gain: 0.2,
		Comparisons: []InstructionComparison{{Name: "reward over base", Candidate: "reward", Baseline: "base"}, {Name: "reward over control", Candidate: "reward", Baseline: "control"}},
		Pairs:       200, Discordant: 10, Basis: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvaluation, "naming fixture development reading")},
	}
	if _, err := artifact.Publish(ctx, prompts, artifact.Batch{Key: "naming-fixture/training", Artifacts: []artifact.Descriptor{{ID: fixed.Training}}}); err != nil {
		t.Fatal(err)
	}
	acceptance, batch, err := BuildInstructionAcceptance(fixed)
	if err != nil {
		t.Fatal(err)
	}
	var partition runrecord.SplitPartition
	for _, content := range batch.Contents {
		if content.Descriptor.ID == acceptance.Split {
			if partition, err = runrecord.ParseSplitPartition(content.Data); err != nil {
				t.Fatal(err)
			}
		}
	}
	today = unnamed(prompts, batch, partition.Promotion)
	theirs := func() (map[string]string, artifact.ID) { return contents, held.Dataset }
	anothers := func() (map[string]string, artifact.ID) {
		return contents, testutil.ArtifactID(t, artifact.KindDataset, "naming fixture another catalog")
	}
	// A grant that counts something else, charged, on a part nothing names, is no grant of confirmations.
	chargedThen(prompts, "readings of another kind", partition.Development, 1, acceptance.ID, acceptance.Training, "a reading")
	if named, err := nameEarlierParts(ctx, prompts, anothers); err != nil || named != 0 {
		t.Fatalf("an uncharged acceptance whose prompts the store's cases do not name, beside a charged grant of another unit = %d named (%v), want both passed by", named, err)
	}
	chargedThen(prompts, poolUnit, partition.Promotion, uint64(len(acceptance.Comparisons)), acceptance.ID, acceptance.Training, "comparison reward over base: "+model("reward").String()+" over "+fixed.Base.String())
	if named, err := nameEarlierParts(ctx, prompts, anothers); err == nil || !strings.Contains(err.Error(), "cannot follow") || named != 0 {
		t.Fatalf("a charged acceptance whose prompts nothing names = %d named (%v), want the naming refused", named, err)
	}
	if named, err := nameEarlierParts(ctx, prompts, theirs); err != nil || named != 1 || !namedAs(prompts, partition.Promotion, today) {
		t.Fatalf("a charged acceptance whose prompts are the store's cases = %d named (%v), and its part contains what an acceptance built today names: %t", named, err, namedAs(prompts, partition.Promotion, today))
	}
}
