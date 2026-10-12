package evaluation

import (
	"fmt"
	"maps"
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
// earlier part answers again, counted like any other. The earlier pool's audit
// part, which answered nothing, is named too. A second naming names nothing.
//
// An instruction acceptance published then is named by its cases' contents,
// as one built today is, when the store's cases are records of its dataset.
// When they are another dataset's nothing names its prompts: uncharged it is
// passed by, and charged the naming is refused, and with it whatever asked to
// admit material.
//
// A part is named whole or not at all. One that holds a document with no
// recorded content, beside others that have one, would be found from another
// registration of the others only, and the one left out would read as fresh
// when its bytes were registered again. Such a part is given no names:
// uncharged it is passed by, and charged the naming is refused, also when the
// names of its other documents were published before. Two records of one
// content share a name, and a part of them is whole. An acceptance one of
// whose prompts the store's cases do not name is held to the same.
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
	// grantedThen publishes a grant when the fixture has not published it; one it has is no change.
	grantedThen := func(store *overgodb.Store, unit string, part artifact.ID, issued uint64, authority artifact.ID) runrecord.Budget {
		t.Helper()
		granted, err := runrecord.NewBudget(unit, part, issued, authority)
		if err != nil {
			t.Fatal(err)
		}
		grant, err := granted.Batch("test/granted-before-names/" + granted.ID.String())
		if err == nil {
			_, err = artifact.Publish(ctx, store, grant)
		}
		if err != nil {
			t.Fatal(err)
		}
		return granted
	}
	// chargedThen charges a grant directly, as a confirmation was charged before a part without names was refused one.
	chargedThen := func(store *overgodb.Store, unit string, part artifact.ID, issued uint64, authority, owner artifact.ID, purpose string) {
		t.Helper()
		granted := grantedThen(store, unit, part, issued, authority)
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
	if err != nil || named != 2 || !namedAs(store, earlier.Partition.Promotion, today) {
		t.Fatalf("naming the earlier parts named %d (%v), want two, the earlier pool's charged promotion part and its audit part, which has answered nothing; and the charged part contains what it is published with today: %t", named, err, namedAs(store, earlier.Partition.Promotion, today))
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

	// A part is named whole or not at all. The record one past the inventory's
	// files stands for a file registered with no digest: the naming reads a
	// record's content from the same mapping, which has none for either.
	recorded, err := recordContents(ctx, store, earlier.Partition.Dataset)
	if err != nil {
		t.Fatal(err)
	}
	promoted, found, err := dataset.LoadMembership(ctx, store, earlier.Partition.Promotion)
	if err != nil || !found {
		t.Fatalf("the earlier pool's promotion part is not committed (%v)", err)
	}
	record := func(ordinal int) dataset.Record {
		id := dataset.RecordID(earlier.Partition.Dataset, dataset.InventoryAssetName, uint64(ordinal))
		return dataset.Record{ID: id, Group: id}
	}
	first, copied, unrecorded := record(0), record(poolFixtureDocuments-poolFixtureCopies), record(poolFixtureDocuments)
	if recorded[first.ID] == "" || recorded[first.ID] != recorded[copied.ID] || recorded[promoted.Records[0].ID] == "" || recorded[promoted.Records[1].ID] == "" || recorded[unrecorded.ID] != "" {
		t.Fatal("the fixture needs a document and its copy with one recorded content, two promotion documents with a content, and a record with none")
	}
	// partThen publishes a promotion part of the earlier dataset with no names, and gives the batch that names each of its records a content is recorded for.
	partThen := func(records ...dataset.Record) (dataset.Membership, artifact.Batch) {
		t.Helper()
		part, err := dataset.NewMembership(earlier.Partition.Dataset, 0, PoolParts[poolPromotion], records)
		if err != nil {
			t.Fatal(err)
		}
		content, err := part.Content()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "test/part-before-names/" + part.ID.String(), Contents: []artifact.Content{content}, Lineage: []artifact.Lineage{part.Lineage()}}); err != nil {
			t.Fatal(err)
		}
		names := artifact.Batch{Key: "test/named-for-some/" + part.ID.String()}
		if err := containContents(&names, []dataset.Membership{part}, recorded); err != nil {
			t.Fatal(err)
		}
		return part, names
	}
	// A charged part of a document and its copy holds two records under one name, and is whole.
	twin, shared := partThen(first, copied)
	chargedThen(store, poolUnit, twin.ID, 1, earlier.Acceptance.ID, earlier.Acceptance.Objective, "a confirmation of the part of a document and its copy")
	if named, err := NameEarlierParts(ctx, store); err != nil || named != 1 || !namedAs(store, twin.ID, []artifact.ID{shared.Lineage[0].Child}) {
		t.Fatalf("a charged part of a document and its copy = %d named (%v), and it contains their one name: %t; want it named", named, err, namedAs(store, twin.ID, []artifact.ID{shared.Lineage[0].Child}))
	}
	// Uncharged, a part that holds a document with no recorded content is passed by with no names, and so is the charged part of two records under one name.
	idle, _ := partThen(promoted.Records[1], unrecorded)
	grantedThen(store, poolUnit, idle.ID, 1, earlier.Acceptance.ID)
	if named, err := NameEarlierParts(ctx, store); err != nil || named != 0 || !namedAs(store, idle.ID, nil) {
		t.Errorf("an uncharged part that holds a document with no recorded content = %d named (%v), and it has no names: %t; want it passed by unnamed", named, err, namedAs(store, idle.ID, nil))
	}
	// Charged, it is refused, and nothing is named.
	mixed, some := partThen(promoted.Records[0], unrecorded)
	chargedThen(store, poolUnit, mixed.ID, 1, earlier.Acceptance.ID, earlier.Acceptance.Objective, "a confirmation of the part that holds a document with no recorded content")
	if named, err := NameEarlierParts(ctx, store); err == nil || !strings.Contains(err.Error(), "cannot follow") || named != 0 || !namedAs(store, mixed.ID, nil) {
		t.Errorf("a charged part that holds a document with no recorded content = %d named (%v), and it has no names: %t; want the naming refused and nothing named", named, err, namedAs(store, mixed.ID, nil))
	}
	// It is refused also when the names of its other documents were published before.
	if len(some.Lineage) != 1 {
		t.Fatalf("the fixture's part of a document with a content and one without is named for %d of them, want one", len(some.Lineage))
	}
	if _, err := artifact.Publish(ctx, store, some); err != nil {
		t.Fatal(err)
	}
	if named, err := NameEarlierParts(ctx, store); err == nil || !strings.Contains(err.Error(), "cannot follow") || named != 0 {
		t.Errorf("a charged part already named for one of its two documents = %d named (%v), want the naming refused", named, err)
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
	// The store's cases less one of the part's prompts name the others only.
	if promoted, found, err = dataset.LoadMembership(ctx, prompts, partition.Promotion); err != nil || !found {
		t.Fatalf("the acceptance's promotion part is not committed (%v)", err)
	}
	lacking := maps.Clone(contents)
	delete(lacking, promoted.Records[0].ID)
	fewer := func() (map[string]string, artifact.ID) { return lacking, held.Dataset }
	if named, err := nameEarlierParts(ctx, prompts, fewer); err == nil || !strings.Contains(err.Error(), "cannot follow") || named != 0 || !namedAs(prompts, partition.Promotion, nil) {
		t.Errorf("a charged acceptance one of whose prompts the store's cases do not name = %d named (%v), and its part has no names: %t; want the naming refused and nothing named", named, err, namedAs(prompts, partition.Promotion, nil))
	}
	if named, err := nameEarlierParts(ctx, prompts, theirs); err != nil || named != 1 || !namedAs(prompts, partition.Promotion, today) {
		t.Fatalf("a charged acceptance whose prompts are the store's cases = %d named (%v), and its part contains what an acceptance built today names: %t", named, err, namedAs(prompts, partition.Promotion, today))
	}
	// Named whole and charged, the part is followed by the names it has, whatever cases the store lists now.
	for name, cases := range map[string]func() (map[string]string, artifact.ID){"another dataset's": anothers, "its own": theirs, "its own less one": fewer} {
		if named, err := nameEarlierParts(ctx, prompts, cases); err != nil || named != 0 {
			t.Fatalf("a charged acceptance named whole, with the store's cases %s = %d named (%v), want it passed by", name, named, err)
		}
	}
}
