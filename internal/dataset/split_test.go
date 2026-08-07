package dataset

import (
	"context"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestGroupSplitIsDeterministicAndGroupSafe(t *testing.T) {
	source := fixtureID(t, artifact.KindDataset, "source")
	records := []Record{
		{ID: "a-2", Group: "a"}, {ID: "b-1", Group: "b"},
		{ID: "a-1", Group: "a"}, {ID: "c-1", Group: "c"},
		{ID: "d-1", Group: "d"}, {ID: "e-1", Group: "e"},
	}
	partitions := []SplitPartition{{Name: "train", Weight: 4}, {Name: "test", Weight: 1}}
	first, err := BuildGroupSplit(source, records, 42, partitions)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildGroupSplit(source, []Record{
		records[5], records[3], records[1], records[4], records[0], records[2],
	}, 42, []SplitPartition{partitions[1], partitions[0]})
	if err != nil {
		t.Fatal(err)
	}
	if first.Split.ID != second.Split.ID || !reflect.DeepEqual(first.Memberships, second.Memberships) {
		t.Fatal("same source and seed produced different split")
	}
	groupPartition := map[string]string{}
	for _, membership := range first.Memberships {
		content, err := membership.ContentBytes()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ParseMembership(content); err != nil {
			t.Fatal(err)
		}
		for _, record := range membership.Records {
			if prior, exists := groupPartition[record.Group]; exists && prior != membership.Partition {
				t.Fatalf("group %q straddled %q and %q", record.Group, prior, membership.Partition)
			}
			groupPartition[record.Group] = membership.Partition
		}
	}
}

func TestGroupSplitPublishesMembershipSelectors(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	source := fixtureID(t, artifact.KindDataset, "source")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/source", Artifacts: []artifact.Descriptor{{ID: source}},
	}); err != nil {
		t.Fatal(err)
	}
	plan, err := BuildGroupSplit(source, []Record{
		{ID: "a", Group: "a"}, {ID: "b", Group: "b"}, {ID: "c", Group: "c"},
	}, 7, []SplitPartition{{Name: "train", Weight: 2}, {Name: "test", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := plan.PublicationBatch("fixture/split", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	for _, membership := range plan.Memberships {
		if _, ok, err := store.Content(ctx, membership.ID); err != nil || !ok {
			t.Fatalf("membership content = (%t, %v)", ok, err)
		}
	}
}

func TestDuplicateLineageRequiresSameArtifactKind(t *testing.T) {
	duplicate := fixtureID(t, artifact.KindDatasetShard, "duplicate")
	canonical := fixtureID(t, artifact.KindDatasetShard, "canonical")
	edge, err := DuplicateLineage(duplicate, canonical)
	if err != nil || edge.Relation != artifact.RelationDuplicateOf {
		t.Fatalf("duplicate edge = (%+v, %v)", edge, err)
	}
	if _, err := DuplicateLineage(duplicate, fixtureID(t, artifact.KindFile, "file")); err == nil {
		t.Fatal("cross-kind duplicate accepted")
	}
}

func TestSplitPublicationRejectsCrossWiredPlan(t *testing.T) {
	source := fixtureID(t, artifact.KindDataset, "source")
	records := []Record{
		{ID: "a", Group: "a"}, {ID: "b", Group: "b"}, {ID: "c", Group: "c"},
	}
	partitions := []SplitPartition{{Name: "train", Weight: 2}, {Name: "test", Weight: 1}}
	first, err := BuildGroupSplit(source, records, 7, partitions)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildGroupSplit(source, records, 8, partitions)
	if err != nil {
		t.Fatal(err)
	}
	first.Memberships[0] = second.Memberships[0]
	if _, err := first.PublicationBatch("fixture/cross-wired", nil); err == nil {
		t.Fatal("cross-wired split plan accepted")
	}
}
