package repodb

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
)

func TestQueryFiltersAndFollowsImmutableCatalog(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	tensorID, _ := artifact.IdentifyBytes(artifact.KindTensorSet, []byte("tensor"))
	datasetID, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte("dataset"))
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: tensorID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/query/base",
		Artifacts: []artifact.Descriptor{
			{ID: tensorID, Size: 6}, {ID: datasetID, Size: 7},
		},
		Manifests: []artifact.Manifest{manifest},
		Aliases:   []artifact.AliasBinding{{Name: "model/active", Target: manifest.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	outputID, _ := artifact.IdentifyBytes(artifact.KindOutput, []byte("output"))
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/query/output",
		Artifacts: []artifact.Descriptor{{ID: outputID, Size: 6}},
		Lineage: []artifact.Lineage{{
			Child: outputID, Parent: manifest.ID, Relation: artifact.RelationProducedBy,
		}},
	}); err != nil {
		t.Fatal(err)
	}

	models, err := store.Query(context.Background(), Query{
		Kind: artifact.KindModel, MaxResults: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(models.Artifacts) != 1 || models.Artifacts[0].ID != manifest.ID ||
		len(models.Manifests) != 1 || len(models.Aliases) != 1 {
		t.Fatalf("model query = %+v", models)
	}

	followed, err := store.Query(context.Background(), Query{
		Alias: "model/active", Follow: FollowBoth, MaxDepth: 2, MaxResults: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(followed.Artifacts) != 3 || len(followed.Lineage) != 2 || followed.Truncated {
		t.Fatalf("follow query = %+v", followed)
	}

	produced, err := store.Query(context.Background(), Query{
		Artifact: &manifest.ID, Relation: artifact.RelationProducedBy,
		Follow: FollowChildren, MaxDepth: 1, MaxResults: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(produced.Artifacts) != 2 || len(produced.Lineage) != 1 || produced.Lineage[0].Child != outputID {
		t.Fatalf("relation query = %+v", produced)
	}

	commits, err := store.Query(context.Background(), Query{
		FromSequence: 2, ToSequence: 2, MaxResults: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(commits.Commits) != 1 || commits.Commits[0].Key != "fixture/query/output" || len(commits.Artifacts) != 0 {
		t.Fatalf("commit query = %+v", commits)
	}
}

func TestQueryBoundsAndCycleRejection(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parent, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("parent"))
	child, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("child"))
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/query/cycle-base",
		Artifacts: []artifact.Descriptor{{ID: parent, Size: 6}, {ID: child, Size: 5}},
		Lineage: []artifact.Lineage{{
			Child: child, Parent: parent, Relation: artifact.RelationDerivedFrom,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/query/cycle-attempt",
		Lineage: []artifact.Lineage{{
			Child: parent, Parent: child, Relation: artifact.RelationDerivedFrom,
		}},
	}); !errors.Is(err, ErrLineageCycle) {
		t.Fatalf("cycle error = %v", err)
	}
	result, err := store.Query(context.Background(), Query{MaxResults: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || !result.Truncated {
		t.Fatalf("bounded query = %+v", result)
	}
	if _, err := store.Query(context.Background(), Query{MaxResults: MaxQueryResults + 1}); err == nil {
		t.Fatal("unbounded query accepted")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Query(cancelled, Query{MaxResults: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled query error = %v", err)
	}
}
