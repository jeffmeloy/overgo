package repodb

import (
	"context"
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestQueryFiltersAndFollowsImmutableCatalog(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	tensorID := testutil.ArtifactID(t, artifact.KindTensorSet, "tensor")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "dataset")
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
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "output")
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
		Kind: artifact.KindModel, MaxResults: 10, Projection: ProjectCatalog,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(models.Artifacts) != 1 || models.Artifacts[0].ID != manifest.ID ||
		len(models.Manifests) != 1 || len(models.Aliases) != 1 {
		t.Fatalf("model query = %+v", models)
	}

	followed, err := store.Query(context.Background(), Query{
		Alias: "model/active", Follow: FollowBoth, MaxDepth: 2, MaxResults: 10, Projection: ProjectCatalog,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(followed.Artifacts) != 3 || len(followed.Lineage) != 2 || followed.Truncated {
		t.Fatalf("follow query = %+v", followed)
	}

	produced, err := store.Query(context.Background(), Query{
		Artifact: &manifest.ID, Relation: artifact.RelationProducedBy,
		Follow: FollowChildren, MaxDepth: 1, MaxResults: 10, Projection: ProjectCatalog,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(produced.Artifacts) != 2 || len(produced.Lineage) != 1 || produced.Lineage[0].Child != outputID {
		t.Fatalf("relation query = %+v", produced)
	}

	commits, err := store.Query(context.Background(), Query{
		FromSequence: 2, ToSequence: 2, MaxResults: 10, Projection: ProjectCatalog,
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
	parent := testutil.ArtifactID(t, artifact.KindEvidence, "parent")
	child := testutil.ArtifactID(t, artifact.KindEvidence, "child")
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
	result, err := store.Query(context.Background(), Query{MaxResults: 1, Projection: ProjectArtifacts})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || !result.Truncated {
		t.Fatalf("bounded query = %+v", result)
	}
	if _, err := store.Query(context.Background(), Query{MaxResults: 1}); err == nil {
		t.Fatal("projection-free query accepted")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Query(cancelled, Query{MaxResults: 1, Projection: ProjectArtifacts}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled query error = %v", err)
	}
}

func TestServingObservationDescriptorIndexes(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const (
		servingMedia  = "application/vnd.overgo.serving-observation+json"
		servingSchema = "overgo/serving-observation/v1"
		otherSchema   = "overgo/other/v1"
	)
	serving := fixtureDescriptor(t, artifact.KindEvidence, "serving-observation")
	serving.MediaType, serving.Schema = servingMedia, servingSchema
	other := fixtureDescriptor(t, artifact.KindEvidence, "other-observation")
	other.MediaType, other.Schema = servingMedia, otherSchema
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/query/descriptor-indexes", Artifacts: []artifact.Descriptor{serving, other},
	}); err != nil {
		t.Fatal(err)
	}
	result, err := store.Query(context.Background(), Query{
		Kind: artifact.KindEvidence, MediaType: servingMedia, Schema: servingSchema, MaxResults: 2,
		Projection: ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0] != serving || result.Truncated {
		t.Fatalf("descriptor query = %+v", result)
	}
	if _, err := store.Query(context.Background(), Query{MediaType: " invalid", MaxResults: 1, Projection: ProjectArtifacts}); err == nil {
		t.Fatal("invalid media filter accepted")
	}
}

func TestQueryCursorBindsHeadAndContract(t *testing.T) {
	const pageSize = 2
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	descriptors := []artifact.Descriptor{
		fixtureDescriptor(t, artifact.KindEvidence, "cursor-a"),
		fixtureDescriptor(t, artifact.KindEvidence, "cursor-b"),
		fixtureDescriptor(t, artifact.KindEvidence, "cursor-c"),
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{Key: "fixture/query/cursor", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	query := Query{Kind: artifact.KindEvidence, MaxResults: pageSize, Projection: ProjectArtifacts}
	first, err := store.Query(context.Background(), query)
	if err != nil || len(first.Artifacts) != pageSize || first.Next == nil {
		t.Fatalf("first page = (%+v, %v)", first, err)
	}
	query.Cursor = first.Next
	second, err := store.Query(context.Background(), query)
	if err != nil || len(second.Artifacts) != len(descriptors)-pageSize || second.Next != nil {
		t.Fatalf("second page = (%+v, %v)", second, err)
	}
	query.Kind = artifact.KindRun
	if _, err := store.Query(context.Background(), query); err == nil {
		t.Fatal("cursor accepted for another query contract")
	}
	query.Kind = artifact.KindEvidence
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/query/cursor-tail",
		Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindEvidence, "cursor-tail")},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Query(context.Background(), query); err == nil {
		t.Fatal("cursor accepted after catalog head changed")
	}
}

func TestQueryProjectionReturnsRequestedFactsOnly(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	content := artifact.Content{
		Descriptor: fixtureDescriptor(t, artifact.KindEvidence, fixturePayload),
		Data:       []byte(fixturePayload),
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{Key: "fixture/query/projection", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	result, err := store.Query(context.Background(), Query{
		Artifact: &content.Descriptor.ID, MaxResults: 1, Projection: ProjectContentData,
	})
	if err != nil || len(result.Artifacts) != 0 || len(result.Contents) != 1 ||
		result.Contents[0].Artifact != content.Descriptor.ID || !slices.Equal(result.Contents[0].Data, content.Data) {
		t.Fatalf("projected content = (%+v, %v)", result, err)
	}
}
