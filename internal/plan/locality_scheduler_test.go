package plan

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestScheduleByArtifactLocality(t *testing.T) {
	ctx := context.Background()
	repository, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	descriptor := func(payload string) artifact.Descriptor {
		id, err := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte(payload))
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Descriptor{ID: id, Size: uint64(len(payload))}
	}
	first := descriptor("first")
	second := descriptor("second is larger")
	third := descriptor("third is largest of these artifacts")
	location := func(value string, descriptor artifact.Descriptor) artifact.Location {
		return artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationRemote, Value: value}
	}
	firstLocation := location("store-a/first", first)
	secondLocation := location("store-b/second", second)
	thirdLocation := location("store-a/third", third)
	if _, err := repository.Commit(ctx, artifact.Batch{
		Key:       "fixture/locality/artifacts",
		Artifacts: []artifact.Descriptor{first, second, third},
		Locations: []artifact.LocationEvent{
			{Location: firstLocation, Action: artifact.LocationAdd},
			{Location: secondLocation, Action: artifact.LocationAdd},
			{Location: thirdLocation, Action: artifact.LocationAdd},
		},
	}); err != nil {
		t.Fatal(err)
	}
	workerA := testutil.ArtifactID(t, artifact.KindEvidence, "locality worker a")
	workerB := testutil.ArtifactID(t, artifact.KindEvidence, "locality worker b")
	workerStale := testutil.ArtifactID(t, artifact.KindEvidence, "locality stale worker")
	workers := []LocalityWorker{
		{Worker: workerA, Locations: []artifact.Location{firstLocation, thirdLocation}},
		{Worker: workerB, Locations: []artifact.Location{firstLocation, secondLocation}},
		{Worker: workerStale, Locations: []artifact.Location{location("retired/second", second)}},
	}
	requirements := []ArtifactRequirement{
		{Artifact: first.ID, Constraint: LocalityPreferred},
		{Artifact: second.ID, Constraint: LocalityPreferred},
		{Artifact: third.ID, Constraint: LocalityPreferred},
	}
	preferred, err := ScheduleByArtifactLocality(ctx, repository, LocalityScheduleRequest{
		Requirements: requirements, Workers: workers,
	})
	if err != nil {
		t.Fatal(err)
	}
	if preferred.Selected != workerA {
		t.Fatalf("preferred worker = %s, want %s; assessments=%+v", preferred.Selected, workerA, preferred.Assessments)
	}
	reversedRequirements, reversedWorkers := slices.Clone(requirements), slices.Clone(workers)
	slices.Reverse(reversedRequirements)
	slices.Reverse(reversedWorkers)
	replayed, err := ScheduleByArtifactLocality(ctx, repository, LocalityScheduleRequest{
		Requirements: reversedRequirements, Workers: reversedWorkers,
	})
	if err != nil || !reflect.DeepEqual(replayed, preferred) {
		t.Fatalf("order-dependent schedule = (%+v, %v), want %+v", replayed, err, preferred)
	}

	hardRequirements := slices.Clone(requirements)
	for index := range hardRequirements {
		if hardRequirements[index].Artifact == second.ID {
			hardRequirements[index].Constraint = LocalityRequired
		}
	}
	hard, err := ScheduleByArtifactLocality(ctx, repository, LocalityScheduleRequest{
		Requirements: hardRequirements, Workers: workers,
	})
	if err != nil || hard.Selected != workerB {
		t.Fatalf("hard-local schedule = (%+v, %v), want %s", hard, err, workerB)
	}
	for _, assessment := range hard.Assessments {
		if assessment.Worker == workerA &&
			(assessment.Eligibility != LocalityMissingRequired || !slices.Equal(assessment.MissingRequired, []artifact.ID{second.ID})) {
			t.Fatalf("worker A rejection is not explained: %+v", assessment)
		}
	}

	tie, err := ScheduleByArtifactLocality(ctx, repository, LocalityScheduleRequest{
		Requirements: requirements[:1],
		Workers: []LocalityWorker{
			{Worker: workerB, Locations: []artifact.Location{firstLocation}},
			{Worker: workerA, Locations: []artifact.Location{firstLocation}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantTie := workerA
	if compareArtifactID(workerB, workerA) < 0 {
		wantTie = workerB
	}
	if tie.Selected != wantTie {
		t.Fatalf("tie selected %s, want stable identity %s", tie.Selected, wantTie)
	}
}
