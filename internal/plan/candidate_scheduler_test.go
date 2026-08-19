package plan

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestSchedulerRankedByRealizedDelta(t *testing.T) {
	const (
		gibibyte        = uint64(1 << 30)
		budgetVRAM      = 8 * gibibyte
		candidateVRAM   = 6 * gibibyte
		oversizedVRAM   = 10 * gibibyte
		candidateBudget = 3 * time.Hour
	)
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	slowStrategy := id(artifact.KindProfile, "slow strategy")
	fastStrategy := id(artifact.KindProfile, "fast strategy")
	slowCandidate := id(artifact.KindModelDefinition, "slow candidate")
	fastCandidate := id(artifact.KindModelDefinition, "fast candidate")
	oversizedCandidate := id(artifact.KindModelDefinition, "oversized candidate")
	changes := func(quality, loss float64) []CapabilityChange {
		return []CapabilityChange{
			{Name: "quality", Before: 0, After: quality, Direction: runrecord.DirectionMaximize},
			{Name: "loss", Before: 1, After: loss, Direction: runrecord.DirectionMinimize},
		}
	}
	outcomes := []CandidateOutcome{
		{
			Evidence: id(artifact.KindEvidence, "slow outcome"), Candidate: slowCandidate, Strategy: slowStrategy,
			WallNS: uint64(2 * time.Hour), PeakDeviceBytes: candidateVRAM, Changes: changes(1, 0.5),
		},
		{
			Evidence: id(artifact.KindEvidence, "fast outcome"), Candidate: fastCandidate, Strategy: fastStrategy,
			WallNS: uint64(30 * time.Minute), PeakDeviceBytes: candidateVRAM, Changes: changes(0.75, 0.5),
		},
	}
	candidates := []ScheduledCandidate{
		{Candidate: slowCandidate, Strategy: slowStrategy, PredictedWallNS: uint64(time.Hour), PredictedVRAM: candidateVRAM},
		{Candidate: fastCandidate, Strategy: fastStrategy, PredictedWallNS: uint64(time.Hour), PredictedVRAM: candidateVRAM},
		{Candidate: oversizedCandidate, Strategy: fastStrategy, PredictedWallNS: uint64(time.Hour), PredictedVRAM: oversizedVRAM},
	}
	scheduler, err := CompileCandidateScheduler(outcomes, candidates, SchedulerBudget{
		WallNS: uint64(candidateBudget), VRAMBytes: budgetVRAM,
	})
	if err != nil {
		t.Fatal(err)
	}
	if scheduler.Selected != fastCandidate || len(scheduler.Ranking) != 2 || scheduler.Ranking[1] != slowCandidate {
		t.Fatalf("scheduler ranking = %v; selected=%s", scheduler.Ranking, scheduler.Selected)
	}
	if len(scheduler.EvidenceRows) != 2 || scheduler.EvidenceRows[0].Strategy == scheduler.EvidenceRows[1].Strategy {
		t.Fatalf("scheduler evidence rows = %+v", scheduler.EvidenceRows)
	}
	replay, err := CompileCandidateScheduler(outcomes, candidates, scheduler.Budget)
	if err != nil || replay.ID != scheduler.ID {
		t.Fatalf("scheduler replay = %s, %v; want %s", replay.ID, err, scheduler.ID)
	}
	content, err := scheduler.Content()
	if err != nil || content.Descriptor.ID != scheduler.ID || len(scheduler.Lineage()) < len(outcomes) {
		t.Fatalf("scheduler artifact = (%s, %d, %v)", content.Descriptor.ID, len(scheduler.Lineage()), err)
	}
}
