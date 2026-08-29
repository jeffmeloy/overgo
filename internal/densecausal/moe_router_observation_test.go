package densecausal

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestMoERouterObservationIdentityCoverageAndValidation(t *testing.T) {
	policy := MoERouterPolicy{TopK: 2, Scoring: MoEScoringSoftmax, NormalizeTopKProb: true, RoutedScaling: 1, ExpertInter: 1}
	route, scores, err := routeTopK([]float32{1, 2}, []float32{1, 0, 0, 1, -1, 0}, 1, 2, 3, policy)
	if err != nil {
		t.Fatal(err)
	}
	selected := make([]uint32, len(route.indices))
	accepted := make([]bool, len(route.indices))
	for index, expert := range route.indices {
		selected[index] = uint32(expert)
		accepted[index] = true
	}
	margin := scores[route.indices[policy.TopK-1]]
	for expert, score := range scores {
		chosen := false
		for _, selectedExpert := range route.indices {
			chosen = chosen || expert == selectedExpert
		}
		if !chosen {
			margin -= score
			break
		}
	}
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	observation, err := runrecord.NewMoERouterObservation(runrecord.MoERouterObservation{
		Run: id(artifact.KindRun, "run"), Model: id(artifact.KindModel, "model"), Dataset: id(artifact.KindDataset, "dataset"),
		Split: id(artifact.KindDatasetShard, "split"), Recipe: id(artifact.KindRecipe, "recipe"), Code: id(artifact.KindEvidence, "code"),
		Checkpoint: id(artifact.KindCheckpoint, "checkpoint"), Policy: id(artifact.KindRecipe, "policy"),
		Rows: 1, Experts: 3, TopK: 2, Selections: selected, CombineWeights: route.weights, Accepted: accepted,
		Margins: []runrecord.MoERouterMargin{{Observed: true, Value: margin}},
	})
	if err != nil || observation.Rows != 1 || len(observation.Selections) != policy.TopK {
		t.Fatalf("router observation = (%d, %d, %v)", observation.Rows, len(observation.Selections), err)
	}
}
