package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestCapacityEvaluationPairsConstrainedAndDroplessRuns(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	pair, err := moeCapacityPairCodec.New(moeCapacityPair{
		Version: artifact.InitialDocumentVersion,
		Model:   id(artifact.KindModel, "model"), Dataset: id(artifact.KindDataset, "dataset"),
		Split: id(artifact.KindDatasetShard, "split"), Checkpoint: id(artifact.KindCheckpoint, "checkpoint"),
		Seed: id(artifact.KindEvidence, "seed"), Code: id(artifact.KindEvidence, "code"),
		Environment: id(artifact.KindEvidence, "environment"), Evaluator: id(artifact.KindProfile, "evaluator"),
		Constrained: moeCapacityEndpoint{
			Run: id(artifact.KindRun, "constrained-run"), Policy: id(artifact.KindRecipe, "capacity-policy"),
			Evaluation: id(artifact.KindEvidence, "constrained-evaluation"), Dropped: 3, Total: 20,
		},
		Dropless: moeCapacityEndpoint{
			Run: id(artifact.KindRun, "dropless-run"), Policy: id(artifact.KindRecipe, "dropless-policy"),
			Evaluation: id(artifact.KindEvidence, "dropless-evaluation"), Total: 20, Admitted: true,
		},
		DropQualityGap: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := moeCapacityPairCodec.Content(pair)
	if err != nil || content.Descriptor.ID != pair.ID || len(moeCapacityPairLineage(pair)) != 14 {
		t.Fatalf("capacity pair = (%+v, %v)", pair, err)
	}
	wrong := pair
	wrong.ID, wrong.DropQualityGap = artifact.ID{}, false
	if _, err := moeCapacityPairCodec.New(wrong); err == nil {
		t.Fatal("quality gap hidden")
	}
	notDropless := pair
	notDropless.ID, notDropless.Dropless.Dropped = artifact.ID{}, 1
	if _, err := moeCapacityPairCodec.New(notDropless); err == nil {
		t.Fatal("dropped-token reference admitted as dropless")
	}
	unpaired := pair
	unpaired.ID, unpaired.Dropless.Run = artifact.ID{}, pair.Constrained.Run
	if _, err := moeCapacityPairCodec.New(unpaired); err == nil {
		t.Fatal("same run admitted on both sides")
	}
	unmatched := pair
	unmatched.ID, unmatched.Dropless.Total = artifact.ID{}, pair.Constrained.Total+1
	if _, err := moeCapacityPairCodec.New(unmatched); err == nil {
		t.Fatal("different token population admitted as a pair")
	}
}
