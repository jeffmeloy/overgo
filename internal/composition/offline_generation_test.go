package composition

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestOfflineArtifactGenerationExactPassthrough(t *testing.T) {
	plan := offlineGenerationPlan(t, OfflineArtifactExactPassthrough)
	directory := offlineGenerationDirectory(t)
	evidence := testcheck.Must(ValidateOfflineArtifactGeneration(
		plan, directory, testutil.ArtifactID(t, artifact.KindModel, "generated passthrough"),
		offlineGenerationTrials(t),
	))(t)
	if evidence.Operator != OfflineArtifactExactPassthrough || evidence.ExecutionPlan != plan.ID ||
		evidence.TensorCount != uint32(len(plan.Operations)) || evidence.TensorBytes != plan.Operations[0].OutputBytes {
		t.Fatalf("passthrough generation evidence = %+v", evidence)
	}
}

func TestOfflineArtifactGenerationTaskArithmetic(t *testing.T) {
	plan := offlineGenerationPlan(t, OfflineArtifactTaskArithmetic)
	directory := offlineGenerationDirectory(t)
	evidence := testcheck.Must(ValidateOfflineArtifactGeneration(
		plan, directory, testutil.ArtifactID(t, artifact.KindModel, "generated arithmetic"),
		offlineGenerationTrials(t),
	))(t)
	if evidence.Operator != OfflineArtifactTaskArithmetic || len(evidence.Trials) != tensor.PairedExtent {
		t.Fatalf("task arithmetic generation evidence = %+v", evidence)
	}
	broken := plan
	broken.Operations = append([]OfflineTensorOperation(nil), plan.Operations...)
	broken.Operations[0].OutputBytes++
	if _, err := ValidateOfflineArtifactGeneration(broken, directory, evidence.ProducedModel, evidence.Trials); err == nil {
		t.Fatal("plan identity drift accepted")
	}
}

func offlineGenerationPlan(t *testing.T, operator OfflineArtifactOperator) OfflineTensorExecutionPlan {
	t.Helper()
	modelCount := tensor.SingletonExtent
	if operator == OfflineArtifactTaskArithmetic {
		modelCount = tensor.PairedExtent
	}
	inputs := make([]OfflineArtifactModel, modelCount)
	for index := range inputs {
		inputs[index] = OfflineArtifactModel{
			Model:       testutil.ArtifactID(t, artifact.KindModel, "offline generation input model "+string(rune('a'+index))),
			Definition:  testutil.ArtifactID(t, artifact.KindModelDefinition, "offline generation input definition "+string(rune('a'+index))),
			Profile:     testutil.ArtifactID(t, artifact.KindProfile, "offline generation input profile "+string(rune('a'+index))),
			Inventory:   testutil.ArtifactID(t, artifact.KindTensorInventory, "offline generation input inventory "+string(rune('a'+index))),
			Coefficient: 1,
		}
	}
	bytes := testcheck.Must(safetensors.TensorBytes("F32", []uint64{2}))(t)
	resident := bytes * uint64(modelCount)
	sourceBytes := make([]uint64, modelCount)
	for index := range sourceBytes {
		sourceBytes[index] = bytes
	}
	return mustOfflineExecutionPlan(t, OfflineTensorExecutionPlan{
		Version:        offlineTensorExecutionPlanVersion,
		ArtifactPlan:   testutil.ArtifactID(t, artifact.KindRecipe, "offline generation artifact plan"),
		ResourcePolicy: testutil.ArtifactID(t, artifact.KindProfile, "offline generation resource policy"),
		Operator:       operator, Inputs: inputs,
		Operations: []OfflineTensorOperation{{
			Index: 0, Name: "weight", Storage: "f32", SourceBytes: sourceBytes,
			OutputBytes: bytes, ResidentBytes: resident, Placement: recipe.PlacementHost,
			Lifetime: OfflineTensorLifetime{First: 0, Last: 0}, Shard: 0,
		}},
		Shards:            []OfflineTensorShard{{Index: 0, Bytes: bytes, Tensors: []string{"weight"}}},
		PeakResidentBytes: resident,
	})
}

func mustOfflineExecutionPlan(t *testing.T, value OfflineTensorExecutionPlan) OfflineTensorExecutionPlan {
	t.Helper()
	return testcheck.Must(offlineTensorExecutionPlanCodec.New(value))(t)
}

func offlineGenerationDirectory(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	if err := safetensors.Save(
		filepath.Join(directory, "model.safetensors"),
		map[string][]float32{"weight": {1, 2}}, map[string][]int{"weight": {2}}, nil,
	); err != nil {
		t.Fatal(err)
	}
	return directory
}

func offlineGenerationTrials(t *testing.T) []OfflineArtifactGenerationTrial {
	t.Helper()
	output := testutil.ArtifactID(t, artifact.KindOutput, "offline generation output")
	return []OfflineArtifactGenerationTrial{
		{Seed: 7, Output: output, Run: testutil.ArtifactID(t, artifact.KindRun, "offline generation run a"), Observation: testutil.ArtifactID(t, artifact.KindEvidence, "offline generation observation a"), LatencyNS: 100, PeakHostBytes: 200},
		{Seed: 7, Output: output, Run: testutil.ArtifactID(t, artifact.KindRun, "offline generation run b"), Observation: testutil.ArtifactID(t, artifact.KindEvidence, "offline generation observation b"), LatencyNS: 110, PeakHostBytes: 210},
	}
}
