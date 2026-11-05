package evaluation

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestOfflineArtifactGenerationRepeatedSeeds(t *testing.T) {
	t.Parallel()
	evidence := offlinePromotionEvidence(t, false)
	policy := offlinePromotionPolicy(t)
	promotion, err := PromoteOfflineArtifactGeneration(evidence, policy)
	testcheck.NoError(t, err)
	if promotion.ExecutionPlan != evidence.ExecutionPlan || promotion.GenerationEvidence != evidence.ID ||
		promotion.DistinctSeeds != policy.MinimumDistinctSeeds || promotion.RunCount != uint32(len(evidence.Trials)) {
		t.Fatalf("offline generation promotion = %+v", promotion)
	}
}

func TestOfflineArtifactGenerationPromotionRefusal(t *testing.T) {
	t.Parallel()
	policy := offlinePromotionPolicy(t)
	unstable := offlinePromotionEvidence(t, true)
	if _, err := PromoteOfflineArtifactGeneration(unstable, policy); err == nil {
		t.Fatal("unstable repeated seed promoted")
	}
	insufficient := offlinePromotionEvidence(t, false)
	insufficient.Trials = insufficient.Trials[:tensor.PairedExtent]
	data, err := json.Marshal(insufficient)
	testcheck.NoError(t, err)
	insufficient.ID, err = artifact.IdentifyBytes(artifact.KindEvidence, data)
	testcheck.NoError(t, err)
	if _, err := PromoteOfflineArtifactGeneration(insufficient, policy); err == nil {
		t.Fatal("insufficient seed coverage promoted")
	}
	strict, err := NewOfflineArtifactGenerationPolicy(
		policy.MinimumDistinctSeeds, policy.RunsPerSeed, 1, policy.MaximumPeakHostBytes,
		"strict measured latency envelope", "re-evaluate after hardware or runtime changes",
	)
	testcheck.NoError(t, err)
	if _, err := PromoteOfflineArtifactGeneration(offlinePromotionEvidence(t, false), strict); err == nil {
		t.Fatal("resource regression promoted")
	}
}

func offlinePromotionPolicy(t *testing.T) offlineArtifactGenerationPolicy {
	t.Helper()
	policy, err := NewOfflineArtifactGenerationPolicy(
		2, 2, 1_000, 2_000,
		"two repeated seeds bound deterministic generation and measured host resources",
		"re-evaluate when generation, hardware, or the execution recipe changes",
	)
	testcheck.NoError(t, err)
	return policy
}

func offlinePromotionEvidence(t *testing.T, unstable bool) composition.OfflineArtifactGenerationEvidence {
	t.Helper()
	model := testutil.ArtifactID(t, artifact.KindModel, "offline promotion model")
	plan := offlinePromotionPlan(t, model)
	directory := t.TempDir()
	if err := safetensors.Save(
		filepath.Join(directory, "model.safetensors"),
		map[string][]float32{"weight": {1, 2}}, map[string][]int{"weight": {2}}, nil,
	); err != nil {
		t.Fatal(err)
	}
	first := testutil.ArtifactID(t, artifact.KindOutput, "offline promotion seed first")
	second := testutil.ArtifactID(t, artifact.KindOutput, "offline promotion seed second")
	changed := first
	if unstable {
		changed = testutil.ArtifactID(t, artifact.KindOutput, "offline promotion unstable")
	}
	trials := []composition.OfflineArtifactGenerationTrial{
		offlinePromotionTrial(t, 11, first, "a"),
		offlinePromotionTrial(t, 11, changed, "b"),
		offlinePromotionTrial(t, 29, second, "c"),
		offlinePromotionTrial(t, 29, second, "d"),
	}
	evidence, err := composition.ValidateOfflineArtifactGeneration(plan, directory, model, trials)
	testcheck.NoError(t, err)
	return evidence
}

func offlinePromotionTrial(t *testing.T, seed uint64, output artifact.ID, suffix string) composition.OfflineArtifactGenerationTrial {
	t.Helper()
	return composition.OfflineArtifactGenerationTrial{
		Seed: seed, Output: output,
		Run:         testutil.ArtifactID(t, artifact.KindRun, "offline promotion run "+suffix),
		Observation: testutil.ArtifactID(t, artifact.KindEvidence, "offline promotion observation "+suffix),
		LatencyNS:   100, PeakHostBytes: 200,
	}
}

func offlinePromotionPlan(t *testing.T, model artifact.ID) composition.OfflineTensorExecutionPlan {
	t.Helper()
	bytes, err := safetensors.TensorBytes("F32", []uint64{2})
	testcheck.NoError(t, err)
	plan := composition.OfflineTensorExecutionPlan{
		Version:        artifact.InitialDocumentVersion,
		ArtifactPlan:   testutil.ArtifactID(t, artifact.KindRecipe, "offline promotion artifact plan"),
		ResourcePolicy: testutil.ArtifactID(t, artifact.KindProfile, "offline promotion resource policy"),
		Operator:       composition.OfflineArtifactExactPassthrough,
		Inputs: []composition.OfflineArtifactModel{{
			Model:       model,
			Definition:  testutil.ArtifactID(t, artifact.KindModelDefinition, "offline promotion definition"),
			Profile:     testutil.ArtifactID(t, artifact.KindProfile, "offline promotion profile"),
			Inventory:   testutil.ArtifactID(t, artifact.KindTensorInventory, "offline promotion inventory"),
			Coefficient: 1,
		}},
		Operations: []composition.OfflineTensorOperation{{
			Index: 0, Name: "weight", Storage: "f32", SourceBytes: []uint64{bytes},
			OutputBytes: bytes, ResidentBytes: bytes, Placement: recipe.PlacementHost,
			Lifetime: composition.OfflineTensorLifetime{First: 0, Last: 0}, Shard: 0,
		}},
		Shards:            []composition.OfflineTensorShard{{Index: 0, Bytes: bytes, Tensors: []string{"weight"}}},
		PeakResidentBytes: bytes,
	}
	data, err := json.Marshal(plan)
	testcheck.NoError(t, err)
	plan.ID, err = artifact.IdentifyBytes(artifact.KindProfile, data)
	testcheck.NoError(t, err)
	testcheck.NoError(t, plan.ValidateIdentity())
	return plan
}
