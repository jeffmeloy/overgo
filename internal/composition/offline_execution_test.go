package composition

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestOfflineTensorExecutionPlan(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	base := publishOfflineModel(t, store, "execution-base", offlineTensorWidth, false)
	trained := publishOfflineModel(t, store, "execution-trained", offlineTensorWidth, false)
	artifactPlan := publishOfflineArtifactPlan(t, store, OfflineArtifactTaskArithmetic, []OfflineArtifactInput{
		{Definition: base, Coefficient: 1},
		{Definition: trained, Coefficient: 1},
	})
	tensorBytes := offlineTensorWidth * offlineTensorWidth * offlineFloatBytes
	policy := publishOfflineTensorPolicy(t, store, tensorBytes*uint64(tensor.PairedExtent), tensorBytes)

	execution, err := CompileOfflineTensorExecutionPlan(ctx, store, artifactPlan, policy)
	if err != nil {
		t.Fatal(err)
	}
	if execution.Operator != OfflineArtifactTaskArithmetic || execution.ArtifactPlan != artifactPlan.ID ||
		execution.ResourcePolicy != policy.ID || execution.PeakResidentBytes != tensorBytes*uint64(tensor.PairedExtent) ||
		len(execution.Operations) != tensor.SingletonExtent || len(execution.Shards) != tensor.SingletonExtent {
		t.Fatalf("execution plan = %+v", execution)
	}
	operation := execution.Operations[tensor.FirstOffset]
	if operation.Name != "token_embd.weight" || operation.OutputBytes != tensorBytes ||
		operation.ResidentBytes != execution.PeakResidentBytes || operation.Placement != recipe.PlacementHost ||
		operation.Lifetime.First != operation.Index || operation.Lifetime.Last != operation.Index ||
		len(operation.SourceBytes) != tensor.PairedExtent {
		t.Fatalf("operation = %+v", operation)
	}
	batch, err := execution.Batch("composition/offline/execution")
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Contents) != tensor.SingletonExtent || len(batch.Lineage) < tensor.PairedExtent {
		t.Fatalf("execution publication = contents %d lineage %d", len(batch.Contents), len(batch.Lineage))
	}
}

func TestOfflinePlanDerivesResourcePolicy(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	definition := publishOfflineModel(t, store, "resource-derived", offlineTensorWidth, false)
	artifactPlan := publishOfflineArtifactPlan(t, store, OfflineArtifactExactPassthrough, []OfflineArtifactInput{
		{Definition: definition, Coefficient: 1},
	})
	tensorBytes := offlineTensorWidth * offlineTensorWidth * offlineFloatBytes
	policy := publishOfflineTensorPolicy(t, store, tensorBytes, tensorBytes)
	execution, err := CompileOfflineTensorExecutionPlan(t.Context(), store, artifactPlan, policy)
	if err != nil {
		t.Fatal(err)
	}
	operation := execution.Operations[tensor.FirstOffset]
	if execution.PeakResidentBytes != tensorBytes || operation.ResidentBytes != tensorBytes ||
		execution.Shards[tensor.FirstOffset].Bytes != tensorBytes {
		t.Fatalf("derived extents = peak %d operation %d shard %d", execution.PeakResidentBytes,
			operation.ResidentBytes, execution.Shards[tensor.FirstOffset].Bytes)
	}
}

func TestOfflinePlanRejectsLiteralPolicy(t *testing.T) {
	if _, err := NewOfflineTensorResourcePolicy(recipe.PlacementHost, 1, 1, "", ""); err == nil {
		t.Fatal("accepted numeric resource limits without a recorded rationale and reopen trigger")
	}
	if _, err := NewOfflineTensorResourcePolicy(recipe.PlacementDevice, 1, 1,
		"device requested", "host executor gains device support"); err == nil {
		t.Fatal("accepted an unsupported placement")
	}

	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	definition := publishOfflineModel(t, store, "resource-refusal", offlineTensorWidth, false)
	artifactPlan := publishOfflineArtifactPlan(t, store, OfflineArtifactExactPassthrough, []OfflineArtifactInput{
		{Definition: definition, Coefficient: 1},
	})
	policy := publishOfflineTensorPolicy(t, store, 1, 1)
	if _, err := CompileOfflineTensorExecutionPlan(t.Context(), store, artifactPlan, policy); err == nil ||
		!strings.Contains(err.Error(), "resident policy") {
		t.Fatalf("undersized typed policy accepted: %v", err)
	}
	missing := policy
	missing.ID = testutil.ArtifactID(t, artifact.KindProfile, "missing offline tensor policy")
	if _, err := CompileOfflineTensorExecutionPlan(t.Context(), store, artifactPlan, missing); err == nil ||
		!strings.Contains(err.Error(), "resource policy") {
		t.Fatalf("mismatched policy identity accepted: %v", err)
	}
}

func publishOfflineArtifactPlan(
	t *testing.T,
	store *repodb.Store,
	operator OfflineArtifactOperator,
	inputs []OfflineArtifactInput,
) OfflineArtifactPlan {
	t.Helper()
	plan, err := CompileOfflineArtifactPlan(t.Context(), store, operator, inputs)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := plan.Batch("composition/offline/artifact-plan")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return plan
}

func publishOfflineTensorPolicy(
	t *testing.T,
	store *repodb.Store,
	maxResidentBytes, maxShardBytes uint64,
) OfflineTensorResourcePolicy {
	t.Helper()
	policy, err := NewOfflineTensorResourcePolicy(
		recipe.PlacementHost, maxResidentBytes, maxShardBytes,
		"bounded by the selected host-memory resource decision",
		"recompile when the selected memory budget or artifact inventory changes",
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := policy.Batch("composition/offline/resource-policy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return policy
}
