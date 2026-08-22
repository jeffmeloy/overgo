package composition

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/testutil"
)

func TestExternalCrossAttentionPlan(t *testing.T) {
	source := testutil.ArtifactID(t, artifact.KindModel, "external-plan-source")
	target := testutil.ArtifactID(t, artifact.KindModel, "external-plan-target")
	sourceContract := testutil.ArtifactID(t, artifact.KindProfile, "external-plan-source-contract")
	targetContract := testutil.ArtifactID(t, artifact.KindProfile, "external-plan-target-contract")
	definition := testutil.ArtifactID(t, artifact.KindProfile, "external-plan-definition")
	adapter := testutil.ArtifactID(t, artifact.KindAdapter, "external-plan-adapter")
	layer := uint32(4)
	execution := CompositionExecutionPlan{
		SourceModel: source, TargetModel: target, SourceContract: sourceContract, TargetContract: targetContract,
		BridgeDefinitions: []artifact.ID{definition}, BridgeWeights: []artifact.ID{adapter},
		Operators: []bridgegraph.Operator{bridgegraph.OperatorExternalAttention},
		Capture:   CompositionBoundary{Channels: 3}, Injection: CompositionBoundary{Channels: 6, Layer: &layer},
		ID: testutil.ArtifactID(t, artifact.KindProfile, "external-plan-execution"),
	}
	bridge := BridgeDefinition{
		SourceModel: source, TargetModel: target, Weights: adapter, ID: definition,
		Graph: bridgegraph.Definition{
			Source: sourceContract, Target: targetContract, Operator: bridgegraph.OperatorExternalAttention,
			HeadCount: 2, SourceTokenLimit: 16,
		},
	}
	plan, err := CompileExternalCrossAttentionPlan(execution, bridge)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Execution != execution.ID || plan.Adapter != adapter || plan.Layer != layer ||
		plan.SourceChannels != 3 || plan.TargetChannels != 6 || plan.HeadCount != 2 || plan.SourceTokenLimit != 16 {
		t.Fatalf("external plan = %+v", plan)
	}
	bridge.Weights = testutil.ArtifactID(t, artifact.KindAdapter, "other-adapter")
	if _, err := CompileExternalCrossAttentionPlan(execution, bridge); err == nil {
		t.Fatal("adapter outside the active recipe was accepted")
	}
}
