package composition

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/checked"
	"overgo/internal/tensor"
)

// ExternalCrossAttentionPlan is the recipe-bound adapter and target seam used
// by inference. It deliberately contains no target KV-cache authority.
type ExternalCrossAttentionPlan struct {
	Execution        artifact.ID `json:"execution"`
	SourceModel      artifact.ID `json:"source_model"`
	TargetModel      artifact.ID `json:"target_model"`
	Adapter          artifact.ID `json:"adapter"`
	Layer            uint32      `json:"layer"`
	SourceChannels   uint64      `json:"source_channels"`
	TargetChannels   uint64      `json:"target_channels"`
	HeadCount        uint64      `json:"head_count"`
	SourceTokenLimit uint64      `json:"source_token_limit"`
	ID               artifact.ID `json:"-"`
}

// CompileExternalCrossAttentionPlan binds an external-attention bridge to the
// exact active execution plan. Callers cannot supply another adapter or seam.
func CompileExternalCrossAttentionPlan(execution CompositionExecutionPlan, bridge BridgeDefinition) (ExternalCrossAttentionPlan, error) {
	if execution.ID.Kind() != artifact.KindProfile || len(execution.Operators) != tensor.SingletonExtent ||
		execution.Operators[tensor.FirstOffset] != bridgegraph.OperatorExternalAttention ||
		len(execution.BridgeDefinitions) != tensor.SingletonExtent || execution.BridgeDefinitions[tensor.FirstOffset] != bridge.ID ||
		len(execution.BridgeWeights) != tensor.SingletonExtent || execution.BridgeWeights[tensor.FirstOffset] != bridge.Weights ||
		bridge.SourceModel != execution.SourceModel || bridge.TargetModel != execution.TargetModel ||
		bridge.Graph.Operator != bridgegraph.OperatorExternalAttention ||
		bridge.Graph.Source != execution.SourceContract || bridge.Graph.Target != execution.TargetContract ||
		!checked.NonzeroAll(execution.Capture.Channels, execution.Injection.Channels) || execution.Injection.Layer == nil ||
		!checked.Nonzero(bridge.Graph.HeadCount) || !checked.Nonzero(bridge.Graph.SourceTokenLimit) {
		return ExternalCrossAttentionPlan{}, errors.New("composition: external cross-attention authority differs from execution plan")
	}
	value := ExternalCrossAttentionPlan{
		Execution: execution.ID, SourceModel: execution.SourceModel, TargetModel: execution.TargetModel,
		Adapter: bridge.Weights, Layer: *execution.Injection.Layer,
		SourceChannels: execution.Capture.Channels, TargetChannels: execution.Injection.Channels,
		HeadCount: bridge.Graph.HeadCount, SourceTokenLimit: bridge.Graph.SourceTokenLimit,
	}
	identity, err := artifact.JSONID(artifact.KindProfile, value)
	if err != nil {
		return ExternalCrossAttentionPlan{}, err
	}
	value.ID = identity
	return value, nil
}
