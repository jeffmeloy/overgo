package model

import (
	"errors"

	"overgo/internal/tensor"
)

// buildGatedHyperMix reduces normalized residual streams with learned sigmoid gates.
// GGUF normalization weights already include the original model's unit offset.
func buildGatedHyperMix(builder *tensor.Builder, input, norm, down, up, inject *tensor.Tensor, spec Spec) (*tensor.Tensor, *tensor.Tensor, error) {
	if err := (graphWeights{input, norm, down, up}).validate("gated hyperconnection mix"); err != nil {
		return nil, nil, err
	}
	width, streams := uint64(spec.EmbeddingLength), uint64(spec.HyperConnectionCount)
	if input.Shape.Rank != tensor.TripleExtent || input.Shape.Dims[0] != width || input.Shape.Dims[1] != streams || streams < tensor.PairedExtent {
		return nil, nil, errors.New("gated hyperconnection input requires [embedding, streams, tokens]")
	}
	tokens := input.Shape.Dims[tensor.PairedExtent]
	normalized := builder.Multiply(builder.RMSNorm(input, spec.RMSNormEpsilon), builder.Reshape(norm, width, streams))
	normalized = builder.Reshape(normalized, width*streams, tokens)
	low := builder.SiLU(builder.Scale(builder.MulMat(down, normalized), tensor.UnitScale/float32(streams)))
	gated := builder.Multiply(normalized, builder.Sigmoid(builder.MulMat(up, low)))
	var mixed *tensor.Tensor
	for stream := range streams {
		part := builder.Reshape(builder.GroupSlice(gated, stream*width, width, tensor.SingletonExtent, width), width, tokens)
		if mixed == nil {
			mixed = part
		} else {
			mixed = builder.Add(mixed, part)
		}
	}
	mixed = builder.Scale(mixed, tensor.UnitScale/float32(streams))
	var injection *tensor.Tensor
	if inject != nil {
		injection = builder.MulMat(inject, normalized)
		injection = builder.Scale(builder.Sigmoid(builder.Scale(injection, tensor.UnitScale/float32(streams))), tensor.PairedExtent)
		injection = builder.Reshape(injection, tensor.SingletonExtent, streams, tokens)
	}
	return mixed, injection, builder.Err()
}

func executeGatedHyperMix(options blockDispatchOptions, operator layerOperator, execution *layerExecution) error {
	builder, spec, weights := options.Context.Builder, options.Spec, options.Weights
	if operator == layerOperatorGatedHyperHead && options.Context.Layer+tensor.SingletonExtent != spec.BlockCount {
		return nil
	}
	if options.Context.Sequences != tensor.SingletonExtent {
		return errors.New("gated hyperconnection requires one sequence")
	}
	if execution.current == nil {
		return errors.New("gated hyperconnection input is nil")
	}
	if operator == layerOperatorGatedHyperAttention && execution.current.Shape.Rank == tensor.PairedExtent {
		execution.current = builder.HyperConnectionInit(execution.current, spec.HyperConnectionCount)
	}
	if operator == layerOperatorGatedHyperAttention && spec.NGram != nil && spec.NGram.Layer == options.Context.Layer {
		result, err := buildNGramInjection(options, execution.current)
		if err != nil {
			return err
		}
		execution.current = result.Output
		mergeLayerResult(&execution.result, result)
	}
	norm, down, up, inject := weights.HyperAttentionNorm, weights.HyperAttentionDown, weights.HyperAttentionUp, weights.HyperAttentionInject
	if operator == layerOperatorGatedHyperFeedForward {
		norm, down, up, inject = weights.HyperFeedForwardNorm, weights.HyperFeedForwardDown, weights.HyperFeedForwardUp, weights.HyperFeedForwardInject
	} else if operator == layerOperatorGatedHyperHead {
		norm, down, up, inject = weights.HyperHeadNorm, weights.HyperHeadDown, weights.HyperHeadUp, nil
	}
	if operator != layerOperatorGatedHyperHead && inject == nil {
		return errors.New("gated hyperconnection injection weight is nil")
	}
	execution.residual = execution.current
	var err error
	execution.current, execution.injection, err = buildGatedHyperMix(builder, execution.current, norm, down, up, inject, spec)
	if operator == layerOperatorGatedHyperHead {
		execution.result.Output = execution.current
	}
	return err
}
