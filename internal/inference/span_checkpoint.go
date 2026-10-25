package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

// deviceCacheCheckpoint is a verify span's cache at one of its checkpoints: the tokens and position after the span's
// first tokens, and each layer's fixed state there (zero where a layer keeps none), which commitSpanCheckpoint keeps
// when a window keeps those tokens alone.
type deviceCacheCheckpoint struct {
	tokens, position uint32
	keys, values     []executor.DeviceValue
	states           []deviceLayerStates
}

// maxSpanCheckpoints bounds a span's checkpoints, one fewer than the q8-input span's columns.
const maxSpanCheckpoints = q8InputSpanColumnsWindow - 1

// spanCheckpointIdentity is a span's checkpoint token counts as a decode session's identity holds them.
func spanCheckpointIdentity(checkpoints []uint32) [maxSpanCheckpoints]uint32 {
	var identity [maxSpanCheckpoints]uint32
	copy(identity[:], checkpoints)
	return identity
}

// collectSpanCheckpoints reads a span's checkpoint outputs from its retained results, one checkpoint a token count;
// none for a graph without them.
func collectSpanCheckpoints(retained *executor.RetainedOutputs, graph deviceBatchGraph) ([]*deviceCacheCheckpoint, error) {
	checkpoints := make([]*deviceCacheCheckpoint, 0, len(graph.checkpointTokens))
	for index, at := range graph.checkpointTokens {
		checkpoint, err := collectSpanCheckpoint(retained, graph, index, at)
		if err != nil || checkpoint == nil {
			return nil, err
		}
		checkpoints = append(checkpoints, checkpoint)
	}
	return checkpoints, nil
}

// collectSpanCheckpoint reads the span's checkpoint after its first at tokens, the index-th; nil where its layers kept
// none there.
func collectSpanCheckpoint(retained *executor.RetainedOutputs, graph deviceBatchGraph, index int, at uint32) (*deviceCacheCheckpoint, error) {
	if at == 0 || at >= graph.tokenCount {
		return nil, nil
	}
	checkpoint := &deviceCacheCheckpoint{
		tokens: graph.pastTokens + at, position: graph.nextPosition + at,
		keys: make([]executor.DeviceValue, len(graph.checkpoints)), values: make([]executor.DeviceValue, len(graph.checkpoints)),
		states: make([]deviceLayerStates, len(graph.checkpoints)),
	}
	read := func(node *tensor.Tensor, layer int, name string) (executor.DeviceValue, error) {
		value, ok := retained.Value(node)
		if !ok {
			return executor.DeviceValue{}, fmt.Errorf("inference: missing retained checkpoint %s for layer %d", name, layer)
		}
		return value, nil
	}
	for layer, layerCheckpoints := range graph.checkpoints {
		if index >= len(layerCheckpoints) {
			continue
		}
		kept := layerCheckpoints[index]
		var err error
		if kept.Key != nil {
			if checkpoint.keys[layer], err = read(kept.Key, layer, "key"); err != nil {
				return nil, err
			}
		}
		if kept.Value != nil {
			if checkpoint.values[layer], err = read(kept.Value, layer, "value"); err != nil {
				return nil, err
			}
		}
		if len(kept.States) != 0 {
			checkpoint.states[layer] = make(deviceLayerStates, len(kept.States))
			for name, state := range kept.States {
				value, err := read(state.Value, layer, string(name))
				if err != nil {
					return nil, err
				}
				checkpoint.states[layer][name] = deviceLayerState{Mode: state.Mode, Value: value}
			}
		}
	}
	return checkpoint, nil
}

// commitSpanCheckpoint keeps a verify span's cache at its checkpoint: its layers' fixed state there, its token-aligned
// keys, values and states narrowed to the tokens kept (a later span writes the rows after them again), and the lexical
// history of the kept tokens after the boundary's. kept are the span's tokens before the checkpoint it keeps: its pending
// tokens and a prefix of its drafts.
func (r *Runner) commitSpanCheckpoint(ctx context.Context, span, boundary *deviceKVCache, kept []tokenizer.TokenID) error {
	if boundary == nil {
		return errors.New("inference: a span commits onto its boundary")
	}
	// The checkpoint after the kept tokens: the pending tokens and a prefix of the drafts.
	var checkpoint *deviceCacheCheckpoint
	for _, candidate := range span.checkpoints {
		if candidate.tokens == boundary.Tokens+uint32(len(kept)) && len(candidate.keys) == len(span.Keys) {
			checkpoint = candidate
		}
	}
	if checkpoint == nil {
		return errors.New("inference: the span holds no checkpoint after its kept tokens")
	}
	narrow := func(value executor.DeviceValue) executor.DeviceValue {
		if value.Pointer != 0 && value.Shape.Rank > 0 {
			value.Shape.Dims[value.Shape.Rank-tensor.SingletonExtent] = uint64(checkpoint.tokens)
		}
		return value
	}
	for layer := range span.Keys {
		if r.layerProgram(layer).Layer().SharedKV {
			continue
		}
		_, schema, err := r.cacheSchema(layer, checkpoint.tokens)
		if err != nil {
			return err
		}
		switch {
		case checkpoint.keys[layer].Pointer != 0:
			span.Keys[layer] = checkpoint.keys[layer]
		case schema.Primary.Key.Mode.TokenAligned():
			span.Keys[layer] = narrow(span.Keys[layer])
		}
		switch {
		case checkpoint.values[layer].Pointer != 0:
			span.Values[layer] = checkpoint.values[layer]
		case schema.Primary.Value.Mode.TokenAligned():
			span.Values[layer] = narrow(span.Values[layer])
		}
		if layer >= len(span.States) {
			continue
		}
		for name, state := range span.States[layer] {
			if kept, ok := checkpoint.states[layer][name]; ok {
				span.States[layer][name] = kept
			} else if state.Mode.TokenAligned() {
				state.Value = narrow(state.Value)
				span.States[layer][name] = state
			}
		}
	}
	for layer := range span.Keys {
		if plan := r.layerProgram(layer).Layer(); plan.SharedKV {
			span.Keys[layer], span.Values[layer] = span.Keys[plan.KVSource], span.Values[plan.KVSource]
		}
	}
	if r.spec.NGram != nil {
		past, err := r.deviceNGramHistory(ctx, boundary)
		if err != nil {
			return err
		}
		previous := make([]uint32, len(past))
		for index, value := range past {
			previous[index] = uint32(value)
		}
		tokens := make([]uint32, len(kept))
		for index, token := range kept {
			tokens[index] = uint32(token)
		}
		_, next, err := r.spec.NGramRows(tokens, previous)
		if err != nil {
			return err
		}
		span.ngramHistory = make([]float32, len(next))
		for index, token := range next {
			span.ngramHistory[index] = float32(token)
		}
		// The card's copy follows, for a cache the prompt cache later reads back.
		if layer := int(r.spec.NGram.Layer); layer < len(span.States) {
			if state, ok := span.States[layer][model.CacheStateNGramHistory]; ok {
				if err := r.cuda.WriteDeviceValues(ctx, []executor.DeviceValue{state.Value}, [][]float32{span.ngramHistory}); err != nil {
					return err
				}
			}
		}
	}
	span.Tokens, span.Position = checkpoint.tokens, checkpoint.position
	span.Selection, span.session, span.checkpoints = executor.DeviceValue{}, nil, nil
	return rebuildDeviceCachePages(span, r.program.Decode.Session)
}
