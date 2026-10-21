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

// deviceCacheCheckpoint is a verify span's cache at its checkpoint: the tokens and position after the span's first
// tokens, and each layer's fixed state there (zero where a layer keeps none), which commitSpanCheckpoint keeps when the
// window's drafts are rejected.
type deviceCacheCheckpoint struct {
	tokens, position uint32
	keys, values     []executor.DeviceValue
	states           []deviceLayerStates
}

// collectSpanCheckpoint reads a span's checkpoint outputs from its retained results; nil for a graph without one.
func collectSpanCheckpoint(retained *executor.RetainedOutputs, graph deviceBatchGraph) (*deviceCacheCheckpoint, error) {
	if graph.checkpoint == 0 || graph.checkpoint >= graph.tokenCount {
		return nil, nil
	}
	checkpoint := &deviceCacheCheckpoint{
		tokens: graph.pastTokens + graph.checkpoint, position: graph.nextPosition + graph.checkpoint,
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
	for layer, kept := range graph.checkpoints {
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
// history of the kept tokens after the boundary's. kept are the span's tokens before its checkpoint.
func (r *Runner) commitSpanCheckpoint(ctx context.Context, span, boundary *deviceKVCache, kept []tokenizer.TokenID) error {
	checkpoint := span.checkpoint
	if checkpoint == nil || boundary == nil || checkpoint.tokens != boundary.Tokens+uint32(len(kept)) || len(checkpoint.keys) != len(span.Keys) {
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
	span.Selection, span.session, span.checkpoint = executor.DeviceValue{}, nil, nil
	return rebuildDeviceCachePages(span, r.program.Decode.Session)
}
