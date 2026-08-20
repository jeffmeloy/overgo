//go:build windows

// Census helpers over the real artifact.
package main

import (
	"fmt"
	"reflect"

	"overgo/internal/gguf"
	"overgo/internal/model"
)

// printLayerSlots lists the non-nil tensor slots of one layer inventory.
func printLayerSlots(index int, layer model.LayerWeights) {
	value := reflect.ValueOf(layer)
	kind := "attention"
	if layer.Recurrent {
		kind = "recurrent"
	}
	fmt.Printf("layer %d (%s):", index, kind)
	for field := range value.NumField() {
		item := value.Field(field)
		if item.Kind() != reflect.Pointer || item.IsNil() {
			continue
		}
		info, ok := item.Interface().(*gguf.TensorInfo)
		if !ok {
			continue
		}
		fmt.Printf(" %s%v", value.Type().Field(field).Name, info.Shape[:info.Dimensions])
	}
	fmt.Println()
}

// printCensus dumps the spec geometry and per-layer inventory.
func printCensus(path string) error {
	file, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return err
	}
	fmt.Printf("architecture=%s name=%q blocks=%d hidden=%d ffn=%d vocab=%d\n",
		spec.Architecture, spec.Name, spec.BlockCount, spec.EmbeddingLength, spec.FeedForwardLength, spec.VocabularySize)
	fmt.Printf("attn: heads=%d kvHeads=%d keyLen=%d valLen=%d ropeDim=%d ropeBase=%g eps=%g\n",
		spec.HeadCount, spec.HeadCountKV, spec.KeyLength, spec.ValueLength, spec.RopeDimensionCount, spec.RopeFrequencyBase, spec.RMSNormEpsilon)
	fmt.Printf("ssm: stateSize=%d groupCount=%d timeStepRank=%d convKernel=%d innerSize=%d\n",
		spec.SSMStateSize, spec.SSMGroupCount, spec.SSMTimeStepRank, spec.SSMConvKernel, spec.SSMInnerSize)
	recurrent, attention := 0, 0
	for i := range weights.Layers {
		if spec.IsRecurrentLayer(uint32(i)) {
			recurrent++
		} else {
			attention++
		}
	}
	fmt.Printf("layers: %d recurrent, %d attention; output=%v tokenEmbedding=%v\n",
		recurrent, attention, weights.Output != nil, weights.TokenEmbedding.Shape[:weights.TokenEmbedding.Dimensions])
	seen := map[bool]bool{}
	for i := range weights.Layers {
		if kind := spec.IsRecurrentLayer(uint32(i)); !seen[kind] {
			seen[kind] = true
			printLayerSlots(i, weights.Layers[i])
		}
	}
	return nil
}
