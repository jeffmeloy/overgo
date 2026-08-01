package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const projectedInputsJSONLimit = 1 << 30

type projectedInputsDocument struct {
	EmbeddingOverrides  []projectedEmbeddingOverride  `json:"embedding_overrides"`
	MultiAxisPositions  *inference.MultiAxisPositions `json:"multi_axis_positions"`
	DeepstackEmbeddings []projectedTensor             `json:"deepstack_embeddings"`
}

type projectedEmbeddingOverride struct {
	TokenIndex uint32    `json:"token_index"`
	Embedding  []float32 `json:"embedding"`
}

type projectedTensor struct {
	Shape []uint64  `json:"shape"`
	Data  []float32 `json:"data"`
}

func readProjectedInputs(path string) (inference.ProjectedInputs, error) {
	file, err := os.Open(path)
	if err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: open projected inputs: %w", err)
	}
	defer file.Close()
	decoder := json.NewDecoder(io.LimitReader(file, projectedInputsJSONLimit+1))
	decoder.DisallowUnknownFields()
	var document projectedInputsDocument
	if err := decoder.Decode(&document); err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: decode projected inputs: %w", err)
	}
	if err := requireJSONEOF(decoder); err != nil {
		return inference.ProjectedInputs{}, fmt.Errorf("generate: decode projected inputs: %w", err)
	}
	result := inference.ProjectedInputs{MultiAxisPositions: document.MultiAxisPositions}
	result.EmbeddingOverrides = make([]inference.EmbeddingOverride, len(document.EmbeddingOverrides))
	for index, item := range document.EmbeddingOverrides {
		if len(item.Embedding) == 0 || !finiteFloats(item.Embedding) {
			return inference.ProjectedInputs{}, fmt.Errorf("generate: embedding override %d is empty or non-finite", index)
		}
		result.EmbeddingOverrides[index] = inference.EmbeddingOverride{
			TokenIndex: item.TokenIndex,
			Embedding:  append([]float32(nil), item.Embedding...),
		}
	}
	result.DeepstackEmbeddings = make([]reference.Value, len(document.DeepstackEmbeddings))
	for index, item := range document.DeepstackEmbeddings {
		shape, shapeErr := tensor.NewShape(item.Shape...)
		if shapeErr != nil {
			return inference.ProjectedInputs{}, fmt.Errorf("generate: deepstack tensor %d shape: %w", index, shapeErr)
		}
		if !finiteFloats(item.Data) {
			return inference.ProjectedInputs{}, fmt.Errorf("generate: deepstack tensor %d contains non-finite data", index)
		}
		value, valueErr := reference.NewValue(shape, item.Data)
		if valueErr != nil {
			return inference.ProjectedInputs{}, fmt.Errorf("generate: deepstack tensor %d: %w", index, valueErr)
		}
		result.DeepstackEmbeddings[index] = value
	}
	return result, nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return nil
	} else if err != nil {
		return err
	}
	return errors.New("trailing JSON value")
}

func finiteFloats(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}
