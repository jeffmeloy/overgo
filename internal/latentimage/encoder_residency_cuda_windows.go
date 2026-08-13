//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

type ResidentEncoder struct {
	Program     *EncoderProgram
	WeightBytes uint64
	runtime     *residentRuntime
	graph       *residentGraph
}

func NewResidentEncoder(
	ctx context.Context,
	program *EncoderProgram,
	modelDir string,
	ordinal int,
) (encoder *ResidentEncoder, err error) {
	if program == nil {
		return nil, errors.New("resident encoder: program is nil")
	}
	runtime, err := newResidentRuntime(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident encoder: device: %w", err)
	}
	encoder = &ResidentEncoder{Program: program, runtime: runtime}
	defer func() {
		if err != nil {
			_ = encoder.Close(ctx)
		}
	}()
	encoder.graph, err = runtime.compile(
		ctx, "resident encoder", filepath.Join(modelDir, "text_encoder"),
		program.weightInputs, program.Selected...,
	)
	if err != nil {
		return nil, err
	}
	encoder.WeightBytes = encoder.graph.bytes
	return encoder, nil
}

func (r *ResidentEncoder) Encode(ctx context.Context, embedRows []float32) (*SelectedHiddenStates, error) {
	if r == nil || r.Program == nil {
		return nil, errors.New("resident encoder: unavailable")
	}
	program := r.Program
	if len(embedRows) != program.Seq*program.E.Hidden {
		return nil, fmt.Errorf("resident encode: embed len=%d want %d", len(embedRows), program.Seq*program.E.Hidden)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		program.Embed: {Shape: program.Embed.Shape, Data: embedRows},
	}
	if program.keyBias != nil {
		feeds[program.keyBias] = reference.Value{Shape: program.keyBias.Shape, Data: program.keyBiasData}
	}
	results, err := r.runtime.execute(ctx, r.graph, feeds)
	if err != nil {
		return nil, fmt.Errorf("resident encode: %w", err)
	}
	return program.assemble(results)
}

func readEmbedRowsF32(modelDir string, spec TextEncoderSpec, ids []int) ([]float32, error) {
	source, err := safetensors.OpenSource(filepath.Join(modelDir, "text_encoder"))
	if err != nil {
		return nil, fmt.Errorf("resident encoder embed: open text_encoder: %w", err)
	}
	defer source.Close()
	rows, err := readEmbedRows(source, spec, ids)
	if err != nil {
		return nil, err
	}
	result := make([]float32, len(rows))
	for index, value := range rows {
		result[index] = float32(value)
	}
	return result, nil
}

func (r *ResidentEncoder) Close(ctx context.Context) error {
	if r == nil || r.runtime == nil {
		return nil
	}
	err := r.runtime.close(ctx)
	r.runtime = nil
	r.graph = nil
	return err
}
