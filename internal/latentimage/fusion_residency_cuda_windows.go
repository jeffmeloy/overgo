//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

type ResidentFusion struct {
	Program     *FusionProgram
	WeightBytes uint64
	runtime     *residentRuntime
	graph       *residentGraph
}

func NewResidentFusion(
	ctx context.Context,
	program *FusionProgram,
	modelDir string,
	ordinal int,
) (fusion *ResidentFusion, err error) {
	if program == nil {
		return nil, errors.New("resident fusion: program is nil")
	}
	runtime, err := newResidentRuntime(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident fusion: device: %w", err)
	}
	fusion = &ResidentFusion{Program: program, runtime: runtime}
	defer func() {
		if err != nil {
			_ = fusion.Close(ctx)
		}
	}()
	fusion.graph, err = runtime.compile(
		ctx, "resident fusion", filepath.Join(modelDir, "transformer"),
		program.weightInputs, program.Fused,
	)
	if err != nil {
		return nil, err
	}
	fusion.WeightBytes = fusion.graph.bytes
	return fusion, nil
}

func (r *ResidentFusion) Fuse(ctx context.Context, encoderHidden []float32) ([]float32, error) {
	if r == nil || r.Program == nil {
		return nil, errors.New("resident fusion: unavailable")
	}
	program := r.Program
	if want := program.TextSeq * program.T.TextLayers * program.T.TextHidden; len(encoderHidden) != want {
		return nil, fmt.Errorf("resident fuse: encoder hidden len=%d want %d", len(encoderHidden), want)
	}
	results, err := r.runtime.execute(ctx, r.graph, map[*tensor.Tensor]reference.Value{
		program.InEncoder: {Shape: program.InEncoder.Shape, Data: encoderHidden},
	})
	if err != nil {
		return nil, fmt.Errorf("resident fuse: %w", err)
	}
	return program.assemble(results)
}

func (r *ResidentFusion) Close(ctx context.Context) error {
	if r == nil || r.runtime == nil {
		return nil
	}
	err := r.runtime.close(ctx)
	r.runtime = nil
	r.graph = nil
	return err
}
