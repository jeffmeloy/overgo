package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type report struct {
	Model            string  `json:"model"`
	Layer            int     `json:"layer"`
	Tokens           int     `json:"tokens"`
	LoadSeconds      float64 `json:"loadSeconds"`
	ReferenceSeconds float64 `json:"referenceSeconds"`
	CUDASeconds      float64 `json:"cudaSeconds"`
	MaxAbsError      float64 `json:"maxAbsError"`
	MeanAbsError     float64 `json:"meanAbsError"`
}

func main() {
	clioptions.Main(run)
}

func run() error {
	layerIndex := flag.Int("layer", 0, "zero-based transformer layer")
	tokenCount := flag.Int("tokens", 1, "synthetic token count")
	deviceOrdinal := flag.Int("device", 0, "CUDA device ordinal")
	flag.Parse()
	if flag.NArg() != 1 {
		return errors.New("usage: block-check [options] <model.gguf>")
	}
	if *layerIndex < 0 || *tokenCount <= 0 {
		return errors.New("layer must be non-negative and tokens must be positive")
	}
	ctx := context.Background()
	file, err := gguf.Open(flag.Arg(0))
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
	modelPlan, err := model.CompileModelPlan(spec, weights)
	if err != nil {
		return err
	}
	if *layerIndex >= len(weights.Layers) {
		return fmt.Errorf("layer %d exceeds model layer count %d", *layerIndex, len(weights.Layers))
	}

	loadStart := time.Now()
	hostLayer, err := model.LoadHostLayer(ctx, file, weights.Layers[*layerIndex])
	if err != nil {
		return err
	}
	loadDuration := time.Since(loadStart)

	builder := tensor.NewBuilder()
	inputShape := tensor.MustShape(uint64(spec.EmbeddingLength), uint64(*tokenCount))
	input := builder.Input("input", dtype.F32, inputShape)
	graphWeights, feeds, err := hostLayer.GraphInputs(builder, fmt.Sprintf("blk.%d.", *layerIndex))
	if err != nil {
		return err
	}
	positions := make([]uint32, *tokenCount)
	for index := range positions {
		positions[index] = uint32(index)
	}
	program, err := modelPlan.LayerProgram(spec, *layerIndex)
	if err != nil {
		return err
	}
	plan := program.Layer()
	blockResult, err := program.Build(model.CachedBlockContext{
		Builder: builder, Input: input, Positions: positions,
		Layer: plan.Layer, Recurrent: plan.Recurrent,
	}, graphWeights)
	if err != nil {
		return err
	}
	output := blockResult.Output
	inputData := make([]float32, int(spec.EmbeddingLength)*(*tokenCount))
	for index := range inputData {
		inputData[index] = float32(math.Sin(float64(index+1)*0.017)) * 0.25
	}
	inputValue, err := reference.NewValue(inputShape, inputData)
	if err != nil {
		return err
	}
	feeds[input] = inputValue

	referenceStart := time.Now()
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		return err
	}
	referenceDuration := time.Since(referenceStart)

	cuda, err := executor.New(*deviceOrdinal)
	if err != nil {
		return err
	}
	defer cuda.Close()
	cudaStart := time.Now()
	got, err := cuda.Execute(ctx, []*tensor.Tensor{output}, feeds)
	if err != nil {
		return err
	}
	cudaDuration := time.Since(cudaStart)

	var maximum, sum float64
	for index, expected := range want[output].Data {
		difference := math.Abs(float64(got[output].Data[index] - expected))
		sum += difference
		if difference > maximum {
			maximum = difference
		}
	}
	result := report{
		Model:            spec.Name,
		Layer:            *layerIndex,
		Tokens:           *tokenCount,
		LoadSeconds:      loadDuration.Seconds(),
		ReferenceSeconds: referenceDuration.Seconds(),
		CUDASeconds:      cudaDuration.Seconds(),
		MaxAbsError:      maximum,
		MeanAbsError:     sum / float64(len(want[output].Data)),
	}
	return clioptions.WritePrettyJSON(os.Stdout, result)
}
