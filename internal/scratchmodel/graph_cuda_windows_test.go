//go:build windows

package scratchmodel

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestScratchTensorForwardDeviceParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, testDerivationProfile(t))
	if err != nil {
		t.Fatal(err)
	}
	slab := construction.InitialWeights()
	cuda, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	hostModel, err := construction.BindMADTransformer(slab)
	if err != nil {
		t.Fatal(err)
	}
	worstLogit, worstReferenceLoss, worstDeviceLoss := 0.0, 0.0, 0.0
	for document, text := range oracle.Train {
		tokens, err := construction.Tokens(text)
		if err != nil {
			t.Fatal(err)
		}
		graph, err := construction.CompileForwardGraph(tokens)
		if err != nil {
			t.Fatal(err)
		}
		feeds, err := graph.Feeds(construction, slab)
		if err != nil {
			t.Fatal(err)
		}
		want, err := reference.Execute([]*tensor.Tensor{graph.Output}, feeds)
		if err != nil {
			t.Fatal(err)
		}
		got, err := cuda.Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{graph.Output}, feeds)
		if err != nil {
			t.Fatal(err)
		}
		for index := range want[graph.Output].Data {
			worstLogit = max(worstLogit, math.Abs(float64(got[graph.Output].Data[index]-want[graph.Output].Data[index])))
		}
		hostLoss, _, err := hostmath.MADTransformerForward(hostModel, tokens)
		if err != nil {
			t.Fatal(err)
		}
		referenceLoss := scratchGraphLoss(want[graph.Output].Data, tokens, construction.config.VocabSize)
		deviceLoss := scratchGraphLoss(got[graph.Output].Data, tokens, construction.config.VocabSize)
		worstReferenceLoss = max(worstReferenceLoss, math.Abs(referenceLoss-hostLoss))
		worstDeviceLoss = max(worstDeviceLoss, math.Abs(deviceLoss-hostLoss))
		if worstLogit > 2e-4 || worstReferenceLoss > 2e-5 || worstDeviceLoss > 2e-4 {
			t.Fatalf("scratch tensor document=%d logits=%.3e host/ref/dev delta=%.3e/%.3e", document, worstLogit, worstReferenceLoss, worstDeviceLoss)
		}
	}
	t.Logf("scratch tensor corpus CUDA/reference logits=%.3e host/ref/dev loss delta=%.3e/%.3e", worstLogit, worstReferenceLoss, worstDeviceLoss)
}

func scratchGraphLoss(logits []float32, tokens []int, vocab int) float64 {
	positions := len(tokens) - 1
	var loss float64
	for position := range positions {
		row := logits[position*vocab : (position+1)*vocab]
		maximum := row[0]
		for _, value := range row[1:] {
			maximum = max(maximum, value)
		}
		var denominator float64
		for _, value := range row {
			denominator += math.Exp(float64(value - maximum))
		}
		probability := math.Exp(float64(row[tokens[position+1]]-maximum)) / denominator
		loss -= math.Log(probability) / float64(positions)
	}
	return loss
}

// Feeds materializes graph feeds from one flat parameter slab.
func (g ForwardGraph) Feeds(c Construction, slab []float32) (map[*tensor.Tensor]reference.Value, error) {
	if g.Output == nil || g.Mask == nil || g.Positions <= 0 || len(g.Parameters) != len(c.parameters) || len(slab) != len(c.weights) {
		return nil, errors.New("scratch model: forward graph or slab differs")
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(g.Parameters)+1)
	for name, input := range g.Parameters {
		binding, ok := c.bindings[name]
		if !ok {
			return nil, fmt.Errorf("scratch model: graph parameter %q has no slab binding", name)
		}
		value, err := reference.NewValue(input.Shape, slab[binding.start:binding.end])
		if err != nil {
			return nil, err
		}
		feeds[input] = value
	}
	mask, err := reference.NewValue(g.Mask.Shape, g.MaskValues)
	if err != nil {
		return nil, err
	}
	feeds[g.Mask] = mask
	return feeds, nil
}
