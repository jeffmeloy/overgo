//go:build windows

package scratchmodel

import (
	"context"
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
	slab := construction.initialF32()
	cuda, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	hostModel, err := construction.bindMADTransformer(slab)
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
