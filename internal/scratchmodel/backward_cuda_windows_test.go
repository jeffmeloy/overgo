//go:build windows

package scratchmodel

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

func TestScratchTensorDeviceVJPParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, testDerivationProfile(t))
	if err != nil {
		t.Fatal(err)
	}
	weights := construction.initialF32()
	cuda, err := executor.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	model, err := construction.bindMADTransformer(weights)
	if err != nil {
		t.Fatal(err)
	}
	worst, worstLoss := 0.0, 0.0
	groupWorst := make(map[string]float64, len(construction.parameters))
	for _, text := range oracle.Train {
		tokens, err := construction.Tokens(text)
		if err != nil {
			t.Fatal(err)
		}
		graph, err := construction.CompileForwardGraph(tokens)
		if err != nil {
			t.Fatal(err)
		}
		feeds, err := graph.Feeds(construction, weights)
		if err != nil {
			t.Fatal(err)
		}
		values, err := cuda.Execute(context.WithoutCancel(t.Context()), graph.cacheOutputs(), feeds)
		if err != nil {
			t.Fatal(err)
		}
		deviceLoss, got, err := graph.DeviceLossAndGrad(worker, construction, weights, tokens, values)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]float32, len(weights))
		gradient, err := construction.bindMADTransformer(want)
		if err != nil {
			t.Fatal(err)
		}
		hostLoss, trace, err := hostmath.MADTransformerForward(model, tokens)
		if err != nil {
			t.Fatal(err)
		}
		if err := hostmath.MADTransformerBackward(model, gradient, tokens, trace); err != nil {
			t.Fatal(err)
		}
		worstLoss = max(worstLoss, math.Abs(deviceLoss-hostLoss))
		for _, parameter := range construction.parameters {
			binding := construction.bindings[parameter.Name]
			delta := 0.0
			for index := binding.start; index < binding.end; index++ {
				delta = max(delta, math.Abs(float64(got[index]-want[index])))
			}
			groupWorst[parameter.Name] = max(groupWorst[parameter.Name], delta)
			worst = max(worst, delta)
		}
	}
	for _, parameter := range construction.parameters {
		t.Logf("%-12s corpus max|device-host| %.3e", parameter.Name, groupWorst[parameter.Name])
	}
	t.Logf("scratch VJP corpus loss delta=%.3e gradient delta=%.3e", worstLoss, worst)
	if worstLoss > 2e-5 || worst > 2e-4 {
		t.Fatalf("scratch device VJP differs: loss=%.3e gradient=%.3e", worstLoss, worst)
	}
}
