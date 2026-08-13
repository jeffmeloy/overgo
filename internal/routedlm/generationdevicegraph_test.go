package routedlm

import (
	"testing"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
)

func TestSenseNovaGenerationLayerGraphStructure(t *testing.T) {
	cfg := generationGraphConfig()
	rope, err := newRopePlan(cfg.HeadDim, []RopeSection{
		{Width: 4, Theta: 5e6, Axis: AxisTime},
		{Width: 2, Theta: 1e4, Axis: AxisHeight},
		{Width: 2, Theta: 1e4, Axis: AxisWidth},
	})
	if err != nil {
		t.Fatal(err)
	}
	graph, err := BuildDeviceGenerationLayer(cfg, rope, 3, 2, 2, 11)
	if err != nil {
		t.Fatal(err)
	}
	if !graph.Output.Shape.Equal(tensor.MustShape(uint64(cfg.HiddenSize), 4)) {
		t.Fatalf("generation output shape=%v", graph.Output.Shape)
	}
	if _, err := executor.Compile(graph.Output); err != nil {
		t.Fatal(err)
	}
}

func generationGraphConfig() Config {
	return Config{
		HiddenSize: 16, IntermediateSize: 24, NumHiddenLayers: 1,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 8,
		RopeTheta: 5e6, VocabSize: 32, MaxPositionEmbeddings: 128,
		RMSNormEps: 1e-6, HiddenAct: "silu",
	}
}

func generationGraphNodes(graph *DeviceGenerationLayerGraph) []*tensor.Tensor {
	return []*tensor.Tensor{
		graph.Row, graph.PrefixKey, graph.PrefixValue,
		graph.Vision.InputNorm, graph.Vision.Q, graph.Vision.K, graph.Vision.V,
		graph.Vision.O, graph.Vision.QNorm, graph.Vision.KNorm, graph.Vision.PostNorm,
		graph.Vision.Gate, graph.Vision.Up, graph.Vision.Down,
	}
}
