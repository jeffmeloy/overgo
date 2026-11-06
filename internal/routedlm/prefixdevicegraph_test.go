package routedlm

import (
	"testing"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/testcheck"
)

func TestDevicePrefixLayerUsesBlockCausalMultiAxisPlan(t *testing.T) {
	cfg := generationGraphConfig()
	rope := testcheck.Must(newRopePlan(cfg.HeadDim, []RopeSection{
		{Width: 4, Theta: 5e6, Axis: AxisTime},
		{Width: 2, Theta: 1e4, Axis: AxisHeight},
		{Width: 2, Theta: 1e4, Axis: AxisWidth},
	}))(t)
	positions := []RowPosition{
		{Time: 0},
		{Branch: 1, Time: 1, H: 0, W: 0},
		{Branch: 1, Time: 1, H: 0, W: 1},
		{Time: 2},
	}
	graph := testcheck.Must(BuildDevicePrefixLayer(cfg, rope, positions))(t)
	if !graph.Output.Shape.Equal(tensor.MustShape(16, 4)) ||
		!graph.KeyKV.Shape.Equal(tensor.MustShape(8, 1, 4)) {
		t.Fatalf("prefix shapes output=%v key=%v", graph.Output.Shape, graph.KeyKV.Shape)
	}
	if _, err := executor.Compile(graph.Output, graph.KeyKV, graph.ValueKV); err != nil {
		t.Fatal(err)
	}
}
