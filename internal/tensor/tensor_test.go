package tensor

import (
	"strings"
	"testing"

	"llamacpp2go/internal/tensor/dtype"
)

func TestBuilderAndTopological(t *testing.T) {
	builder := NewBuilder()
	shape := MustShape(4, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.SiLU(builder.Scale(builder.Add(left, right), 0.5))
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	nodes, err := Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 5 || nodes[len(nodes)-1] != output {
		t.Fatalf("unexpected topological order: %#v", nodes)
	}
}

func TestBuilderRejectsShapeMismatch(t *testing.T) {
	builder := NewBuilder()
	left := builder.Input("left", dtype.F32, MustShape(4))
	right := builder.Input("right", dtype.F32, MustShape(8))
	if result := builder.Add(left, right); result != nil {
		t.Fatal("shape-mismatched add returned a tensor")
	}
	if err := builder.Err(); err == nil || !strings.Contains(err.Error(), "cannot broadcast") {
		t.Fatalf("error = %v", err)
	}
}

func TestBuilderBroadcastWeightedNormAndSwiGLU(t *testing.T) {
	builder := NewBuilder()
	activation := builder.Input("activation", dtype.F32, MustShape(4, 3))
	weight := builder.Input("weight", dtype.F32, MustShape(4))
	norm := builder.WeightedRMSNorm(activation, weight, 1e-5)
	up := builder.Input("up", dtype.F32, MustShape(4, 3))
	output := builder.SwiGLU(norm, up)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(4, 3)) {
		t.Fatalf("output shape = %v", output.Shape.Slice())
	}
	if norm.Op != OpMultiply || output.Op != OpMultiply {
		t.Fatalf("unexpected composed ops: %s and %s", norm.Op, output.Op)
	}
}

func TestBuilderQwen35UnaryPrimitives(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(4, 3))
	output := builder.L2Norm(builder.Softplus(builder.Sigmoid(input)), 1e-6)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if output.Op != OpL2Norm || output.Inputs[0].Op != OpSoftplus ||
		output.Inputs[0].Inputs[0].Op != OpSigmoid {
		t.Fatalf("unexpected unary graph ending in %s", output.Op)
	}

	invalid := NewBuilder()
	invalid.L2Norm(invalid.Input("input", dtype.F32, MustShape(1)), -1)
	if invalid.Err() == nil {
		t.Fatal("negative L2Norm epsilon was accepted")
	}
}

func TestBuilderSSMConv(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(6, 4, 2))
	weights := builder.Input("weights", dtype.F32, MustShape(3, 4))
	output := builder.SSMConv(input, weights)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(4, 4, 2)) {
		t.Fatalf("SSMConv output shape = %v", output.Shape.Slice())
	}
}

func TestBuilderGatedDeltaNet(t *testing.T) {
	builder := NewBuilder()
	q := builder.Input("q", dtype.F32, MustShape(4, 2, 3, 2))
	k := builder.Input("k", dtype.F32, MustShape(4, 1, 3, 2))
	v := builder.Input("v", dtype.F32, MustShape(4, 4, 3, 2))
	gate := builder.Input("gate", dtype.F32, MustShape(4, 4, 3, 2))
	beta := builder.Input("beta", dtype.F32, MustShape(1, 4, 3, 2))
	state := builder.Input("state", dtype.F32, MustShape(4, 4, 4, 2))
	output := builder.GatedDeltaNet(q, k, v, gate, beta, state)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(MustShape(16, 14)) {
		t.Fatalf("GatedDeltaNet output shape = %v", output.Shape.Slice())
	}
}

func TestBuilderQwen35LayoutOperations(t *testing.T) {
	builder := NewBuilder()
	projection := builder.Input("projection", dtype.F32, MustShape(16, 3))
	query := builder.GroupSlice(projection, 0, 2, 4, 4)
	gate := builder.GroupSlice(projection, 2, 2, 4, 4)
	transposed := builder.Transpose2D(projection)
	state := builder.Input("state", dtype.F32, MustShape(2, 16))
	convInput := builder.Concat(state, transposed, 0)
	packed := builder.Input("packed", dtype.F32, MustShape(8, 6))
	attention := builder.FlatSlice(packed, 0, 2, 4, 3)
	newState := builder.FlatSlice(packed, 24, 2, 2, 6)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !query.Shape.Equal(MustShape(2, 4, 3)) ||
		!gate.Shape.Equal(query.Shape) ||
		!convInput.Shape.Equal(MustShape(5, 16)) ||
		!attention.Shape.Equal(MustShape(2, 4, 3)) ||
		!newState.Shape.Equal(MustShape(2, 2, 6)) {
		t.Fatalf(
			"unexpected layout shapes: q=%v conv=%v attn=%v state=%v",
			query.Shape.Slice(),
			convInput.Shape.Slice(),
			attention.Shape.Slice(),
			newState.Shape.Slice(),
		)
	}
}

func TestBuilderGetRowsAndRoPE(t *testing.T) {
	builder := NewBuilder()
	table := builder.Input("table", dtype.F32, MustShape(8, 16))
	embedding := builder.GetRows(table, []uint32{3, 5})
	rope := builder.RoPENeoX(
		builder.Input("query", dtype.F32, MustShape(8, 2, 2)),
		[]uint32{4, 5},
		8,
		10000,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if !embedding.Shape.Equal(MustShape(8, 2)) || !rope.Shape.Equal(MustShape(8, 2, 2)) {
		t.Fatalf("unexpected shapes %v and %v", embedding.Shape.Slice(), rope.Shape.Slice())
	}

	var positions [4][]uint32
	positions[0] = []uint32{1, 2}
	positions[1] = []uint32{3, 4}
	positions[2] = []uint32{5, 6}
	positions[3] = []uint32{7, 8}
	multi := builder.RoPEMulti(
		builder.Input("multi", dtype.F32, MustShape(8, 1, 2)),
		positions,
		[4]int32{1, 1, 1, 1},
		8,
		10000,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if multi.Op != OpRoPEMulti || !multi.Shape.Equal(MustShape(8, 1, 2)) {
		t.Fatalf("unexpected multi-axis RoPE tensor: %+v", multi)
	}
}

func TestBuilderQ8DeviceOperationsProduceF32(t *testing.T) {
	builder := NewBuilder()
	table := builder.Input("table", dtype.Q8_0, MustShape(32, 4))
	rows := builder.GetRows(table, []uint32{2})
	input := builder.Input("input", dtype.F32, MustShape(32, 2))
	product := builder.MulMat(table, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if rows.Type != dtype.F32 || product.Type != dtype.F32 {
		t.Fatalf("Q8 outputs have types %s and %s, want f32", rows.Type, product.Type)
	}
}

func TestBuilderQ6KDeviceOperationsProduceF32(t *testing.T) {
	builder := NewBuilder()
	table := builder.Input("table", dtype.Q6K, MustShape(256, 4))
	rows := builder.GetRows(table, []uint32{2})
	input := builder.Input("input", dtype.F32, MustShape(256, 2))
	product := builder.MulMat(table, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	if rows.Type != dtype.F32 || product.Type != dtype.F32 {
		t.Fatalf("Q6_K outputs have types %s and %s, want f32", rows.Type, product.Type)
	}
}

func TestBuilderKQuantDeviceOperationsProduceF32(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.Q2K,
		dtype.Q3K,
		dtype.Q4K,
		dtype.Q5K,
		dtype.Q8K,
		dtype.IQ2XXS,
		dtype.IQ2XS,
		dtype.IQ2S,
		dtype.IQ3XXS,
		dtype.IQ3S,
		dtype.IQ1S,
		dtype.IQ1M,
		dtype.IQ4XS,
		dtype.TQ1_0,
		dtype.TQ2_0,
	} {
		builder := NewBuilder()
		table := builder.Input("table", dataType, MustShape(256, 4))
		rows := builder.GetRows(table, []uint32{2})
		input := builder.Input("input", dtype.F32, MustShape(256, 2))
		product := builder.MulMat(table, input)
		if err := builder.Err(); err != nil {
			t.Fatalf("%s: %v", dataType, err)
		}
		if rows.Type != dtype.F32 || product.Type != dtype.F32 {
			t.Fatalf("%s outputs have types %s and %s, want f32", dataType, rows.Type, product.Type)
		}
	}
}

func TestBuilderClassicQuantDeviceOperationsProduceF32(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.Q4_0,
		dtype.Q4_1,
		dtype.Q5_0,
		dtype.Q5_1,
		dtype.Q8_1,
		dtype.IQ4NL,
		dtype.MXFP4,
	} {
		builder := NewBuilder()
		table := builder.Input("table", dataType, MustShape(32, 4))
		rows := builder.GetRows(table, []uint32{2})
		input := builder.Input("input", dtype.F32, MustShape(32, 2))
		product := builder.MulMat(table, input)
		if err := builder.Err(); err != nil {
			t.Fatalf("%s: %v", dataType, err)
		}
		if rows.Type != dtype.F32 || product.Type != dtype.F32 {
			t.Fatalf("%s outputs have types %s and %s, want f32", dataType, rows.Type, product.Type)
		}
	}
}

func TestBuilderSmallQuantDeviceOperationsProduceF32(t *testing.T) {
	for _, test := range []struct {
		dataType dtype.Type
		width    uint64
	}{
		{dtype.Q1_0, 128},
		{dtype.Q2_0, 64},
		{dtype.NVFP4, 64},
	} {
		builder := NewBuilder()
		table := builder.Input("table", test.dataType, MustShape(test.width, 4))
		rows := builder.GetRows(table, []uint32{2})
		input := builder.Input("input", dtype.F32, MustShape(test.width, 2))
		product := builder.MulMat(table, input)
		if err := builder.Err(); err != nil {
			t.Fatalf("%s: %v", test.dataType, err)
		}
		if rows.Type != dtype.F32 || product.Type != dtype.F32 {
			t.Fatalf("%s outputs have types %s and %s, want f32", test.dataType, rows.Type, product.Type)
		}
	}
}

func TestTopologicalRejectsCycle(t *testing.T) {
	node := &Tensor{ID: 1, Type: dtype.F32, Shape: MustShape(1), Op: OpScale}
	node.Inputs = []*Tensor{node}
	if _, err := Topological(node); err == nil {
		t.Fatal("cyclic graph was accepted")
	}
}
