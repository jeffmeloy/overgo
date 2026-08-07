package reference

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func TestExecuteElementwiseNormSoftmax(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Softmax(builder.RMSNorm(builder.Add(left, right), 1e-5))
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	leftValue, _ := NewValue(shape, []float32{1, 2, 3, 4, -1, -2, -3, -4})
	rightValue, _ := NewValue(shape, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		left:  leftValue,
		right: rightValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	got := results[output]
	for row := 0; row < 2; row++ {
		var sum float32
		for _, value := range got.Data[row*4 : row*4+4] {
			sum += value
		}
		if math.Abs(float64(sum-1)) > 1e-6 {
			t.Fatalf("softmax row %d sums to %v", row, sum)
		}
	}
}

func TestExecuteReLUSquared(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(5))
	output := builder.ReLUSquared(input)
	value, _ := NewValue(input.Shape, []float32{-2, -0.5, 0, 1.5, 3})
	results, err := Execute(
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]Value{input: value},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 0, 0, 2.25, 9}
	for index := range want {
		if results[output].Data[index] != want[index] {
			t.Fatalf(
				"ReLU squared[%d] = %v, want %v",
				index,
				results[output].Data[index],
				want[index],
			)
		}
	}
}

func TestExecuteGELUErf(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(5))
	output := builder.GELUErf(input)
	values := []float32{-2, -0.5, 0, 1.5, 3}
	value, _ := NewValue(input.Shape, values)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{input: value})
	if err != nil {
		t.Fatal(err)
	}
	for index, item := range results[output].Data {
		want := float32(0.5 * float64(values[index]) * (1 + math.Erf(float64(values[index])/math.Sqrt2)))
		if math.Abs(float64(item-want)) > 1e-7 {
			t.Fatalf("GELU erf[%d] = %v, want %v", index, item, want)
		}
	}
}

func TestExecuteClamp(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(5))
	output := builder.Clamp(input, -1, 2)
	value, _ := NewValue(input.Shape, []float32{-3, -1, 0.5, 2, 4})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{input: value})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-1, -1, 0.5, 2, 2}
	for index, item := range results[output].Data {
		if item != want[index] {
			t.Fatalf("clamp[%d] = %v, want %v", index, item, want[index])
		}
	}
}

func TestExecuteBF16Round(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(6))
	output := builder.BF16Round(input)
	values := []float32{1, 1.00390625, 1.01171875, -1.01171875, float32(math.Inf(1)), float32(math.NaN())}
	value, _ := NewValue(input.Shape, values)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{input: value})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 1, 1.015625, -1.015625, float32(math.Inf(1)), values[5]}
	for index, item := range results[output].Data {
		if math.Float32bits(item) != math.Float32bits(want[index]) {
			t.Fatalf("BF16 round[%d] = %08x, want %08x", index, math.Float32bits(item), math.Float32bits(want[index]))
		}
	}
}

func TestExecuteXIELU(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(5))
	output := builder.XIELU(input, 0.8, 0.2, 0.5, -0.1)
	value, _ := NewValue(input.Shape, []float32{-2, -0.5, 0, 1.5, 3})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{input: value})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		(float32(math.Expm1(-2))+2)*0.8 - 1,
		(float32(math.Expm1(-0.5))+0.5)*0.8 - 0.25,
		float32(math.Expm1(-0.1)) * 0.8,
		0.2*1.5*1.5 + 0.5*1.5,
		0.2*3*3 + 0.5*3,
	}
	for index, item := range results[output].Data {
		if math.Abs(float64(item-want[index])) > 1e-6 {
			t.Fatalf("xIELU[%d] = %v, want %v", index, item, want[index])
		}
	}
}

func TestExecuteAffineLayerNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(4))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(4))
	output := builder.AffineLayerNorm(input, weight, bias, 1e-5)
	inputValue, _ := NewValue(input.Shape, []float32{
		1, 2, 3, 4,
		-4, -2, 0, 2,
	})
	weightValue, _ := NewValue(weight.Shape, []float32{1, 2, 3, 4})
	biasValue, _ := NewValue(bias.Shape, []float32{0.5, -0.5, 1, -1})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  inputValue,
		weight: weightValue,
		bias:   biasValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	for row := range 2 {
		values := results[output].Data[row*4 : row*4+4]
		for column, value := range values {
			normalized := (value - biasValue.Data[column]) / weightValue.Data[column]
			if math.IsNaN(float64(normalized)) || math.IsInf(float64(normalized), 0) {
				t.Fatalf("LayerNorm row %d contains invalid value %v", row, value)
			}
		}
	}
	normalized := results[output].Data
	if math.Abs(float64(normalized[0]-(-0.841635))) > 1e-5 ||
		math.Abs(float64(normalized[3]-4.36654)) > 1e-5 {
		t.Fatalf("unexpected affine LayerNorm output: %v", normalized)
	}
}

func TestExecuteSigmoidSoftplusAndL2Norm(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(3, 2))
	sigmoid := builder.Sigmoid(input)
	softplus := builder.Softplus(input)
	normalized := builder.L2Norm(builder.Add(sigmoid, softplus), 1e-6)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	value, _ := NewValue(input.Shape, []float32{-100, 0, 100, -2, 1, 3})
	results, err := Execute(
		[]*tensor.Tensor{sigmoid, softplus, normalized},
		map[*tensor.Tensor]Value{input: value},
	)
	if err != nil {
		t.Fatal(err)
	}
	if results[sigmoid].Data[0] != 0 || results[sigmoid].Data[1] != 0.5 ||
		results[softplus].Data[2] != 100 {
		t.Fatalf("unexpected sigmoid/softplus values: %v %v", results[sigmoid].Data, results[softplus].Data)
	}
	for row := range 2 {
		var sumSquares float64
		for _, item := range results[normalized].Data[row*3 : row*3+3] {
			sumSquares += float64(item) * float64(item)
		}
		if math.Abs(sumSquares-1) > 1e-6 {
			t.Fatalf("L2Norm row %d squared norm = %v", row, sumSquares)
		}
	}
}

func TestExecuteRoPENormalWithFrequencyFactors(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 1, 1))
	factors := builder.Input("factors", dtype.F32, tensor.MustShape(2))
	output := builder.RoPENormalWithFactors(input, []uint32{2}, 4, 1, factors)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue, _ := NewValue(input.Shape, []float32{1, 0, 1, 0})
	factorValue, _ := NewValue(factors.Shape, []float32{1, 2})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:   inputValue,
		factors: factorValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		float32(math.Cos(2)),
		float32(math.Sin(2)),
		float32(math.Cos(1)),
		float32(math.Sin(1)),
	}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("RoPE output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteSSMConv(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(5, 2, 1))
	weights := builder.Input("weights", dtype.F32, tensor.MustShape(3, 2))
	output := builder.SSMConv(input, weights)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue, _ := NewValue(input.Shape, []float32{
		1, 2, 3, 4, 5,
		10, 20, 30, 40, 50,
	})
	weightValue, _ := NewValue(weights.Shape, []float32{
		1, 0, -1,
		0.1, 0.2, 0.3,
	})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:   inputValue,
		weights: weightValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-2, 14, -2, 20, -2, 26}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("SSMConv output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteSSMScan(t *testing.T) {
	builder := tensor.NewBuilder()
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 1, 1, 1))
	x := builder.Input("x", dtype.F32, tensor.MustShape(1, 1, 2, 1))
	dt := builder.Input("dt", dtype.F32, tensor.MustShape(1, 2, 1))
	a := builder.Input("a", dtype.F32, tensor.MustShape(1, 1))
	beta := builder.Input("beta", dtype.F32, tensor.MustShape(2, 1, 2, 1))
	c := builder.Input("c", dtype.F32, tensor.MustShape(2, 1, 2, 1))
	output := builder.SSMScan(state, x, dt, a, beta, c)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]Value{}
	feeds[state], _ = NewValue(state.Shape, []float32{1, 2})
	feeds[x], _ = NewValue(x.Shape, []float32{2, 3})
	feeds[dt], _ = NewValue(dt.Shape, []float32{0, 0})
	feeds[a], _ = NewValue(a.Shape, []float32{0})
	feeds[beta], _ = NewValue(beta.Shape, []float32{1, 0, 0, 1})
	feeds[c], _ = NewValue(c.Shape, []float32{1, 1, 2, -1})
	results, err := Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	logTwo := float32(math.Log(2))
	want := []float32{3 + 2*logTwo, logTwo, 1 + 2*logTwo, 2 + 3*logTwo}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("SSMScan output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteGatedDeltaNet(t *testing.T) {
	builder := tensor.NewBuilder()
	q := builder.Input("q", dtype.F32, tensor.MustShape(2, 1, 2, 1))
	k := builder.Input("k", dtype.F32, tensor.MustShape(2, 1, 2, 1))
	v := builder.Input("v", dtype.F32, tensor.MustShape(2, 1, 2, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 1, 2, 1))
	beta := builder.Input("beta", dtype.F32, tensor.MustShape(1, 1, 2, 1))
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 1, 1))
	output := builder.GatedDeltaNet(q, k, v, gate, beta, state)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	qValue, _ := NewValue(q.Shape, []float32{1, 1, 1, -1})
	kValue, _ := NewValue(k.Shape, []float32{1, 0, 0, 1})
	vValue, _ := NewValue(v.Shape, []float32{2, 3, 4, 5})
	gateValue, _ := NewValue(gate.Shape, []float32{0, 0})
	betaValue, _ := NewValue(beta.Shape, []float32{1, 1})
	stateValue, _ := NewValue(state.Shape, []float32{0, 0, 0, 0})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		q: qValue, k: kValue, v: vValue, gate: gateValue,
		beta: betaValue, state: stateValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	rootTwo := float32(math.Sqrt(2))
	want := []float32{
		rootTwo, 3 / rootTwo,
		-rootTwo, -rootTwo,
		2, 4, 3, 5,
	}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("GatedDeltaNet output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteGatedLinearAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(2, 1, 2, 1)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	receptance := builder.Input("receptance", dtype.F32, shape)
	decay := builder.Input("decay", dtype.F32, shape)
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 1, 1))
	output := builder.GatedLinearAttention(key, value, receptance, decay, state, 1)
	feeds := map[*tensor.Tensor]Value{
		key:        {Shape: shape, Data: []float32{1, 2, 2, 1}},
		value:      {Shape: shape, Data: []float32{3, 4, 1, 2}},
		receptance: {Shape: shape, Data: []float32{5, 6, 1, 1}},
		decay:      {Shape: shape, Data: []float32{0.5, 0.25, 0.5, 0.5}},
		state:      {Shape: state.Shape, Data: make([]float32, 4)},
	}
	results, err := Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{34.5, 46, 4.5, 7, 1.75, 2.75, 3, 4}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("GatedLinearAttention output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteRWKV6(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(2, 1, 2, 1)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	receptance := builder.Input("receptance", dtype.F32, shape)
	first := builder.Input("first", dtype.F32, tensor.MustShape(2, 1))
	decay := builder.Input("decay", dtype.F32, shape)
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 1, 1))
	output := builder.RWKV6(key, value, receptance, first, decay, state)
	feeds := map[*tensor.Tensor]Value{
		key:        {Shape: shape, Data: []float32{1, 2, 2, 1}},
		value:      {Shape: shape, Data: []float32{3, 4, 1, 2}},
		receptance: {Shape: shape, Data: []float32{5, 6, 1, 1}},
		first:      {Shape: first.Shape, Data: []float32{0.5, 0.25}},
		decay:      {Shape: shape, Data: []float32{0.5, 0.25, 0.5, 0.5}},
		state:      {Shape: state.Shape, Data: make([]float32, 4)},
	}
	results, err := Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{16.5, 22, 10.25, 14.5, 3.5, 6, 4, 6}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("RWKV6 output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteSumRowsAndRWKV7(t *testing.T) {
	builder := tensor.NewBuilder()
	rows := builder.Input("rows", dtype.F32, tensor.MustShape(3, 2))
	reduced := builder.SumRows(rows)
	shape := tensor.MustShape(2, 1, 1, 1)
	receptance := builder.Input("receptance", dtype.F32, shape)
	decay := builder.Input("decay", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	a := builder.Input("a", dtype.F32, shape)
	bVector := builder.Input("b", dtype.F32, shape)
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 1, 1))
	packed := builder.RWKV7(receptance, decay, key, value, a, bVector, state)
	results, err := Execute([]*tensor.Tensor{reduced, packed}, map[*tensor.Tensor]Value{
		rows:       {Shape: rows.Shape, Data: []float32{1, 2, 3, 4, 5, 6}},
		receptance: {Shape: shape, Data: []float32{5, 6}},
		decay:      {Shape: shape, Data: []float32{0.5, 0.25}},
		key:        {Shape: shape, Data: []float32{1, 2}},
		value:      {Shape: shape, Data: []float32{3, 4}},
		a:          {Shape: shape, Data: []float32{0.1, 0.2}},
		bVector:    {Shape: shape, Data: []float32{0.3, 0.4}},
		state:      {Shape: state.Shape, Data: []float32{1, 2, 3, 4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range []float32{6, 15} {
		if results[reduced].Data[index] != want {
			t.Fatalf("SumRows output[%d] = %v, want %v", index, results[reduced].Data[index], want)
		}
	}
	for index, want := range []float32{58.45, 85.79, 3.65, 6.7, 5.83, 9.44} {
		if math.Abs(float64(results[packed].Data[index]-want)) > 1e-5 {
			t.Fatalf("RWKV7 output[%d] = %v, want %v", index, results[packed].Data[index], want)
		}
	}
}

func TestExecuteSparsePrimitives(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	transformed := builder.FWHT(input)
	indices := builder.TopK(transformed, 2)
	table := builder.Input("table", dtype.F32, tensor.MustShape(2, 4))
	gathered := builder.GatherLast(table, indices)
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 4))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 4))
	attention := builder.SparseAttention(query, key, value, indices, 1)
	causalAttention := builder.SparseAttentionWithOffset(query, key, value, indices, 1, true, 0)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := Execute(
		[]*tensor.Tensor{transformed, indices, gathered, attention, causalAttention},
		map[*tensor.Tensor]Value{
			input: {Shape: input.Shape, Data: []float32{1, 2, 3, 4, 4, 3, 2, 1}},
			table: {Shape: table.Shape, Data: []float32{10, 11, 20, 21, 30, 31, 40, 41}},
			query: {Shape: query.Shape, Data: []float32{1, 1}},
			key:   {Shape: key.Shape, Data: []float32{0, 1, 2, 3}},
			value: {Shape: value.Shape, Data: []float32{10, 20, 30, 40}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	for name, test := range map[string]struct {
		got  []float32
		want []float32
	}{
		"fwht":   {results[transformed].Data, []float32{5, -1, -2, 0, 5, 1, 2, 0}},
		"top-k":  {results[indices].Data, []float32{0, 3, 0, 2}},
		"gather": {results[gathered].Data, []float32{10, 11, 40, 41, 10, 11, 30, 31}},
	} {
		if !reflect.DeepEqual(test.got, test.want) {
			t.Fatalf("%s = %v, want %v", name, test.got, test.want)
		}
	}
	wantAttention := []float32{
		float32((10 + 40*math.Exp(3)) / (1 + math.Exp(3))),
		float32((10 + 30*math.Exp(2)) / (1 + math.Exp(2))),
	}
	for index, want := range wantAttention {
		if got := results[attention].Data[index]; math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("SparseAttention output[%d] = %v, want %v", index, got, want)
		}
	}
	if got := results[causalAttention].Data; !reflect.DeepEqual(got, []float32{10, 10}) {
		t.Fatalf("causal SparseAttention = %v, want [10 10]", got)
	}
}

func TestExecuteGatherLastRejectsFractionalIndex(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 2))
	indices := builder.Input("indices", dtype.F32, tensor.MustShape(1))
	output := builder.GatherLast(input, indices)
	_, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:   {Shape: input.Shape, Data: []float32{1, 2, 3, 4}},
		indices: {Shape: indices.Shape, Data: []float32{0.5}},
	})
	if err == nil {
		t.Fatal("fractional gather index was accepted")
	}
}

func TestExecuteGatedDeltaNetRepeatInterleave(t *testing.T) {
	builder := tensor.NewBuilder()
	q := builder.Input("q", dtype.F32, tensor.MustShape(2, 2, 1, 1))
	k := builder.Input("k", dtype.F32, tensor.MustShape(2, 2, 1, 1))
	v := builder.Input("v", dtype.F32, tensor.MustShape(2, 4, 1, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 4, 1, 1))
	beta := builder.Input("beta", dtype.F32, tensor.MustShape(1, 4, 1, 1))
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 4, 1))
	cyclic := builder.GatedDeltaNet(q, k, v, gate, beta, state)
	interleaved := builder.GatedDeltaNetRepeatInterleave(q, k, v, gate, beta, state)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	qValue, _ := NewValue(q.Shape, []float32{1, 0, 0, 1})
	kValue, _ := NewValue(k.Shape, []float32{1, 0, 1, 0})
	vValue, _ := NewValue(v.Shape, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	gateValue, _ := NewValue(gate.Shape, make([]float32, 4))
	betaValue, _ := NewValue(beta.Shape, []float32{1, 1, 1, 1})
	stateValue, _ := NewValue(state.Shape, make([]float32, 16))
	results, err := Execute([]*tensor.Tensor{cyclic, interleaved}, map[*tensor.Tensor]Value{
		q: qValue, k: kValue, v: vValue, gate: gateValue,
		beta: betaValue, state: stateValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(1 / math.Sqrt(2))
	for index, want := range []float32{scale, scale, scale, scale, 0, 0, 0, 0} {
		if got := results[interleaved].Data[index]; math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("repeat-interleave output[%d] = %v, want %v", index, got, want)
		}
	}
	if reflect.DeepEqual(results[cyclic].Data[:8], results[interleaved].Data[:8]) {
		t.Fatal("cyclic and repeat-interleave head mapping unexpectedly match")
	}
}

func TestExecuteQwen35LayoutOperations(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	query := builder.GroupSlice(input, 0, 2, 2, 4)
	gate := builder.GroupSlice(input, 2, 2, 2, 4)
	transposed := builder.Transpose2D(input)
	state := builder.Input("state", dtype.F32, tensor.MustShape(1, 8))
	convInput := builder.Concat(state, transposed, 0)
	slice := builder.FlatSlice(convInput, 3, 5)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue, _ := NewValue(input.Shape, []float32{
		0, 1, 2, 3, 4, 5, 6, 7,
		10, 11, 12, 13, 14, 15, 16, 17,
	})
	stateValue, _ := NewValue(state.Shape, []float32{
		-1, -2, -3, -4, -5, -6, -7, -8,
	})
	results, err := Execute(
		[]*tensor.Tensor{query, gate, transposed, convInput, slice},
		map[*tensor.Tensor]Value{input: inputValue, state: stateValue},
	)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual := func(name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
		}
		for index := range want {
			if got[index] != want[index] {
				t.Fatalf("%s[%d] = %v, want %v", name, index, got[index], want[index])
			}
		}
	}
	assertEqual("query", results[query].Data, []float32{0, 1, 4, 5, 10, 11, 14, 15})
	assertEqual("gate", results[gate].Data, []float32{2, 3, 6, 7, 12, 13, 16, 17})
	assertEqual("transpose", results[transposed].Data, []float32{
		0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17,
	})
	assertEqual("concat", results[convInput].Data, []float32{
		-1, 0, 10, -2, 1, 11, -3, 2, 12, -4, 3, 13,
		-5, 4, 14, -6, 5, 15, -7, 6, 16, -8, 7, 17,
	})
	assertEqual("flat slice", results[slice].Data, []float32{-2, 1, 11, -3, 2})
}

func TestExecuteAttentionALiBiMatchesLlamaSlope(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 1))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 2))
	output := builder.AttentionALiBiWithOffset(query, key, value, 1, 2, true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	queryValue, _ := NewValue(query.Shape, []float32{0})
	keyValue, _ := NewValue(key.Shape, []float32{0, 0})
	valueValue, _ := NewValue(value.Shape, []float32{1, 3})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: queryValue, key: keyValue, value: valueValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	older := math.Exp(-0.25) // one-token distance times the single-head 2^-2 slope
	want := float32((older + 3) / (older + 1))
	if got := results[output].Data[0]; math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("ALiBi attention = %v, want %v", got, want)
	}
}

func TestExecuteMoETopKNormalization(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	output := builder.MoE(input, router, gate, up, down, 2, true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	values := func(shape tensor.Shape, data []float32) Value {
		value, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  values(input.Shape, []float32{1, 0}),
		router: values(router.Shape, []float32{1, 0, 0, 0}),
		gate:   values(gate.Shape, []float32{1, 0, 2, 0}),
		up:     values(up.Shape, []float32{1, 0, 1, 0}),
		down:   values(down.Shape, []float32{1, 2, 3, 4}),
	})
	if err != nil {
		t.Fatal(err)
	}
	p0 := math.Exp(1) / (math.Exp(1) + 1)
	p1 := 1 - p0
	wantActivation0 := 1 / (1 + math.Exp(-1))
	wantActivation1 := 2 / (1 + math.Exp(-2))
	want := []float32{
		float32(p0*wantActivation0 + p1*3*wantActivation1),
		float32(p0*2*wantActivation0 + p1*4*wantActivation1),
	}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-5 {
			t.Fatalf("MoE output[%d] = %v, want %v", index, got, want[index])
		}
	}
}

func TestExecuteUngatedMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	output := builder.MoEUngated(input, router, up, down, 2, true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	values := func(shape tensor.Shape, data []float32) Value {
		value, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  values(input.Shape, []float32{1, 0}),
		router: values(router.Shape, []float32{1, 0, 0, 0}),
		up:     values(up.Shape, []float32{1, 0, 2, 0}),
		down:   values(down.Shape, []float32{1, 2, 3, 4}),
	})
	if err != nil {
		t.Fatal(err)
	}
	p0 := math.Exp(1) / (math.Exp(1) + 1)
	p1 := 1 - p0
	wantActivation0 := 1 / (1 + math.Exp(-1))
	wantActivation1 := 2 / (1 + math.Exp(-2))
	want := []float32{
		float32(p0*wantActivation0 + p1*3*wantActivation1),
		float32(p0*2*wantActivation0 + p1*4*wantActivation1),
	}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-5 {
			t.Fatalf("ungated MoE output[%d] = %v, want %v", index, got, want[index])
		}
	}
}

func TestExecuteUngatedMoEWithSelectionBias(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(2))
	output := builder.MoEUngatedWithSelectionBias(input, router, up, down, bias, 1, true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	values := func(shape tensor.Shape, data []float32) Value {
		value, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  values(input.Shape, []float32{1, 0}),
		router: values(router.Shape, []float32{1, 0, 0, 0}),
		up:     values(up.Shape, []float32{1, 0, 2, 0}),
		down:   values(down.Shape, []float32{1, 2, 3, 4}),
		bias:   values(bias.Shape, []float32{-1, 1}),
	})
	if err != nil {
		t.Fatal(err)
	}
	activation := 2 / (1 + math.Exp(-2))
	want := []float32{float32(3 * activation), float32(4 * activation)}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-5 {
			t.Fatalf("biased ungated MoE output[%d] = %v, want %v", index, got, want[index])
		}
	}
}

func TestExecuteReLUMoEWithRouterInput(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	routerInput := builder.Input("router-input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	output := builder.MoEReLUWithRouterInput(
		input, routerInput, router, gate, up, down, 1, true, 1, tensor.MoERoutingSoftmax,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	values := func(shape tensor.Shape, data []float32) Value {
		value, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:       values(input.Shape, []float32{-2, 1}),
		routerInput: values(routerInput.Shape, []float32{1, 0}),
		router:      values(router.Shape, []float32{1, 0, 0, 0}),
		gate:        values(gate.Shape, []float32{0, 1, 0, 0}),
		up:          values(up.Shape, []float32{0, 2, 0, 0}),
		down:        values(down.Shape, []float32{3, 4, 0, 0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{6, 8}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-6 {
			t.Fatalf("ReLU MoE output[%d] = %v, want %v", index, got, want[index])
		}
	}
}

func TestExecuteGroupedMoEWithRouterInput(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	routerInput := builder.Input("router-input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 4))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	output := builder.MoEGroupedWithRouterInput(
		input, routerInput, router, gate, up, down, 1, true, 1, 2,
	)
	value := func(shape tensor.Shape, data []float32) Value {
		result, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:       value(input.Shape, []float32{1, 2}),
		routerInput: value(routerInput.Shape, []float32{1, 0}),
		router:      value(router.Shape, []float32{0, 0, 1, 0, 3, 0, 2, 0}),
		gate:        value(gate.Shape, []float32{1, 0, 2, 0}),
		up:          value(up.Shape, []float32{0, 1, 0, 2}),
		down:        value(down.Shape, []float32{3, 4, 5, 6}),
	})
	if err != nil {
		t.Fatal(err)
	}
	activation := float32(2 / (1 + math.Exp(-2)))
	want := []float32{20 * activation, 24 * activation}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-5 {
			t.Fatalf("grouped MoE output[%d] = %v, want %v", index, got, want[index])
		}
	}
}

func TestExecuteGELUMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 1))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 1))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 1))
	output := builder.MoEGELU(input, router, gate, up, down, 1, true, 1)
	value := func(shape tensor.Shape, data []float32) Value {
		result, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  value(input.Shape, []float32{1, 2}),
		router: value(router.Shape, []float32{0, 0}),
		gate:   value(gate.Shape, []float32{1, 0}),
		up:     value(up.Shape, []float32{0, 1}),
		down:   value(down.Shape, []float32{3, 4}),
	})
	if err != nil {
		t.Fatal(err)
	}
	gateDot := float16Round(1)
	wantActivation := float16Round(float32(0.5*float64(gateDot)*
		(1+math.Tanh(math.Sqrt(2/math.Pi)*float64(gateDot)*(1+0.044715*float64(gateDot*gateDot)))))) * 2
	want := []float32{3 * wantActivation, 4 * wantActivation}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-6 {
			t.Fatalf("GELU MoE output[%d] = %v, want %v", index, got, want[index])
		}
	}
}

func TestExecuteGELUMoEExpertScale(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(1, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 1, 1))
	up := builder.Input("up", dtype.F32, tensor.MustShape(1, 1, 1))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 1, 1))
	scale := builder.Input("scale", dtype.F32, tensor.MustShape(1))
	output := builder.MoEGELUWithRouterInput(
		input, input, router, gate, up, down, scale, 1, true, 1,
	)
	makeValue := func(shape tensor.Shape, data []float32) Value {
		result, err := NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	feeds := map[*tensor.Tensor]Value{
		input:  makeValue(input.Shape, []float32{1}),
		router: makeValue(router.Shape, []float32{0}),
		gate:   makeValue(gate.Shape, []float32{1}),
		up:     makeValue(up.Shape, []float32{1}),
		down:   makeValue(down.Shape, []float32{3}),
		scale:  makeValue(scale.Shape, []float32{2}),
	}
	results, err := Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	gelu := moeGELU(1)
	want := float32(6 * gelu)
	if got := results[output].Data[0]; math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("scaled GELU MoE = %g, want %g", got, want)
	}
}

func TestExecuteRoPEMulti(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 1, 1))
	positions := [4][]uint32{
		{1},
		{2},
		{3},
		{4},
	}
	output := builder.RoPEMultiScaled(
		input,
		positions,
		[4]int32{2, 2, 2, 2},
		8,
		10000,
		0.25,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue, _ := NewValue(input.Shape, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	results, err := Execute(
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]Value{input: inputValue},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 8)
	for pair, position := range []float64{1, 1, 2, 2} {
		theta := position * 0.25 * math.Pow(10000, -2*float64(pair)/8)
		cosine, sine := float32(math.Cos(theta)), float32(math.Sin(theta))
		first := pair * 2
		want[first] = inputValue.Data[first]*cosine - inputValue.Data[first+1]*sine
		want[first+1] = inputValue.Data[first]*sine + inputValue.Data[first+1]*cosine
	}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("RoPEMulti output[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteMulMatGGMLSemantics(t *testing.T) {
	builder := tensor.NewBuilder()
	left := builder.Input("weights", dtype.F32, tensor.MustShape(3, 2))
	right := builder.Input("input", dtype.F32, tensor.MustShape(3, 2))
	output := builder.MulMat(left, right)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	leftValue, _ := NewValue(left.Shape, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	rightValue, _ := NewValue(right.Shape, []float32{
		1, 0, 0,
		0, 1, 0,
	})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		left:  leftValue,
		right: rightValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 4, 2, 5}
	got := results[output].Data
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestExecuteEmbeddingWeightedNormAndSwiGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	table := builder.Input("table", dtype.F32, tensor.MustShape(4, 3))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(4))
	up := builder.Input("up", dtype.F32, tensor.MustShape(4, 2))
	embedding := builder.GetRows(table, []uint32{2, 0})
	output := builder.SwiGLU(builder.WeightedRMSNorm(embedding, weight, 1e-5), up)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	tableValue, _ := NewValue(table.Shape, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	weightValue, _ := NewValue(weight.Shape, []float32{1, 2, 3, 4})
	upValue, _ := NewValue(up.Shape, []float32{1, 1, 1, 1, 2, 2, 2, 2})
	results, err := Execute([]*tensor.Tensor{embedding, output}, map[*tensor.Tensor]Value{
		table:  tableValue,
		weight: weightValue,
		up:     upValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantEmbedding := []float32{9, 10, 11, 12, 1, 2, 3, 4}
	for i, want := range wantEmbedding {
		if results[embedding].Data[i] != want {
			t.Fatalf("embedding[%d] = %v, want %v", i, results[embedding].Data[i], want)
		}
	}
	if len(results[output].Data) != 8 {
		t.Fatalf("output length = %d", len(results[output].Data))
	}
}

func TestExecuteRoPENeoX(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 1, 2))
	output := builder.RoPENeoX(input, []uint32{0, 1}, 4, 10000)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue, _ := NewValue(input.Shape, []float32{
		1, 2, 3, 4,
		1, 2, 3, 4,
	})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{input: inputValue})
	if err != nil {
		t.Fatal(err)
	}
	got := results[output].Data
	for i, want := range []float32{1, 2, 3, 4} {
		if got[i] != want {
			t.Fatalf("position-zero value[%d] = %v, want %v", i, got[i], want)
		}
	}
	cosine := float32(math.Cos(1))
	sine := float32(math.Sin(1))
	want := []float32{
		1*cosine - 3*sine,
		2*float32(math.Cos(0.01)) - 4*float32(math.Sin(0.01)),
		1*sine + 3*cosine,
		2*float32(math.Sin(0.01)) + 4*float32(math.Cos(0.01)),
	}
	for i := range want {
		if difference := math.Abs(float64(got[4+i] - want[i])); difference > 1e-6 {
			t.Fatalf("position-one value[%d] = %v, want %v", i, got[4+i], want[i])
		}
	}
}

func TestExecuteRoPENormal(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 1, 2))
	output := builder.RoPENormal(input, []uint32{0, 1}, 4, 10000)
	feed, err := NewValue(input.Shape, []float32{
		1, 2, 3, 4,
		1, 2, 3, 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{input: feed})
	if err != nil {
		t.Fatal(err)
	}
	got := results[output].Data
	want := []float32{
		1, 2, 3, 4,
		-1.1426396, 1.9220756, 2.9598503, 4.0297995,
	}
	for index := range want {
		if difference := math.Abs(float64(got[index] - want[index])); difference > 1e-5 {
			t.Fatalf("normal RoPE value[%d] = %v, want %v", index, got[index], want[index])
		}
	}
}

func TestExecuteYaRNRoPEAppliesMagnitudeAndFrequencyBlend(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 1, 1))
	yarn := builder.RoPENeoXYaRN(input, []uint32{17}, 4, 8, 10_000, 0.25, 1, 1, 32, 1)
	linear := builder.RoPENeoXScaled(input, []uint32{17}, 4, 10_000, 0.25)
	feed, _ := NewValue(input.Shape, []float32{1, 2, 3, 4})
	results, err := Execute([]*tensor.Tensor{yarn, linear}, map[*tensor.Tensor]Value{input: feed})
	if err != nil {
		t.Fatal(err)
	}
	var differs bool
	for index, value := range results[yarn].Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("YaRN output[%d] is non-finite", index)
		}
		differs = differs || math.Abs(float64(value-results[linear].Data[index])) > 1e-5
	}
	if !differs {
		t.Fatal("YaRN output equals pure linear interpolation")
	}
}

func TestExecuteCausalGroupedQueryAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 2, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(2, 1, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(2, 1, 2))
	output := builder.Attention(query, key, value, 1, true)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	queryValue, _ := NewValue(query.Shape, make([]float32, 8))
	keyValue, _ := NewValue(key.Shape, make([]float32, 4))
	valueValue, _ := NewValue(value.Shape, []float32{1, 2, 3, 4})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: queryValue,
		key:   keyValue,
		value: valueValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		1, 2, 1, 2,
		2, 3, 2, 3,
	}
	for i := range want {
		if results[output].Data[i] != want[i] {
			t.Fatalf("attention[%d] = %v, want %v", i, results[output].Data[i], want[i])
		}
	}
}

func TestExecuteWindowAttentionWithBidirectionalBlocks(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(1, 1, 5)
	query := builder.Input("query", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	blocks := builder.Input("blocks", dtype.F32, tensor.MustShape(5))
	output := builder.AttentionWindowWithBlockMaskWithOffset(
		query, key, value, blocks, 1, 0, 3,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query:  {Shape: shape, Data: make([]float32, 5)},
		key:    {Shape: shape, Data: make([]float32, 5)},
		value:  {Shape: shape, Data: []float32{1, 2, 4, 8, 16}},
		blocks: {Shape: blocks.Shape, Data: []float32{-1, 0, 0, -1, -1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 7.0 / 3, 7.0 / 3, 14.0 / 3, 28.0 / 3}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("block attention[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteAttentionSinksAddHiddenLogit(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 1))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 2))
	sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(1))
	output := builder.AttentionWithSinksWithOffset(query, key, value, sinks, 1, false, 0)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: query.Shape, Data: []float32{0}},
		key:   {Shape: key.Shape, Data: []float32{0, 0}},
		value: {Shape: value.Shape, Data: []float32{2, 4}},
		sinks: {Shape: sinks.Shape, Data: []float32{float32(math.Log(2))}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if difference := math.Abs(float64(results[output].Data[0] - 1.5)); difference > 1e-6 {
		t.Fatalf("sink attention = %v, want 1.5", results[output].Data[0])
	}
}

func TestExecuteRepeatHeads(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1, 2))
	output := builder.RepeatHeads(input, 3)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input: {Shape: input.Shape, Data: []float32{1, 2, 3, 4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}
	for index := range want {
		if results[output].Data[index] != want[index] {
			t.Fatalf("RepeatHeads[%d] = %v, want %v", index, results[output].Data[index], want[index])
		}
	}
}

func TestExecuteSigmoidMoEUsesBiasOnlyForSelection(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(1, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(1, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 1, 2))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(2))
	output := builder.MoESigmoid(input, router, gate, up, down, bias, 1, true, 1)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  {Shape: input.Shape, Data: []float32{1}},
		router: {Shape: router.Shape, Data: []float32{2, 1}},
		gate:   {Shape: gate.Shape, Data: []float32{1, 2}},
		up:     {Shape: up.Shape, Data: []float32{1, 1}},
		down:   {Shape: down.Shape, Data: []float32{1, 3}},
		bias:   {Shape: bias.Shape, Data: []float32{-1, 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := float32(6 / (1 + math.Exp(-2)))
	if difference := math.Abs(float64(results[output].Data[0] - want)); difference > 1e-6 {
		t.Fatalf("sigmoid MoE output = %v, want %v", results[output].Data[0], want)
	}
}

func TestExecuteLimitedSigmoidMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(1, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 1, 1))
	up := builder.Input("up", dtype.F32, tensor.MustShape(1, 1, 1))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 1, 1))
	output := builder.MoESigmoidLimited(input, router, gate, up, down, nil, 1, true, 1, 1)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:  {Shape: input.Shape, Data: []float32{2}},
		router: {Shape: router.Shape, Data: []float32{0}},
		gate:   {Shape: gate.Shape, Data: []float32{1}},
		up:     {Shape: up.Shape, Data: []float32{2}},
		down:   {Shape: down.Shape, Data: []float32{1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if difference := math.Abs(float64(results[output].Data[0] - 1)); difference > 1e-6 {
		t.Fatalf("limited sigmoid MoE output = %v, want 1", results[output].Data[0])
	}
}

func TestExecuteFusedGateUpMoEMatchesSeparate(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(1, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(1, 1, 2))
	gateUp := builder.Input("gate_up", dtype.F32, tensor.MustShape(1, 2, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 1, 2))
	separate := builder.MoESigmoid(input, router, gate, up, down, nil, 2, true, 1)
	fused := builder.MoESigmoidFusedGateUp(input, router, gateUp, down, nil, 2, true, 1)
	results, err := Execute([]*tensor.Tensor{separate, fused}, map[*tensor.Tensor]Value{
		input:  {Shape: input.Shape, Data: []float32{1}},
		router: {Shape: router.Shape, Data: []float32{2, 1}},
		gate:   {Shape: gate.Shape, Data: []float32{1, 2}},
		up:     {Shape: up.Shape, Data: []float32{3, 4}},
		gateUp: {Shape: gateUp.Shape, Data: []float32{1, 3, 2, 4}},
		down:   {Shape: down.Shape, Data: []float32{1, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if difference := math.Abs(float64(results[separate].Data[0] - results[fused].Data[0])); difference > 1e-6 {
		t.Fatalf("fused MoE output = %v, separate = %v", results[fused].Data[0], results[separate].Data[0])
	}
}

func TestExecuteSoftcappedAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 1))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 2))
	output := builder.AttentionSoftcappedWithOffset(query, key, value, 1, 2, false, 0)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: query.Shape, Data: []float32{1}},
		key:   {Shape: key.Shape, Data: []float32{10, -10}},
		value: {Shape: value.Shape, Data: []float32{1, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	capped := 2 * math.Tanh(5)
	want := float32((math.Exp(capped) + 3*math.Exp(-capped)) /
		(math.Exp(capped) + math.Exp(-capped)))
	if difference := math.Abs(float64(results[output].Data[0] - want)); difference > 1e-6 {
		t.Fatalf("softcapped attention = %v, want %v", results[output].Data[0], want)
	}
}

func TestExecuteCachedAttentionAndConcat(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 2, 1))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(2, 1, 2))
	newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(2, 1, 1))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(2, 1, 2))
	newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(2, 1, 1))
	key := builder.Concat(pastKey, newKey, 2)
	value := builder.Concat(pastValue, newValue, 2)
	output := builder.AttentionWithOffset(query, key, value, 1, true, 2)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	zeroQuery, _ := NewValue(query.Shape, make([]float32, 4))
	zeroPastKey, _ := NewValue(pastKey.Shape, make([]float32, 4))
	zeroNewKey, _ := NewValue(newKey.Shape, make([]float32, 2))
	pastValueData, _ := NewValue(pastValue.Shape, []float32{1, 2, 3, 4})
	newValueData, _ := NewValue(newValue.Shape, []float32{5, 6})
	results, err := Execute([]*tensor.Tensor{key, value, output}, map[*tensor.Tensor]Value{
		query:     zeroQuery,
		pastKey:   zeroPastKey,
		newKey:    zeroNewKey,
		pastValue: pastValueData,
		newValue:  newValueData,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := results[key].Shape.Dims[2]; got != 3 {
		t.Fatalf("cached key token count = %d, want 3", got)
	}
	want := []float32{3, 4, 3, 4}
	for index := range want {
		if results[output].Data[index] != want[index] {
			t.Fatalf("cached attention[%d] = %v, want %v", index, results[output].Data[index], want[index])
		}
	}
}

func TestExecuteBatchedCachedAttentionAndMiddleConcat(t *testing.T) {
	const (
		width      = 1
		heads      = 1
		pastTokens = 2
		newTokens  = 1
		sequences  = 2
	)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(width, heads, newTokens, sequences))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(width, heads, pastTokens, sequences))
	newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(width, heads, newTokens, sequences))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(width, heads, pastTokens, sequences))
	newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(width, heads, newTokens, sequences))
	key := builder.Concat(pastKey, newKey, 2)
	value := builder.Concat(pastValue, newValue, 2)
	output := builder.AttentionWithOffset(query, key, value, 1, true, pastTokens)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := Execute([]*tensor.Tensor{value, output}, map[*tensor.Tensor]Value{
		query:     {Shape: query.Shape, Data: make([]float32, sequences)},
		pastKey:   {Shape: pastKey.Shape, Data: make([]float32, pastTokens*sequences)},
		newKey:    {Shape: newKey.Shape, Data: make([]float32, newTokens*sequences)},
		pastValue: {Shape: pastValue.Shape, Data: []float32{1, 3, 10, 30}},
		newValue:  {Shape: newValue.Shape, Data: []float32{5, 50}},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCache := []float32{1, 3, 5, 10, 30, 50}
	if !slices.Equal(results[value].Data, wantCache) {
		t.Fatalf("batched cache = %v, want %v", results[value].Data, wantCache)
	}
	wantOutput := []float32{3, 30}
	if !slices.Equal(results[output].Data, wantOutput) {
		t.Fatalf("batched attention = %v, want %v", results[output].Data, wantOutput)
	}
}

func TestExecuteSymmetricWindowAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(1, 1, 5)
	query := builder.Input("query", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	output := builder.AttentionSymmetricWindow(query, key, value, 1, 4)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: shape, Data: []float32{0, 0, 0, 0, 0}},
		key:   {Shape: shape, Data: []float32{0, 0, 0, 0, 0}},
		value: {Shape: shape, Data: []float32{1, 2, 4, 8, 16}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{7.0 / 3, 15.0 / 4, 31.0 / 5, 30.0 / 4, 28.0 / 3}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("symmetric attention[%d] = %v, want %v", index, value, want[index])
		}
	}
	attributes := output.Attrs.(tensor.AttentionAttributes)
	if !attributes.SymmetricWindow || attributes.Causal || attributes.Window != 4 {
		t.Fatalf("unexpected symmetric attention attributes: %+v", attributes)
	}
}

func TestExecuteSymmetricWindowAttentionWithSinks(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(1, 1, 5)
	query := builder.Input("query", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(1))
	output := builder.AttentionSymmetricWindowWithSinks(query, key, value, sinks, 1, 4)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: shape, Data: make([]float32, 5)},
		key:   {Shape: shape, Data: make([]float32, 5)},
		value: {Shape: shape, Data: []float32{1, 2, 4, 8, 16}},
		sinks: {Shape: sinks.Shape, Data: []float32{0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{7.0 / 4, 15.0 / 5, 31.0 / 6, 30.0 / 5, 28.0 / 4}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("symmetric sink attention[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteChunkedWindowAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(1, 1, 6)
	query := builder.Input("query", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	output := builder.AttentionChunkedWindowWithOffset(query, key, value, 1, true, 0, 4)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: shape, Data: make([]float32, 6)},
		key:   {Shape: shape, Data: make([]float32, 6)},
		value: {Shape: shape, Data: []float32{1, 2, 4, 8, 16, 32}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 1.5, 7.0 / 3, 15.0 / 4, 16, 24}
	for index, value := range results[output].Data {
		if math.Abs(float64(value-want[index])) > 1e-6 {
			t.Fatalf("chunked attention[%d] = %v, want %v", index, value, want[index])
		}
	}
	attributes := output.Attrs.(tensor.AttentionAttributes)
	if !attributes.ChunkedWindow || !attributes.Causal || attributes.Window != 4 {
		t.Fatalf("unexpected chunked attention attributes: %+v", attributes)
	}
}

func TestExecuteOpenAIMoEUsesSelectedSoftmaxAndBiases(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	routerBias := builder.Input("router_bias", dtype.F32, tensor.MustShape(2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 2))
	gateBias := builder.Input("gate_bias", dtype.F32, tensor.MustShape(1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	upBias := builder.Input("up_bias", dtype.F32, tensor.MustShape(1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	downBias := builder.Input("down_bias", dtype.F32, tensor.MustShape(2, 2))
	output := builder.MoEOpenAI(input, router, routerBias, gate, gateBias, up, upBias, down, downBias, 2, 1)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input:      {Shape: input.Shape, Data: []float32{1, 2}},
		router:     {Shape: router.Shape, Data: make([]float32, 4)},
		routerBias: {Shape: routerBias.Shape, Data: []float32{0, float32(math.Log(3))}},
		gate:       {Shape: gate.Shape, Data: make([]float32, 4)},
		gateBias:   {Shape: gateBias.Shape, Data: []float32{1, 2}},
		up:         {Shape: up.Shape, Data: make([]float32, 4)},
		upBias:     {Shape: upBias.Shape, Data: []float32{0, 1}},
		down:       {Shape: down.Shape, Data: []float32{1, 0, 0, 1}},
		downBias:   {Shape: downBias.Shape, Data: []float32{0, 1, 2, 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	activation0 := 1 / (1 + math.Exp(-1.702))
	activation1 := 2 / (1 + math.Exp(-3.404)) * 2
	want := []float64{0.25*activation0 + 0.75*2, 0.25 + 0.75*activation1}
	for index, value := range results[output].Data {
		if math.Abs(float64(value)-want[index]) > 1e-6 {
			t.Fatalf("OpenAI MoE[%d] = %v, want %v", index, value, want[index])
		}
	}
	attributes := output.Attrs.(tensor.MoEAttributes)
	if attributes.Routing != tensor.MoERoutingSelectedSoftmax ||
		attributes.Activation != tensor.MoEActivationSwiGLUOAI ||
		!attributes.HasRouterBias || !attributes.HasExpertBiases {
		t.Fatalf("unexpected OpenAI MoE attributes: %+v", attributes)
	}
}

func TestExecuteDeepSeek4RawAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 1, 1))
	cache := builder.Input("cache", dtype.F32, tensor.MustShape(2, 1, 1))
	positions := builder.Input("positions", dtype.F32, tensor.MustShape(1, 1, 1))
	sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(1))
	output := builder.DeepSeek4Attention(query, cache, positions, sinks, nil, nil, nil, nil, nil, nil, nil, nil,
		tensor.DeepSeek4AttentionAttributes{
			Positions: []uint32{0}, Window: 4, Heads: 1, RotaryDimensions: 2,
			FrequencyBase: 10000, FrequencyScale: 1, AttentionFactor: 1, NormEpsilon: 1e-5,
		})
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query:     {Shape: query.Shape, Data: []float32{1, 0}},
		cache:     {Shape: cache.Shape, Data: []float32{3, 4}},
		positions: {Shape: positions.Shape, Data: []float32{0}},
		sinks:     {Shape: sinks.Shape, Data: []float32{-100}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range []float32{3, 4} {
		if difference := math.Abs(float64(results[output].Data[index] - want)); difference > 1e-6 {
			t.Fatalf("DeepSeek 4 raw attention[%d] = %v, want %v", index, results[output].Data[index], want)
		}
	}
}

func TestExecuteDeepSeek4RawAttentionHonorsPositionGaps(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 1, 1))
	sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(1))
	cacheWithGap := builder.Input("cache_with_gap", dtype.F32, tensor.MustShape(2, 1, 3))
	positionsWithGap := builder.Input("positions_with_gap", dtype.F32, tensor.MustShape(1, 1, 3))
	cacheWithoutOld := builder.Input("cache_without_old", dtype.F32, tensor.MustShape(2, 1, 2))
	positionsWithoutOld := builder.Input("positions_without_old", dtype.F32, tensor.MustShape(1, 1, 2))
	attributes := tensor.DeepSeek4AttentionAttributes{
		Positions: []uint32{3}, Window: 3, Heads: 1, RotaryDimensions: 2,
		FrequencyBase: 10000, FrequencyScale: 1, AttentionFactor: 1, NormEpsilon: 1e-5,
	}
	withGap := builder.DeepSeek4Attention(query, cacheWithGap, positionsWithGap, sinks,
		nil, nil, nil, nil, nil, nil, nil, nil, attributes)
	withoutOld := builder.DeepSeek4Attention(query, cacheWithoutOld, positionsWithoutOld, sinks,
		nil, nil, nil, nil, nil, nil, nil, nil, attributes)
	results, err := Execute([]*tensor.Tensor{withGap, withoutOld}, map[*tensor.Tensor]Value{
		query:               {Shape: query.Shape, Data: []float32{0, 0}},
		sinks:               {Shape: sinks.Shape, Data: []float32{-100}},
		cacheWithGap:        {Shape: cacheWithGap.Shape, Data: []float32{100, 25, 2, 1, 4, 3}},
		positionsWithGap:    {Shape: positionsWithGap.Shape, Data: []float32{0, 2, 3}},
		cacheWithoutOld:     {Shape: cacheWithoutOld.Shape, Data: []float32{2, 1, 4, 3}},
		positionsWithoutOld: {Shape: positionsWithoutOld.Shape, Data: []float32{2, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range results[withoutOld].Data {
		if difference := math.Abs(float64(results[withGap].Data[index] - want)); difference > 1e-6 {
			t.Fatalf("DeepSeek 4 gapped attention[%d] = %v, want %v", index, results[withGap].Data[index], want)
		}
	}
}

func TestDeepSeek4CompressedBlocksSkipIncompletePositionGroup(t *testing.T) {
	shape := tensor.MustShape(2, 1, 4)
	kv := Value{Shape: shape, Data: make([]float32, 8)}
	score := Value{Shape: shape, Data: make([]float32, 8)}
	norm := Value{Shape: tensor.MustShape(1), Data: []float32{1}}
	attributes := tensor.DeepSeek4AttentionAttributes{RotaryDimensions: 0, NormEpsilon: 1e-5}
	blocks, err := deepSeek4CompressedBlocks(kv, score, norm, 4, 1, []uint32{0, 1, 3, 4}, attributes)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 0 {
		t.Fatalf("incomplete DeepSeek 4 compressed blocks = %d", len(blocks))
	}
}

func TestExecuteDeepSeek4CompressedAttention(t *testing.T) {
	for _, test := range []struct {
		name  string
		ratio tensor.DeepSeek4CompressionRatio
	}{
		{"overlap", tensor.DeepSeek4CompressionOverlap},
		{"wide", tensor.DeepSeek4CompressionWide},
	} {
		t.Run(test.name, func(t *testing.T) {
			ratio := test.ratio
			ratioWidth := uint32(ratio)
			builder := tensor.NewBuilder()
			position := ratioWidth - 1
			query := builder.Input("query", dtype.F32, tensor.MustShape(2, 1, 1))
			cache := builder.Input("cache", dtype.F32, tensor.MustShape(2, 1, uint64(ratioWidth)))
			positions := builder.Input("positions", dtype.F32, tensor.MustShape(1, 1, uint64(ratioWidth)))
			sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(1))
			coefficient := ratio.KVWidthMultiplier()
			compressorKV := builder.Input("compressor_kv", dtype.F32, tensor.MustShape(2*coefficient, 1, uint64(ratioWidth)))
			compressorScore := builder.Input("compressor_score", dtype.F32, compressorKV.Shape)
			compressorNorm := builder.Input("compressor_norm", dtype.F32, tensor.MustShape(2))
			var indexerQuery, indexerWeights, indexerKV, indexerScore, indexerNorm *tensor.Tensor
			if ratio.UsesIndexer() {
				indexerQuery = builder.Input("indexer_query", dtype.F32, tensor.MustShape(2, 1, 1))
				indexerWeights = builder.Input("indexer_weights", dtype.F32, tensor.MustShape(1, 1))
				indexerKV = builder.Input("indexer_kv", dtype.F32, tensor.MustShape(4, 1, 4))
				indexerScore = builder.Input("indexer_score", dtype.F32, indexerKV.Shape)
				indexerNorm = builder.Input("indexer_norm", dtype.F32, tensor.MustShape(2))
			}
			output := builder.DeepSeek4Attention(
				query, cache, positions, sinks, compressorKV, compressorScore, compressorNorm,
				indexerQuery, indexerWeights, indexerKV, indexerScore, indexerNorm,
				tensor.DeepSeek4AttentionAttributes{
					Positions: []uint32{position}, Ratio: ratio, Window: 1, Heads: 1,
					IndexerHeads: 1, IndexerTopK: 1, RotaryDimensions: 2,
					FrequencyBase: 10000, FrequencyScale: 1, AttentionFactor: 1, NormEpsilon: 1e-5,
				},
			)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			cacheData := make([]float32, 2*ratioWidth)
			for token := uint32(0); token < ratioWidth; token++ {
				cacheData[2*token] = 1
			}
			compressorData := make([]float32, 2*uint32(coefficient)*ratioWidth)
			for token := uint32(0); token < ratioWidth; token++ {
				compressorData[token*2*uint32(coefficient)+2*(uint32(coefficient)-1)] = 2
			}
			feeds := map[*tensor.Tensor]Value{
				query: {Shape: query.Shape, Data: []float32{0, 0}}, cache: {Shape: cache.Shape, Data: cacheData},
				sinks:           {Shape: sinks.Shape, Data: []float32{-100}},
				compressorKV:    {Shape: compressorKV.Shape, Data: compressorData},
				compressorScore: {Shape: compressorScore.Shape, Data: make([]float32, len(compressorData))},
				compressorNorm:  {Shape: compressorNorm.Shape, Data: []float32{1, 1}},
			}
			positionData := make([]float32, ratioWidth)
			for index := range positionData {
				positionData[index] = float32(index)
			}
			feeds[positions] = Value{Shape: positions.Shape, Data: positionData}
			if ratio.UsesIndexer() {
				feeds[indexerQuery] = Value{Shape: indexerQuery.Shape, Data: []float32{0, 0}}
				feeds[indexerWeights] = Value{Shape: indexerWeights.Shape, Data: []float32{1}}
				feeds[indexerKV] = Value{Shape: indexerKV.Shape, Data: append([]float32(nil), compressorData...)}
				feeds[indexerScore] = Value{Shape: indexerScore.Shape, Data: make([]float32, len(compressorData))}
				feeds[indexerNorm] = Value{Shape: indexerNorm.Shape, Data: []float32{1, 1}}
			}
			results, err := Execute([]*tensor.Tensor{output}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			compressed := 2 / math.Sqrt(2+1e-5)
			want := []float64{
				(1 + compressed*math.Cos(float64(position))) / 2,
				-compressed * math.Sin(float64(position)) / 2,
			}
			for index, value := range results[output].Data {
				if math.Abs(float64(value)-want[index]) > 1e-5 {
					t.Fatalf("DeepSeek 4 ratio %d attention[%d] = %v, want %v", ratio, index, value, want[index])
				}
			}
		})
	}
}

func TestExecuteDeepSeek4HCAndFixedRouting(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	hcInput := builder.DeepSeek4HCInit(input, 2)
	fn := builder.Input("hc_fn", dtype.F32, tensor.MustShape(4, 2))
	scale := builder.Input("hc_scale", dtype.F32, tensor.MustShape(1))
	base := builder.Input("hc_base", dtype.F32, tensor.MustShape(2))
	head := builder.DeepSeek4HCHead(hcInput, fn, scale, base, 2, 1e-5, 1e-6)
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 2))
	selected := builder.Input("selected", dtype.F32, tensor.MustShape(1, 1))
	moe := builder.MoESqrtSoftplusLimited(input, router, gate, up, down, nil, selected, 1, true, 1, 1)
	results, err := Execute([]*tensor.Tensor{head, moe}, map[*tensor.Tensor]Value{
		input: {Shape: input.Shape, Data: []float32{2, 4}},
		fn:    {Shape: fn.Shape, Data: make([]float32, 8)}, scale: {Shape: scale.Shape, Data: []float32{0}},
		base: {Shape: base.Shape, Data: []float32{0, 0}}, router: {Shape: router.Shape, Data: make([]float32, 4)},
		gate: {Shape: gate.Shape, Data: []float32{0, 0, 2, 0}}, up: {Shape: up.Shape, Data: []float32{0, 0, 1, 0}},
		down: {Shape: down.Shape, Data: []float32{0, 0, 3, 0}}, selected: {Shape: selected.Shape, Data: []float32{1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range []float32{2.000004, 4.000008} {
		if difference := math.Abs(float64(results[head].Data[index] - want)); difference > 1e-5 {
			t.Fatalf("DeepSeek 4 HC head[%d] = %v, want %v", index, results[head].Data[index], want)
		}
	}
	wantMoE := float32(3 / (1 + math.Exp(-1)))
	if difference := math.Abs(float64(results[moe].Data[0] - wantMoE)); difference > 1e-6 {
		t.Fatalf("DeepSeek 4 fixed MoE = %v, want %v", results[moe].Data[0], wantMoE)
	}
}

func TestExecuteAttentionWithT5RelativeBias(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 2))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(1, 4))
	output := builder.AttentionWithRelativeBias(query, key, value, bias, 1)
	zeroShape := tensor.MustShape(1, 1, 2)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: zeroShape, Data: []float32{0, 0}},
		key:   {Shape: zeroShape, Data: []float32{0, 0}},
		value: {Shape: zeroShape, Data: []float32{1, 3}},
		bias:  {Shape: tensor.MustShape(1, 4), Data: []float32{0, 0, 0, 10}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := results[output].Data
	if math.Abs(float64(got[0]-2.9999092)) > 1e-5 ||
		math.Abs(float64(got[1]-2)) > 1e-5 {
		t.Fatalf("relative-bias attention = %v, want approximately [2.9999092 2]", got)
	}
}

func TestT5DecoderRelativeBucketUsesAbsoluteQueryPosition(t *testing.T) {
	if got := relativePositionBucket(4, 0, 32, false); got != 4 {
		t.Fatalf("past decoder bucket = %d, want 4", got)
	}
	if got := relativePositionBucket(4, 5, 32, false); got != 0 {
		t.Fatalf("future decoder bucket = %d, want 0", got)
	}
}

func TestExecuteReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4))
	output := builder.ReLU(input)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		input: {Shape: input.Shape, Data: []float32{-2, 0, 1.5, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 0, 1.5, 3}
	if !reflect.DeepEqual(results[output].Data, want) {
		t.Fatalf("ReLU = %v, want %v", results[output].Data, want)
	}
}

func TestExecuteSameConv1DAndGroupNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 3))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(3, 2, 1))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(1, 1))
	convolution := builder.Conv1DSame(input, weight, bias, false)
	normInput := builder.Input("norm_input", dtype.F32, tensor.MustShape(4, 2))
	normWeight := builder.Input("norm_weight", dtype.F32, tensor.MustShape(1, 4))
	normBias := builder.Input("norm_bias", dtype.F32, tensor.MustShape(1, 4))
	normalized := builder.GroupNorm(normInput, normWeight, normBias, 2, 1e-5)
	feeds := map[*tensor.Tensor]Value{
		input:      {Shape: input.Shape, Data: []float32{1, 10, 2, 20, 3, 30}},
		weight:     {Shape: weight.Shape, Data: []float32{1, 1, 1, .1, .1, .1}},
		bias:       {Shape: bias.Shape, Data: []float32{1}},
		normInput:  {Shape: normInput.Shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		normWeight: {Shape: normWeight.Shape, Data: []float32{1, 1, 1, 1}},
		normBias:   {Shape: normBias.Shape, Data: make([]float32, 4)},
	}
	results, err := Execute([]*tensor.Tensor{convolution, normalized}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results[convolution].Data, []float32{7, 13, 11}) {
		t.Fatalf("same Conv1D = %v, want [7 13 11]", results[convolution].Data)
	}
	want := []float64{-1.2126768, -0.7276061, -1.2126768, -0.7276061, 0.7276061, 1.2126768, 0.7276061, 1.2126768}
	for index, value := range results[normalized].Data {
		if math.Abs(float64(value)-want[index]) > 1e-5 {
			t.Fatalf("group norm[%d] = %v, want %v", index, value, want[index])
		}
	}
}

func TestExecuteGroupedMulMat(t *testing.T) {
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(2, 2, 2))
	right := builder.Input("right", dtype.F32, tensor.MustShape(2, 2, 2))
	output := builder.GroupedMulMat(left, right)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		left:  {Shape: left.Shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		right: {Shape: right.Shape, Data: []float32{1, 1, 2, 1, 1, 2, 2, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{3, 7, 16, 22, 5, 11, 22, 30}
	if !reflect.DeepEqual(results[output].Data, want) {
		t.Fatalf("grouped matmul = %v, want %v", results[output].Data, want)
	}
}

func TestExecuteLoRAProjectionAndEmbedding(t *testing.T) {
	builder := tensor.NewBuilder()
	builder.SetLoRA(map[string][]tensor.LoRADefinition{
		"projection": {{
			AName: "adapter.projection.lora_a", BName: "adapter.projection.lora_b",
			AShape: tensor.MustShape(2, 1), BShape: tensor.MustShape(1, 2),
			AData: []float32{2, 3}, BData: []float32{4, 5}, Scale: 0.5,
		}},
		"token_embd.weight": {{
			AName: "adapter.token.lora_a", BName: "adapter.token.lora_b",
			AShape: tensor.MustShape(1, 3), BShape: tensor.MustShape(1, 2),
			AData: []float32{7, 11, 13}, BData: []float32{2, 3}, Scale: 0.25, Embedding: true,
		}},
	})
	projection := builder.Input("projection", dtype.F32, tensor.MustShape(2, 2))
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	table := builder.Input("token_embd.weight", dtype.F32, tensor.MustShape(2, 3))
	projected := builder.MulMat(projection, input)
	embedded := builder.GetRows(table, []uint32{1})
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	results, err := Execute(
		[]*tensor.Tensor{projected, embedded},
		map[*tensor.Tensor]Value{
			projection: {Shape: projection.Shape, Data: []float32{1, 0, 0, 1}},
			input:      {Shape: input.Shape, Data: []float32{1, 2}},
			table:      {Shape: table.Shape, Data: []float32{1, 2, 3, 4, 5, 6}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := results[projected].Data, []float32{17, 22}; !reflect.DeepEqual(got, want) {
		t.Fatalf("projection = %v, want %v", got, want)
	}
	if got, want := results[embedded].Data, []float32{8.5, 12.25}; !reflect.DeepEqual(got, want) {
		t.Fatalf("embedding = %v, want %v", got, want)
	}
}

func TestExecuteGroupedLoRAProjection(t *testing.T) {
	builder := tensor.NewBuilder()
	builder.SetLoRA(map[string][]tensor.LoRADefinition{
		"experts": {{
			AName: "adapter.experts.lora_a", BName: "adapter.experts.lora_b",
			AShape: tensor.MustShape(2, 1, 2), BShape: tensor.MustShape(1, 1, 2),
			AData: []float32{1, 1, 2, 0}, BData: []float32{3, 4}, Scale: 0.5,
		}},
	})
	experts := builder.Input("experts", dtype.F32, tensor.MustShape(2, 1, 2))
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 2, 1))
	output := builder.GroupedMulMat(experts, input)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		experts: {Shape: experts.Shape, Data: []float32{1, 0, 0, 1}},
		input:   {Shape: input.Shape, Data: []float32{2, 3, 4, 5}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := results[output].Data, []float32{9.5, 21}; !reflect.DeepEqual(got, want) {
		t.Fatalf("grouped output = %v, want %v", got, want)
	}
}

func TestExecuteLoRAFusedMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	builder.SetLoRA(map[string][]tensor.LoRADefinition{
		"up": {
			{
				AName: "adapter.0.up.lora_a", BName: "adapter.0.up.lora_b",
				AShape: tensor.MustShape(2, 1, 1), BShape: tensor.MustShape(1, 1, 1),
				AData: []float32{1, 1}, BData: []float32{2}, Scale: 1,
			},
			{
				AName: "adapter.1.up.lora_a", BName: "adapter.1.up.lora_b",
				AShape: tensor.MustShape(2, 1, 1), BShape: tensor.MustShape(1, 1, 1),
				AData: []float32{1, 0}, BData: []float32{1}, Scale: 1,
			},
		},
	})
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 1))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 1))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 1))
	output := builder.MoE(input, router, gate, up, down, 1, false, 1)
	feeds := map[*tensor.Tensor]Value{
		input:  {Shape: input.Shape, Data: []float32{1, 2}},
		router: {Shape: router.Shape, Data: []float32{0, 0}},
		gate:   {Shape: gate.Shape, Data: []float32{1, 0}},
		up:     {Shape: up.Shape, Data: []float32{0, 0}},
		down:   {Shape: down.Shape, Data: []float32{1, 2}},
	}
	results, err := Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	activated := float32(7 / (1 + math.Exp(-1)))
	want := []float32{activated, 2 * activated}
	for index, got := range results[output].Data {
		if math.Abs(float64(got-want[index])) > 1e-5 {
			t.Fatalf("MoE output = %v, want %v", results[output].Data, want)
		}
	}
	merged := false
	for _, node := range builder.Nodes() {
		merged = merged || node.Op == tensor.OpLoRAMerge
	}
	if !merged {
		t.Fatal("fused MoE omitted LoRA merge")
	}
}

func TestExecuteIndexerScore(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 2, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(2, 1, 3))
	weights := builder.Input("weights", dtype.F32, tensor.MustShape(2, 2))
	output := builder.IndexerScore(query, key, weights, 0.5, 1)
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query:   {Shape: query.Shape, Data: []float32{1, 0, 0, 1, 1, 1, -1, 1}},
		key:     {Shape: key.Shape, Data: []float32{1, 0, 0, 1, 1, 1}},
		weights: {Shape: weights.Shape, Data: []float32{2, 1, 1, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 0.5, float32(math.Inf(-1)), 0.5, 2, 1}
	if !reflect.DeepEqual(results[output].Data, want) {
		t.Fatalf("indexer score = %v, want %v", results[output].Data, want)
	}
}

func TestExecuteWindowPartition2DRoundTrip(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1, 3, 2))
	partitioned := builder.WindowPartition2D(input, 2)
	output := builder.WindowUnpartition2D(partitioned, 3, 2)
	value := Value{Shape: input.Shape, Data: []float32{1, 2, 3, 4, 5, 6}}
	results, err := Execute([]*tensor.Tensor{partitioned, output}, map[*tensor.Tensor]Value{input: value})
	if err != nil {
		t.Fatal(err)
	}
	wantPartition := []float32{1, 2, 4, 5, 3, 0, 6, 0}
	if !reflect.DeepEqual(results[partitioned].Data, wantPartition) {
		t.Fatalf("partition = %v, want %v", results[partitioned].Data, wantPartition)
	}
	if !reflect.DeepEqual(results[output].Data, value.Data) {
		t.Fatalf("round trip = %v, want %v", results[output].Data, value.Data)
	}
}

func TestExecuteSAMAttentionUniform(t *testing.T) {
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 4, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 4, 2))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 4, 2))
	relativeW := builder.Input("relative_w", dtype.F32, tensor.MustShape(1, 3))
	relativeH := builder.Input("relative_h", dtype.F32, tensor.MustShape(1, 3))
	output := builder.SAMAttention(query, key, value, relativeW, relativeH, 1, 1, 2)
	zero := []float32{0, 0, 0, 0, 0, 0, 0, 0}
	results, err := Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]Value{
		query: {Shape: query.Shape, Data: zero}, key: {Shape: key.Shape, Data: zero},
		value:     {Shape: value.Shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		relativeW: {Shape: relativeW.Shape, Data: []float32{0, 0, 0}},
		relativeH: {Shape: relativeH.Shape, Data: []float32{0, 0, 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2.5, 2.5, 2.5, 2.5, 6.5, 6.5, 6.5, 6.5}
	if !reflect.DeepEqual(results[output].Data, want) {
		t.Fatalf("SAM attention = %v, want %v", results[output].Data, want)
	}
}
