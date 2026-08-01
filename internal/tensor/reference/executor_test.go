package reference

import (
	"math"
	"reflect"
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
		[4]int32{1, 1, 1, 1},
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
	for pair, position := range []float64{1, 2, 3, 4} {
		theta := position * 0.25 * math.Pow(10000, -2*float64(pair)/8)
		cosine, sine := float32(math.Cos(theta)), float32(math.Sin(theta))
		want[pair] = inputValue.Data[pair]*cosine - inputValue.Data[pair+4]*sine
		want[pair+4] = inputValue.Data[pair]*sine + inputValue.Data[pair+4]*cosine
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
