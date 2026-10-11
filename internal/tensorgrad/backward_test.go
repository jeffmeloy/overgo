package tensorgrad

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// feed is a tensor's values, kept in float64 for the reference and narrowed
// for the program.
type feed struct {
	node   *tensor.Tensor
	values []float64
}

// narrowed is the feed's values as the program reads them.
func (f feed) narrowed() []float32 {
	narrow := make([]float32, len(f.values))
	for index, value := range f.values {
		narrow[index] = float32(value)
	}
	return narrow
}

// directional holds a backward graph to a float64 reference of its forward
// graph: along a direction drawn over every wanted tensor at once, the
// gradients the program gives, summed against the direction, are the central
// difference of the reference's scalar, to within tolerance of it.
func directional(t *testing.T, name string, random *rand.Rand, gradients []*tensor.Tensor, wanted, others []feed, scalar func() float64, tolerance float64) {
	t.Helper()
	program, err := reference.Compile(gradients...)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	inputs := program.NewInputs()
	for _, fed := range append(append([]feed(nil), wanted...), others...) {
		value, err := reference.NewValue(fed.node.Shape, fed.narrowed())
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := inputs.Set(fed.node, value); err != nil {
			// A tensor the gradients do not read is no input of their program.
			if !strings.Contains(err.Error(), "not") {
				t.Fatalf("%s: %v", name, err)
			}
		}
	}
	values, err := program.Execute(inputs, program.NewWorkspace())
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	const step = 1e-5
	along, most := 0.0, 0.0
	directions := make([][]float64, len(wanted))
	for index, fed := range wanted {
		directions[index] = make([]float64, len(fed.values))
		for at := range directions[index] {
			directions[index][at] = random.NormFloat64()
			along += float64(values[gradients[index]].Data[at]) * directions[index][at]
		}
	}
	shift := func(by float64) float64 {
		for index, fed := range wanted {
			for at := range fed.values {
				fed.values[at] += by * directions[index][at]
			}
		}
		return scalar()
	}
	up := shift(step)
	down := shift(-2 * step)
	shift(step)
	slope := (up - down) / (2 * step)
	most = max(math.Abs(slope), 1)
	t.Logf("%s: the backward graph gives %.9g along a direction, and the reference's slope is %.9g", name, along, slope)
	if math.IsNaN(along) || math.Abs(along-slope) > tolerance*most {
		t.Errorf("%s: the backward graph gives %v along a direction where the reference's slope is %v", name, along, slope)
	}
}

// TestBackwardGraphAgreesWithDirectionalDerivatives holds the backward graph
// built from the operations' rules to a float64 reference written here.
//
// A feed-forward half of a dense block, a weighted RMS normalization, two
// projections gated by SiLU, a projection back and the residual sum, is
// differentiated by its input and its three weights at once: along a random
// direction over all four, the program's gradients agree with the central
// difference of the reference to a part in ten thousand, which is the
// program's 32-bit arithmetic and not the rules'.
//
// So do a softmax over a projection, the head a loss's gradient is taken
// through; causal attention, four query heads over two heads of keys and
// values and over four, whose backward graph names the one causal mask it
// reads, the query heads of a key head taken together; and a
// whole dense block, a weighted normalization, attention over rotated
// queries and keys with its four projections, the residual and the
// feed-forward half with its own, by its input and all seven weights at
// once. A rotary encoding is linear, so its gradient is held by that alone,
// for both layouts: what it sends back against a direction is what it was
// given against the direction's encoding, which is no encoding of nothing.
//
// What cannot be differentiated is refused by name: an operation with no
// rule on the path to a wanted tensor, and a wanted tensor the output does
// not depend on. A branch no wanted tensor enters is not differentiated, so
// an operation without a rule there is no refusal.
func TestBackwardGraphAgreesWithDirectionalDerivatives(t *testing.T) {
	random := rand.New(rand.NewPCG(7, 11))
	const width, hidden, rows, epsilon = 6, 10, 4, 1e-5
	shape := func(dimensions ...uint64) tensor.Shape {
		made, err := tensor.NewShape(dimensions...)
		if err != nil {
			t.Fatal(err)
		}
		return made
	}
	b := tensor.NewBuilder()
	drawn := func(name string, dimensions ...uint64) feed {
		node := b.Input(name, dtype.F32, shape(dimensions...))
		elements, err := node.Shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		values := make([]float64, elements)
		for index := range values {
			values[index] = random.NormFloat64()
		}
		return feed{node: node, values: values}
	}
	x, gain := drawn("x", width, rows), drawn("gain", width)
	gate, up, down := drawn("gate", width, hidden), drawn("up", width, hidden), drawn("down", hidden, width)
	seed := drawn("seed", width, rows)
	normed := b.Multiply(b.RMSNorm(x.node, epsilon), gain.node)
	output := b.Add(x.node, b.MulMat(down.node, b.Multiply(b.SiLU(b.MulMat(gate.node, normed)), b.MulMat(up.node, normed))))
	gradients, _, err := Backward(b, output, seed.node, x.node, gate.node, up.node, down.node)
	if err != nil {
		t.Fatal(err)
	}
	scalar := func() float64 {
		total := 0.0
		for row := range rows {
			at := x.values[row*width : (row+1)*width]
			square := 0.0
			for _, value := range at {
				square += value * value
			}
			rate := 1 / math.Sqrt(square/width+epsilon)
			gated := make([]float64, hidden)
			for unit := range gated {
				left, right := 0.0, 0.0
				for index, value := range at {
					left += gate.values[unit*width+index] * value * rate * gain.values[index]
					right += up.values[unit*width+index] * value * rate * gain.values[index]
				}
				gated[unit] = left / (1 + math.Exp(-left)) * right
			}
			for index, value := range at {
				back := 0.0
				for unit, activation := range gated {
					back += down.values[index*hidden+unit] * activation
				}
				total += seed.values[row*width+index] * (value + back)
			}
		}
		return total
	}
	directional(t, "the feed-forward half", random, gradients, []feed{x, gate, up, down}, []feed{gain, seed}, scalar, 1e-4)

	// A softmax over a projection: the head a loss's gradient is taken through.
	classes := drawn("classes", width, hidden)
	weighed := drawn("weighed", hidden, rows)
	shares, _, err := Backward(b, b.Softmax(b.MulMat(classes.node, x.node)), weighed.node, classes.node, x.node)
	if err != nil {
		t.Fatal(err)
	}
	directional(t, "a softmax over a projection", random, shares, []feed{classes, x}, []feed{weighed}, func() float64 {
		total := 0.0
		for row := range rows {
			scores, most, sum := make([]float64, hidden), math.Inf(-1), 0.0
			for unit := range scores {
				for index := range width {
					scores[unit] += classes.values[unit*width+index] * x.values[row*width+index]
				}
				most = max(most, scores[unit])
			}
			for _, score := range scores {
				sum += math.Exp(score - most)
			}
			for unit, score := range scores {
				total += weighed.values[row*hidden+unit] * math.Exp(score-most) / sum
			}
		}
		return total
	}, 1e-4)

	// A rotary encoding is linear, so its gradient is held by that alone: the
	// gradient it sends back, against any direction, is the gradient it was
	// given against the encoding of that direction.
	const heads, pairs = 2, 4
	turned, against, direction := drawn("turned", pairs*2, heads, rows), drawn("against", pairs*2, heads, rows), drawn("direction", pairs*2, heads, rows)
	for _, layout := range []tensor.RoPELayout{tensor.RoPELayoutNeoX, tensor.RoPELayoutNormal} {
		options := tensor.RoPEOptions{Layout: layout, Positions: []uint32{0, 1, 2, 5}, RotaryDimensions: pairs * 2, FrequencyBase: 10000, FrequencyScale: 1}
		back, _, err := Backward(b, b.RoPEWithOptions(turned.node, options), against.node, turned.node)
		if err != nil {
			t.Fatal(err)
		}
		encoded := b.RoPEWithOptions(direction.node, options)
		program, err := reference.Compile(back[0], encoded)
		if err != nil {
			t.Fatal(err)
		}
		inputs := program.NewInputs()
		for _, fed := range []feed{against, direction} {
			value, err := reference.NewValue(fed.node.Shape, fed.narrowed())
			if err != nil {
				t.Fatal(err)
			}
			if err := inputs.Set(fed.node, value); err != nil {
				t.Fatal(err)
			}
		}
		values, err := program.Execute(inputs, program.NewWorkspace())
		if err != nil {
			t.Fatal(err)
		}
		sent, given := 0.0, 0.0
		for index, value := range values[back[0]].Data {
			sent += float64(value) * direction.values[index]
		}
		for index, value := range values[encoded].Data {
			given += float64(value) * against.values[index]
		}
		if slices.Equal(values[encoded].Data, direction.narrowed()) {
			t.Fatalf("rotary layout %d encodes a direction as itself: nothing was turned, so nothing is held", layout)
		}
		if math.Abs(sent-given) > 1e-4*max(math.Abs(given), 1) || sent == 0 {
			t.Errorf("rotary layout %d sends back %v against a direction, and was given %v against the direction's encoding", layout, sent, given)
		}
	}

	// Causal attention, grouped and not: four query heads over two heads of
	// keys and values, and over four, differentiated by all three at once. The
	// mask its backward graph reads is a constant it names, fed with what it
	// says.
	const queryHeads, keyWidth, valueWidth, scale = 4, 6, 5, 0.4
	for _, sharedHeads := range []int{2, queryHeads} {
		named := func(name string) string { return fmt.Sprintf("%s over %d", name, sharedHeads) }
		queries, onto := drawn(named("queries"), keyWidth, queryHeads, rows), drawn(named("onto"), valueWidth, queryHeads, rows)
		keys, held := drawn(named("keys"), keyWidth, uint64(sharedHeads), rows), drawn(named("values"), valueWidth, uint64(sharedHeads), rows)
		attention := b.AttentionWithOptions(queries.node, keys.node, held.node, tensor.AttentionOptions{Scale: scale, Causal: true})
		attended, constants, err := Backward(b, attention, onto.node, queries.node, keys.node, held.node)
		if err != nil || len(constants) != 1 {
			t.Fatalf("the attention's backward graph names %d constants (%v), want its one causal mask", len(constants), err)
		}
		fixed := []feed{onto}
		for _, constant := range constants {
			values := make([]float64, len(constant.Data))
			for index, value := range constant.Data {
				values[index] = float64(value)
			}
			fixed = append(fixed, feed{node: constant.Tensor, values: values})
		}
		directional(t, named("causal attention, four query heads"), random, attended, []feed{queries, keys, held}, fixed, func() float64 {
			total := 0.0
			for query := range rows {
				for head := range queryHeads {
					group := head / (queryHeads / sharedHeads)
					scores, most, sum := make([]float64, query+1), math.Inf(-1), 0.0
					for key := range scores {
						for channel := range keyWidth {
							scores[key] += queries.values[(query*queryHeads+head)*keyWidth+channel] * keys.values[(key*sharedHeads+group)*keyWidth+channel]
						}
						scores[key] *= scale
						most = max(most, scores[key])
					}
					for _, score := range scores {
						sum += math.Exp(score - most)
					}
					for key, score := range scores {
						share := math.Exp(score-most) / sum
						for channel := range valueWidth {
							total += onto.values[(query*queryHeads+head)*valueWidth+channel] * share * held.values[(key*sharedHeads+group)*valueWidth+channel]
						}
					}
				}
			}
			return total
		}, 1e-4)
	}

	// A whole dense block: a weighted normalization, grouped causal attention
	// over rotated queries and keys with its four projections, the residual,
	// and the feed-forward half with its own. Its input and all seven weights
	// are differentiated at once.
	const blockWidth, headWidth, blockHeads, blockGroups = 8, 4, 2, 1
	base, sharp := 10000.0, 1/math.Sqrt(headWidth)
	positions := []uint32{0, 1, 2, 3}
	state, first, second := drawn("state", blockWidth, rows), drawn("first", blockWidth), drawn("second", blockWidth)
	toQuery, toKey, toValue := drawn("to_query", blockWidth, blockWidth), drawn("to_key", blockWidth, headWidth*blockGroups), drawn("to_value", blockWidth, headWidth*blockGroups)
	toOut, toGate, toUp, toDown := drawn("to_out", blockWidth, blockWidth), drawn("to_gate", blockWidth, hidden), drawn("to_up", blockWidth, hidden), drawn("to_down", hidden, blockWidth)
	target := drawn("target", blockWidth, rows)
	rotary := tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: headWidth, FrequencyBase: float32(base), FrequencyScale: 1}
	prepared := b.Multiply(b.RMSNorm(state.node, epsilon), first.node)
	mixed := b.AttentionWithOptions(
		b.RoPEWithOptions(b.Reshape(b.MulMat(toQuery.node, prepared), headWidth, blockHeads, rows), rotary),
		b.RoPEWithOptions(b.Reshape(b.MulMat(toKey.node, prepared), headWidth, blockGroups, rows), rotary),
		b.Reshape(b.MulMat(toValue.node, prepared), headWidth, blockGroups, rows),
		tensor.AttentionOptions{Scale: float32(sharp), Causal: true})
	after := b.Add(state.node, b.MulMat(toOut.node, b.Reshape(mixed, blockWidth, rows)))
	settled := b.Multiply(b.RMSNorm(after, epsilon), second.node)
	block := b.Add(after, b.MulMat(toDown.node, b.Multiply(b.SiLU(b.MulMat(toGate.node, settled)), b.MulMat(toUp.node, settled))))
	weights := []feed{state, toQuery, toKey, toValue, toOut, toGate, toUp, toDown}
	differentiated := make([]*tensor.Tensor, len(weights))
	for index, weight := range weights {
		differentiated[index] = weight.node
	}
	whole, masks, err := Backward(b, block, target.node, differentiated...)
	if err != nil {
		t.Fatal(err)
	}
	unmoved := []feed{first, second, target}
	for _, constant := range masks {
		values := make([]float64, len(constant.Data))
		for index, value := range constant.Data {
			values[index] = float64(value)
		}
		unmoved = append(unmoved, feed{node: constant.Tensor, values: values})
	}
	// project is rows of in values through a weight of out rows.
	project := func(weight, from []float64, in, out int) []float64 {
		to := make([]float64, len(from)/in*out)
		for row := range len(from) / in {
			for unit := range out {
				for index := range in {
					to[row*out+unit] += weight[unit*in+index] * from[row*in+index]
				}
			}
		}
		return to
	}
	normal := func(from, gain []float64) []float64 {
		to := make([]float64, len(from))
		for row := range len(from) / blockWidth {
			square := 0.0
			for _, value := range from[row*blockWidth : (row+1)*blockWidth] {
				square += value * value
			}
			for index := range blockWidth {
				to[row*blockWidth+index] = from[row*blockWidth+index] / math.Sqrt(square/blockWidth+epsilon) * gain[index]
			}
		}
		return to
	}
	turn := func(from []float64, heads int) {
		for row, position := range positions {
			for head := range heads {
				at := from[(row*heads+head)*headWidth:]
				for pair := range headWidth / 2 {
					angle := float64(position) * math.Pow(base, -2*float64(pair)/headWidth)
					low, high := at[pair], at[pair+headWidth/2]
					at[pair], at[pair+headWidth/2] = low*math.Cos(angle)-high*math.Sin(angle), low*math.Sin(angle)+high*math.Cos(angle)
				}
			}
		}
	}
	directional(t, "a whole dense block", random, whole, weights, unmoved, func() float64 {
		ready := normal(state.values, first.values)
		asked, offered, carried := project(toQuery.values, ready, blockWidth, blockWidth), project(toKey.values, ready, blockWidth, headWidth*blockGroups), project(toValue.values, ready, blockWidth, headWidth*blockGroups)
		turn(asked, blockHeads)
		turn(offered, blockGroups)
		gathered := make([]float64, rows*blockWidth)
		for query := range rows {
			for head := range blockHeads {
				group := head / (blockHeads / blockGroups)
				scores, most, sum := make([]float64, query+1), math.Inf(-1), 0.0
				for key := range scores {
					for channel := range headWidth {
						scores[key] += asked[(query*blockHeads+head)*headWidth+channel] * offered[(key*blockGroups+group)*headWidth+channel]
					}
					scores[key] *= sharp
					most = max(most, scores[key])
				}
				for _, score := range scores {
					sum += math.Exp(score - most)
				}
				for key, score := range scores {
					for channel := range headWidth {
						gathered[(query*blockHeads+head)*headWidth+channel] += math.Exp(score-most) / sum * carried[(key*blockGroups+group)*headWidth+channel]
					}
				}
			}
		}
		passed := project(toOut.values, gathered, blockWidth, blockWidth)
		for index := range passed {
			passed[index] += state.values[index]
		}
		calm := normal(passed, second.values)
		opened, raised := project(toGate.values, calm, blockWidth, hidden), project(toUp.values, calm, blockWidth, hidden)
		for index, value := range opened {
			opened[index] = value / (1 + math.Exp(-value)) * raised[index]
		}
		total := 0.0
		for index, value := range project(toDown.values, opened, hidden, blockWidth) {
			total += target.values[index] * (passed[index] + value)
		}
		return total
	}, 1e-4)

	// What is refused, and what is not differentiated.
	other := tensor.NewBuilder()
	input, weight := other.Input("input", dtype.F32, shape(width, rows)), other.Input("weight", dtype.F32, shape(width, hidden))
	gradient := other.Input("gradient", dtype.F32, shape(hidden, rows))
	if _, _, err := Backward(other, other.Clamp(other.MulMat(weight, input), -1, 1), gradient, weight); err == nil || !strings.Contains(err.Error(), "no derivative rule") {
		t.Errorf("a path through an operation with no rule = %v, want it refused by name", err)
	}
	if _, _, err := Backward(other, other.MulMat(weight, input), gradient, other.Input("apart", dtype.F32, shape(width))); err == nil || !strings.Contains(err.Error(), "does not depend") {
		t.Errorf("a wanted tensor the output does not depend on = %v, want it refused", err)
	}
	if _, _, err := Backward(other, other.Add(other.MulMat(weight, input), other.Clamp(gradient, -1, 1)), gradient, weight); err != nil {
		t.Errorf("an operation with no rule on a branch no wanted tensor enters = %v, want it left alone", err)
	}
}
