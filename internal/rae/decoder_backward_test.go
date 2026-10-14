package rae

import (
	"math"
	"math/rand/v2"
	"testing"
)

// smallDecoder is a decoder of two blocks with random weights, small enough
// to differentiate by finite differences: four latent channels over a two by
// two grid, eight wide with two heads, patches of two by two pixels.
func smallDecoder(seed uint64) *vitDecoder {
	rng := rand.New(rand.NewPCG(seed, seed))
	random := func(count int, spread float32) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = spread * float32(rng.NormFloat64())
		}
		return values
	}
	near := func(count int) []float32 {
		values := random(count, 0.1)
		for index := range values {
			values[index]++
		}
		return values
	}
	d := &vitDecoder{
		latent: 4, hidden: 8, heads: 2, intermediate: 16, grid: 2, patch: 2, epsilon: 1e-6,
		pixelMean: []float32{0.4, 0.5, 0.6}, pixelStd: []float32{0.2, 0.25, 0.3},
	}
	d.embedWeight, d.embedBias = random(d.hidden*d.latent, 0.5), random(d.hidden, 0.1)
	d.class, d.position = random(d.hidden, 0.5), random((d.grid*d.grid+1)*d.hidden, 0.2)
	d.normWeight, d.normBias = near(d.hidden), random(d.hidden, 0.1)
	d.predWeight, d.predBias = random(d.patch*d.patch*3*d.hidden, 0.5), random(d.patch*d.patch*3, 0.1)
	square := d.hidden * d.hidden
	for range 2 {
		d.layers = append(d.layers, layer{
			norm1Weight: near(d.hidden), norm1Bias: random(d.hidden, 0.1),
			query: random(square, 0.5), queryBias: random(d.hidden, 0.1), key: random(square, 0.5), keyBias: random(d.hidden, 0.1),
			value: random(square, 0.5), valueBias: random(d.hidden, 0.1), output: random(square, 0.5), outputBias: random(d.hidden, 0.1),
			norm2Weight: near(d.hidden), norm2Bias: random(d.hidden, 0.1),
			up: random(d.intermediate*d.hidden, 0.5), upBias: random(d.intermediate, 0.1),
			down: random(d.hidden*d.intermediate, 0.5), downBias: random(d.hidden, 0.1),
		})
	}
	return d
}

// TestDecoderBackwardMatchesFiniteDifferences holds the decoder's backward
// pass to its forward pass: for a loss that weighs every decoded pixel, the
// gradient it gives for the tokens, and for each kind of weight, agrees with
// the change in the loss when that input is nudged along a random direction;
// and the traced forward pass decodes what the plain one does.
func TestDecoderBackwardMatchesFiniteDifferences(t *testing.T) {
	t.Parallel()
	d := smallDecoder(7)
	rng := rand.New(rand.NewPCG(11, 11))
	direction := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(rng.NormFloat64())
		}
		return values
	}
	rows := direction(d.grid * d.grid * d.latent)
	side := d.grid * d.patch
	pixelWeights := direction(3 * side * side)
	loss := func() float64 {
		pixels, err := d.decodeTokens(rows)
		if err != nil {
			t.Fatal(err)
		}
		var total float64
		for index, value := range pixels {
			total += float64(value) * float64(pixelWeights[index])
		}
		return total
	}
	plain, err := d.decodeTokens(rows)
	if err != nil {
		t.Fatal(err)
	}
	traced, trace, err := d.decodeTraced(rows)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range plain {
		if traced[index] != value {
			t.Fatalf("the traced forward pass decodes pixel %d as %v, the plain one as %v", index, traced[index], value)
		}
	}
	gradients := d.newGradients()
	dRows, err := d.backward(trace, pixelWeights, gradients)
	if err != nil {
		t.Fatal(err)
	}
	// Without kept gradients the token gradient is the same.
	bare, err := d.backward(trace, pixelWeights, nil)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range dRows {
		if bare[index] != value {
			t.Fatalf("the token gradient differs when no weight gradient is kept: %v and %v", bare[index], value)
		}
	}
	const nudge = 2e-3
	for name, pair := range map[string][2][]float32{
		"the tokens":                     {rows, dRows},
		"the embedding":                  {d.embedWeight, gradients.embedWeight},
		"the embedding's offset":         {d.embedBias, gradients.embedBias},
		"the class token":                {d.class, gradients.class},
		"the final norm's scale":         {d.normWeight, gradients.normWeight},
		"the final norm's shift":         {d.normBias, gradients.normBias},
		"the prediction":                 {d.predWeight, gradients.predWeight},
		"the prediction's offset":        {d.predBias, gradients.predBias},
		"the first block's query":        {d.layers[0].query, gradients.layers[0].query},
		"the first block's key offset":   {d.layers[0].keyBias, gradients.layers[0].keyBias},
		"the first block's value":        {d.layers[0].value, gradients.layers[0].value},
		"the first block's first norm":   {d.layers[0].norm1Weight, gradients.layers[0].norm1Weight},
		"the second block's output":      {d.layers[1].output, gradients.layers[1].output},
		"the second block's second norm": {d.layers[1].norm2Bias, gradients.layers[1].norm2Bias},
		"the second block's up":          {d.layers[1].up, gradients.layers[1].up},
		"the first block's down":         {d.layers[0].down, gradients.layers[0].down},
		"the first block's down offset":  {d.layers[0].downBias, gradients.layers[0].downBias},
	} {
		values, gradient := pair[0], pair[1]
		along := direction(len(values))
		var predicted float64
		for index, value := range gradient {
			predicted += float64(value) * float64(along[index])
		}
		original := append([]float32(nil), values...)
		for index := range values {
			values[index] = original[index] + nudge*along[index]
		}
		raised := loss()
		for index := range values {
			values[index] = original[index] - nudge*along[index]
		}
		lowered := loss()
		copy(values, original)
		measured := (raised - lowered) / (2 * nudge)
		if math.Abs(measured-predicted) > 2e-2*max(math.Abs(measured), math.Abs(predicted), 1) {
			t.Errorf("along %s the loss changes by %.5f a unit and the backward pass says %.5f", name, measured, predicted)
		}
	}
	if _, err := d.backward(trace, pixelWeights[:3], nil); err == nil {
		t.Fatal("a pixel gradient of another size was carried back")
	}
}

// newGradients allocates zero gradients for every weight of the decoder.
func (d *vitDecoder) newGradients() *decoderGradients {
	zero := func(like []float32) []float32 { return make([]float32, len(like)) }
	gradients := &decoderGradients{
		embedWeight: zero(d.embedWeight), embedBias: zero(d.embedBias), class: zero(d.class),
		normWeight: zero(d.normWeight), normBias: zero(d.normBias), predWeight: zero(d.predWeight), predBias: zero(d.predBias),
	}
	for _, l := range d.layers {
		var g layer
		for index, target := range g.tensors() {
			*target = zero(*l.tensors()[index])
		}
		gradients.layers = append(gradients.layers, g)
	}
	return gradients
}
