//go:build windows

package executor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"maps"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// moePinnedOutputs: SHA-256 of each case's output bytes as the single-block
// routed-expert kernel at d993c2b10 produced them on the compute-capability
// 8.9 device. The routed-expert kernels keep every sum's order, so the bits
// hold; a case whose bits move has changed what an MoE model computes.
var moePinnedOutputs = map[string]string{
	"ocr":                  "678e3172935d2c8ce6833a1c58472a18f176f185459113445d068d98ed6caabf",
	"normalized-scaled":    "434dca90ea8f34491bf170da37f0d8ff993da47cafc0a3a13638b70bf526448a",
	"sigmoid-fused-biased": "45d83c6d13ad937bc48f0d610a381002eec7670beb805c83b55aeb972ff4f2d5",
	"selected-clamped":     "bfa6d790cc1f1ae664718c651a0efa865922f091e12b3274be2fb81d92186a32",
	"openai":               "4cc7a250477280ae72ea113dc262d2e0edd52bb061f86f1e63f6643e54463cbb",
	"grouped-relu-squared": "f2448f5f3fe386cce4e82486ab3e1c14ec2efbe8cde7010947fb7b7299173680",
	"gelu":                 "86582c49baf8236c0874847695b85e2ebfd5f3ba6fffc2c7d2b2e52dabd1aab8",
	"relu":                 "5fba2652baf8082bbf3d9e57ec970bcf95563e86d1a0087178f0c2a4df616a92",
	"ungated":              "39402f6fb65ba43a17ec952c92ee4f9865db08da2052ad048815533938a08336",
	"many-experts":         "f3122a649204f1c36f85e761649d1ea85fd307ffaee6686fb6b71e20d5e37255",
	"token-groups":         "d5809012afc231b750caa4bf34e3c46da106f2eccfa41e977a01755a8922966e",
	"q8_0":                 "dae2e73fd8c5dab0a7bc6a2162bc5519064ccb6ab2d90c94cf956b2e83a073a8",
	"q4_0":                 "046eb93ba939d8e2b2643dd4b67eb12c1785252895d2cd7fec6fe637a6e98539",
	"q4_K":                 "a9cc720814154ede1118c99e0d80b83f78ecfef1fce2f698ca9a922afde13232",
	"iq4_xs":               "d0875e8f5f01b2f74882ff1420976e728f13741cac8a0e5e11b3aa88ad46dcfc",
	"tq2_0":                "3dfd1b21c7b4960d90c66634699a27b12fe4bd18023429059566f803f809a120",
	"mxfp4":                "61ee218b40ffb2382b75817b2d7dbf93e570154204090744d5cc86b38638e240",
	"q1_0":                 "8424da30050035770cd2e2cf96dafabc8b7b3039f4267f4a111e9d566b9a2b33",
}

// TestExecutorMoEOutputBitsArePinned runs each case's node with its expert
// banks at every storage the case admits and holds the output to the pinned
// bits: F32 banks reproduce the single-block kernel, and BF16 banks, widened
// per element on the device, reproduce the F32 banks.
func TestExecutorMoEOutputBitsArePinned(t *testing.T) {
	cuda := newFixtureExecutor(t)
	for _, pinned := range moePinnedCases {
		t.Run(pinned.name, func(t *testing.T) {
			bankTypes := []dtype.Type{dtype.F32, dtype.BF16}
			if pinned.storage != dtype.F32 {
				bankTypes = []dtype.Type{pinned.storage}
			}
			outputs := make(map[dtype.Type]string, len(bankTypes))
			for _, bankType := range bankTypes {
				got, want := pinned.run(t, cuda, bankType)
				digest := sha256.Sum256(driver.Bytes(got))
				outputs[bankType] = hex.EncodeToString(digest[:])
				if outputs[bankType] != moePinnedOutputs[pinned.name] {
					t.Errorf("%s banks: output %s, pinned %q", bankType, outputs[bankType], moePinnedOutputs[pinned.name])
				}
				if outputs[bankType] != outputs[bankTypes[0]] {
					t.Errorf("%s banks: output %s, %s banks %s", bankType, outputs[bankType], bankTypes[0], outputs[bankTypes[0]])
				}
				if want != nil {
					compare(t, got, want, accuracyModel)
				}
			}
		})
	}
}

// moePinnedCase is one routed-expert node over pseudo-random inputs. Every
// value is BF16-representable, so an F32 bank and a BF16 bank hold the same
// numbers.
type moePinnedCase struct {
	name                               string
	hidden, routerHidden, intermediate uint64
	experts, tokens                    uint64
	divisor, topK                      uint32
	routing                            tensor.MoERouting
	activation                         tensor.MoEActivation
	normalize                          bool
	scale, clamp                       float32
	ungated, fused                     bool
	selectionBias, expertScale         bool
	selected, biases                   bool
	// storage is F32 for float banks, or the quantized type the banks carry.
	storage dtype.Type
	// reference also holds the F32-bank output to the host reference; the
	// real geometry is too wide for the host.
	reference bool
}

var moePinnedCases = []moePinnedCase{
	// Unlimited-OCR's routed layer: 64 experts, top-6, 896 wide, hidden 1280.
	{name: "ocr", hidden: 1280, routerHidden: 1280, intermediate: 896, experts: 64, tokens: 5, divisor: 1, topK: 6,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, scale: 1},
	// 600 intermediate rows end the down projection on a partial tile.
	{name: "normalized-scaled", hidden: 24, routerHidden: 24, intermediate: 600, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
		expertScale: true, reference: true},
	{name: "sigmoid-fused-biased", hidden: 24, routerHidden: 24, intermediate: 600, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSigmoid, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.5,
		fused: true, selectionBias: true, reference: true},
	{name: "selected-clamped", hidden: 24, routerHidden: 24, intermediate: 256, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSqrtSoftplus, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
		clamp: 0.15, selected: true, reference: true},
	{name: "openai", hidden: 24, routerHidden: 24, intermediate: 300, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSelectedSoftmax, activation: tensor.MoEActivationSwiGLUOAI, scale: 1,
		biases: true, reference: true},
	{name: "grouped-relu-squared", hidden: 24, routerHidden: 40, intermediate: 300, experts: 8, tokens: 4, divisor: 2, topK: 3,
		routing: tensor.MoERoutingSigmoid, activation: tensor.MoEActivationReLUSquared, normalize: true, scale: 1.25,
		ungated: true, selectionBias: true, reference: true},
	{name: "gelu", hidden: 24, routerHidden: 24, intermediate: 300, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationGELU, normalize: true, scale: 1,
		expertScale: true, reference: true},
	{name: "relu", hidden: 24, routerHidden: 24, intermediate: 300, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationReLU, scale: 0.75, reference: true},
	{name: "ungated", hidden: 24, routerHidden: 24, intermediate: 300, experts: 8, tokens: 4, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
		ungated: true, reference: true},
	// More experts than one block has threads.
	{name: "many-experts", hidden: 16, routerHidden: 16, intermediate: 8, experts: 300, tokens: 3, divisor: 1, topK: 4,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1, reference: true},
	// Seventy tokens of sixteen 8192-wide experts need more scratch than one
	// launch group holds.
	{name: "token-groups", hidden: 8, routerHidden: 8, intermediate: 8192, experts: 16, tokens: 70, divisor: 1, topK: 16,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1, reference: true},
	moePinnedQuantizedCase(dtype.Q8_0),
	moePinnedQuantizedCase(dtype.Q4_0),
	moePinnedQuantizedCase(dtype.Q4K),
	moePinnedQuantizedCase(dtype.IQ4XS),
	moePinnedQuantizedCase(dtype.TQ2_0),
	moePinnedQuantizedCase(dtype.MXFP4),
	moePinnedQuantizedCase(dtype.Q1_0),
}

// moePinnedQuantizedCase: a block-aligned node whose intermediate width spans
// two down-projection tiles, on banks of one quantized type.
func moePinnedQuantizedCase(storage dtype.Type) moePinnedCase {
	const tiledWidth = 320
	traits, _ := storage.Traits()
	return moePinnedCase{
		name: storage.String(), hidden: traits.BlockSize, routerHidden: traits.BlockSize,
		intermediate: (tiledWidth + traits.BlockSize - 1) / traits.BlockSize * traits.BlockSize,
		experts:      4, tokens: 3, divisor: 1, topK: 2,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
		storage: storage,
	}
}

// moePinnedValues: count splitmix64 draws in [-scale, scale), rounded to
// BF16, a function of the case name and the tensor's ordinal alone.
func moePinnedValues(name string, ordinal, count uint64, scale float64) []float32 {
	state := ordinal
	for _, character := range []byte(name) {
		state = state*1099511628211 + uint64(character)
	}
	values := make([]float32, count)
	for index := range values {
		state += 0x9e3779b97f4a7c15
		draw := state
		draw = (draw ^ draw>>30) * 0xbf58476d1ce4e5b9
		draw = (draw ^ draw>>27) * 0x94d049bb133111eb
		draw ^= draw >> 31
		unit := float32(draw>>40)/(1<<23) - 1
		values[index] = dtype.RoundBF16(float32(scale) * unit)
	}
	return values
}

// run executes the case's node with its expert banks stored as bankType and
// returns the device output, with the host reference's when the case asks
// for it and the banks are F32.
func (c moePinnedCase) run(t *testing.T, cuda *Executor, bankType dtype.Type) (got, want []float32) {
	t.Helper()
	builder := tensor.NewBuilder()
	host := make(map[*tensor.Tensor]reference.Value)
	device := make(map[*tensor.Tensor]driver.DevicePtr)
	withReference := c.reference && bankType == dtype.F32
	bankValues := make(map[*tensor.Tensor]reference.Value)
	ordinal := uint64(0)
	feed := func(name string, scale float64, dims ...uint64) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, tensor.MustShape(dims...))
		elements, _ := node.Shape.Elements()
		ordinal++
		host[node] = reference.Value{Shape: node.Shape, Data: moePinnedValues(c.name, ordinal, elements, scale)}
		return node
	}
	bank := func(name string, scale float64, dims ...uint64) *tensor.Tensor {
		node := builder.Input(name, bankType, tensor.MustShape(dims...))
		elements, _ := node.Shape.Elements()
		ordinal++
		values := moePinnedValues(c.name, ordinal, elements, scale)
		storage := driver.Bytes(values)
		switch {
		case bankType == dtype.BF16:
			bits := make([]uint16, len(values))
			for index, value := range values {
				bits[index] = dtype.Float32ToBF16(value)
			}
			storage = driver.Bytes(bits)
		case bankType != dtype.F32:
			quantized, err := quant.Quantize(bankType, values)
			if err != nil {
				t.Fatal(err)
			}
			storage = quantized
		}
		device[node] = copyFixtureDeviceBytes(t, cuda.worker, storage)
		if withReference {
			bankValues[node] = reference.Value{Shape: node.Shape, Data: values}
		}
		return node
	}
	bankExperts := c.experts / uint64(c.divisor)
	dotScale := func(width uint64) float64 { return 3 / math.Sqrt(float64(width)) }
	input := feed("input", 1, c.hidden, c.tokens)
	options := tensor.MoEOptions{
		TopK: c.topK, NormalizeTopKProb: c.normalize, Scale: c.scale, Routing: c.routing,
		Activation: c.activation, FusedGateUp: c.fused, ExpertIndexDivisor: c.divisor, SwiGLUClamp: c.clamp,
	}
	if c.routerHidden != c.hidden {
		options.RouterInput = feed("router_input", 1, c.routerHidden, c.tokens)
	}
	router := feed("router", dotScale(c.routerHidden), c.routerHidden, c.experts)
	upRows := c.intermediate
	switch {
	case c.fused:
		upRows *= 2
	case !c.ungated:
		options.Gate = bank("gate", dotScale(c.hidden), c.hidden, c.intermediate, bankExperts)
	}
	up := bank("up", dotScale(c.hidden), c.hidden, upRows, bankExperts)
	down := bank("down", dotScale(c.intermediate), c.intermediate, c.hidden, bankExperts)
	const biasScale = 0.25
	if c.selectionBias {
		options.SelectionBias = feed("selection_bias", biasScale, c.experts)
	}
	if c.expertScale {
		options.ExpertScale = feed("expert_scale", 1, bankExperts)
	}
	if c.biases {
		options.Biases = &tensor.MoEBiases{
			Router: feed("router_bias", biasScale, c.experts),
			Gate:   feed("gate_bias", biasScale, c.intermediate, bankExperts),
			Up:     feed("up_bias", biasScale, c.intermediate, bankExperts),
			Down:   feed("down_bias", biasScale, c.hidden, bankExperts),
		}
	}
	if c.selected {
		// Distinct experts per token: three is coprime to the expert count.
		const selectedStride = 3
		node := builder.Input("selected", dtype.F32, tensor.MustShape(uint64(c.topK), c.tokens))
		data := make([]float32, uint64(c.topK)*c.tokens)
		for index := range data {
			token, slot := uint64(index)/uint64(c.topK), uint64(index)%uint64(c.topK)
			data[index] = float32((token + slot*selectedStride) % c.experts)
		}
		host[node] = reference.Value{Shape: node.Shape, Data: data}
		options.SelectedExperts = node
	}
	output := builder.MoEWithOptions(input, router, up, down, options)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	result, err := cuda.executeWithDeviceFeeds(
		context.WithoutCancel(t.Context()), []*tensor.Tensor{output}, host, device,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !withReference {
		return result[output].Data, nil
	}
	maps.Copy(host, bankValues)
	expected, err := reference.Execute([]*tensor.Tensor{output}, host)
	if err != nil {
		t.Fatal(err)
	}
	return result[output].Data, expected[output].Data
}
