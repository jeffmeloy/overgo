//go:build windows

package executor

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"math"
	"overgo/internal/gguf"
	"overgo/internal/modeldevice"
	"sync"
	"testing"

	"overgo/internal/cuda/device"
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
			if pinned.storage != dtype.F32 || pinned.banks != [3]dtype.Type{} {
				bankTypes = []dtype.Type{pinned.storage}
			}
			outputs := make(map[dtype.Type]string, len(bankTypes))
			for _, bankType := range bankTypes {
				got, want := pinned.run(t, cuda, bankType)
				tiered := pinned
				tiered.offload = true
				indexed, _ := tiered.run(t, cuda, bankType)
				if !bytes.Equal(driver.Bytes(got), driver.Bytes(indexed)) {
					t.Fatalf("%s indexed banks changed output bits", bankType)
				}
				digest := sha256.Sum256(driver.Bytes(got))
				outputs[bankType] = hex.EncodeToString(digest[:])
				if pinned.banks == [3]dtype.Type{} && outputs[bankType] != moePinnedOutputs[pinned.name] {
					t.Errorf("%s banks: output %s, pinned %q", bankType, outputs[bankType], moePinnedOutputs[pinned.name])
				}
				if outputs[bankType] != outputs[bankTypes[0]] {
					t.Errorf("%s banks: output %s, %s banks %s", bankType, outputs[bankType], bankTypes[0], outputs[bankTypes[0]])
				}
				if want != nil {
					compare(t, got, want, accuracyModel)
				} else if pinned.banks != [3]dtype.Type{} {
					t.Fatal("mixed banks require the dequantized arithmetic reference")
				}
			}
		})
	}
}

// moePinnedCase is one routed-expert node over pseudo-random inputs. Every
// value is BF16-representable, so an F32 bank and a BF16 bank hold the same
// numbers.
type moePinnedCase struct {
	compute tensor.MulMatCompute
	offload bool
	// limit bounds the offloaded banks' residency; zero takes the automatic limit.
	limit                              uint64
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
	// banks overrides gate, up and down storage for mixed-bank reference cases.
	banks [3]dtype.Type
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
	{name: "mixed-iq", hidden: 256, routerHidden: 256, intermediate: 640, experts: 4, tokens: 3, divisor: 1, topK: 2,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
		banks: [3]dtype.Type{dtype.IQ3S, dtype.IQ4XS, dtype.IQ4NL}, reference: true},
	{name: "mixed-tail", hidden: 25, routerHidden: 25, intermediate: 17, experts: 4, tokens: 3, divisor: 1, topK: 3,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
		banks: [3]dtype.Type{dtype.BF16, dtype.F32, dtype.F32}, reference: true},
	{name: "mixed-float", hidden: 32, routerHidden: 32, intermediate: 320, experts: 4, tokens: 3, divisor: 1, topK: 2,
		routing: tensor.MoERoutingSigmoid, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
		banks: [3]dtype.Type{dtype.BF16, dtype.F32, dtype.Q8_0}, reference: true},
	{name: "mixed-fused", hidden: 256, routerHidden: 256, intermediate: 640, experts: 4, tokens: 3, divisor: 1, topK: 2,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
		banks: [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}, fused: true, reference: true},
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

// moePinnedEncodings holds each quantized bank by its storage, its case and
// ordinal, which fix its values, and its length.
//   - a case runs its banks resident and indexed, and some at several token counts
//   - the host IQ3_S quantizer takes about 17 µs a weight (this file's IQ3_S cases on the development PC)
var moePinnedEncodings sync.Map

// moePinnedQuantize returns the bank's encoding, quantizing it on first use.
func moePinnedQuantize(storage dtype.Type, name string, values []float32) ([]byte, error) {
	key := fmt.Sprintf("%s %s %d", storage, name, len(values))
	if encoded, ok := moePinnedEncodings.Load(key); ok {
		return encoded.([]byte), nil
	}
	encoded, err := quant.Quantize(storage, values)
	if err != nil {
		return nil, err
	}
	moePinnedEncodings.Store(key, encoded)
	return encoded, nil
}

// run executes the case's node with its expert banks stored as bankType and
// returns the device output, with the host reference's when the case asks
// for it and the banks are F32.
func (c moePinnedCase) run(t *testing.T, cuda *Executor, bankType dtype.Type) (got, want []float32) {
	output, host, device, bankValues := c.prepare(t, cuda, bankType)
	var result map[*tensor.Tensor]reference.Value
	var err error
	if c.compute == tensor.MulMatComputeNativeTensorCore {
		retained, runErr := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, host, device)
		if runErr != nil {
			t.Fatal(runErr)
		}
		value, copyErr := retained.CopyToHost(t.Context(), output)
		if releaseErr := retained.Release(t.Context()); releaseErr != nil {
			t.Fatal(releaseErr)
		}
		if copyErr != nil {
			t.Fatal(copyErr)
		}
		result = map[*tensor.Tensor]reference.Value{output: value}
	} else {
		result, err = cuda.executeWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, host, device)
	}
	if c.offload {
		if resetErr := cuda.SetExpertBanks(nil); resetErr != nil {
			t.Fatal(resetErr)
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(bankValues) == 0 {
		return result[output].Data, nil
	}
	maps.Copy(host, bankValues)
	for bank := range bankValues {
		bank.Type = dtype.F32
	}
	expected, err := reference.Execute([]*tensor.Tensor{output}, host)
	if err != nil {
		t.Fatal(err)
	}
	return result[output].Data, expected[output].Data
}

func (c moePinnedCase) prepare(t testing.TB, cuda *Executor, bankType dtype.Type) (*tensor.Tensor, map[*tensor.Tensor]reference.Value, map[*tensor.Tensor]driver.DevicePtr, map[*tensor.Tensor]reference.Value) {
	t.Helper()
	builder := tensor.NewBuilder()
	builder.SetMulMatCompute(c.compute)
	host := make(map[*tensor.Tensor]reference.Value)
	device := make(map[*tensor.Tensor]driver.DevicePtr)
	withReference := c.reference && bankType == dtype.F32 && !c.offload
	var fileBanks []gguf.TensorData
	bankValues := make(map[*tensor.Tensor]reference.Value)
	ordinal := uint64(0)
	feed := func(name string, scale float64, dims ...uint64) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, tensor.MustShape(dims...))
		elements, _ := node.Shape.Elements()
		ordinal++
		host[node] = reference.Value{Shape: node.Shape, Data: moePinnedValues(c.name, ordinal, elements, scale)}
		return node
	}
	bank := func(name string, index int, scale float64, dims ...uint64) *tensor.Tensor {
		storageType := bankType
		if c.banks != [3]dtype.Type{} {
			storageType = c.banks[index]
		}
		node := builder.Input(name, storageType, tensor.MustShape(dims...))
		elements, _ := node.Shape.Elements()
		ordinal++
		values := moePinnedValues(c.name, ordinal, elements, scale)
		storage := driver.Bytes(values)
		switch {
		case storageType == dtype.BF16:
			bits := make([]uint16, len(values))
			for index, value := range values {
				bits[index] = dtype.Float32ToBF16(value)
			}
			storage = driver.Bytes(bits)
		case storageType != dtype.F32:
			quantized, err := moePinnedQuantize(storageType, fmt.Sprintf("%s/%d", c.name, ordinal), values)
			if err != nil {
				t.Fatal(err)
			}
			storage = quantized
			if withReference {
				values, err = quant.Dequantize(storageType, storage, elements)
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		if c.offload {
			fileBanks = append(fileBanks, gguf.TensorData{Name: name, Shape: dims, Type: storageType, Data: bytes.NewReader(storage)})
		} else {
			device[node] = copyFixtureDeviceBytes(t, cuda.worker, storage)
		}
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
		options.Gate = bank("gate", 0, dotScale(c.hidden), c.hidden, c.intermediate, bankExperts)
	}
	up := bank("up", 1, dotScale(c.hidden), c.hidden, upRows, bankExperts)
	down := bank("down", 2, dotScale(c.intermediate), c.intermediate, c.hidden, bankExperts)
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
	if c.offload {
		var encoded bytes.Buffer
		if err := gguf.Write(&encoded, nil, fileBanks, gguf.WriteOptions{}); err != nil {
			t.Fatal(err)
		}
		file, err := gguf.Parse(bytes.NewReader(encoded.Bytes()), uint64(encoded.Len()), gguf.DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		weights, err := modeldevice.NewDeviceWeights(cuda.worker)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cuda.SetExpertBanks(nil); err != nil {
				t.Error(err)
			}
			if err := weights.Close(); err != nil {
				t.Error(err)
			}
		})
		loader := modeldevice.NewLoader(file)
		loader.Experts(weights, file.Tensors, c.limit)
		if err := loader.Load(t.Context(), nil); err != nil {
			t.Fatal(err)
		}
		if err := cuda.SetExpertBanks(weights); err != nil {
			t.Fatal(err)
		}
		for _, bank := range []*tensor.Tensor{options.Gate, up, down} {
			if bank == nil {
				continue
			}
			_, pointer, err := weights.Input(builder, bank.Name)
			if err != nil {
				t.Fatal(err)
			}
			device[bank] = pointer
		}
	}
	return output, host, device, bankValues
}

func TestExecutorMoENativeCompute(t *testing.T) {
	for _, c := range moePinnedCases {
		if !c.reference {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			for _, tokens := range []uint64{1, 64} {
				if tokens > 1 && (c.ungated || c.name == "many-experts") {
					continue
				}
				t.Run(fmt.Sprintf("tokens=%d", tokens), func(t *testing.T) {
					c.tokens = tokens
					c.compute = tensor.MulMatComputeNativeTensorCore
					var want []float32
					for _, offload := range []bool{false, true} {
						t.Run(map[bool]string{false: "resident", true: "indexed"}[offload], func(t *testing.T) {
							cuda := newFixtureExecutor(t)
							trace := &device.LaunchTrace{}
							if err := cuda.worker.Do(t.Context(), func(state *device.State) error { state.Trace = trace; return nil }); err != nil {
								t.Fatal(err)
							}
							c.offload = offload
							got, reference := c.run(t, cuda, dtype.F32)
							if !offload {
								want = reference
							}
							compare(t, got, want, accuracyModel)
							// The first expert batch follows the router logits, routing, input gathering and weight staging.
							const firstProduct = (32 + 4*8 + 3*4) + (32 + 5*8 + 5*4) + (32 + 3*8 + 5*4) + (32 + 4*8 + 7*4)
							if tokens > 1 && (len(trace.Data()) < firstProduct+8 || binary.LittleEndian.Uint64(trace.Data()[firstProduct:]) != traceTagSGEMMStrided) {
								t.Fatal("grouped expert matrix product did not execute")
							}
						})
					}
				})
			}
		})
	}
}

// TestExecutorMoEExpertsUseBlockDots runs routed-expert decode natively over
// each block storage of Qwen3.8-Flash-Next's UD-IQ4_XS experts and holds the
// output to the host reference over the dequantized banks. The hidden and
// down launches are the dot kernels with the banks' storages, indexed banks
// keep the output's bits, and an input that starts off 16-byte alignment
// stays within the reference.
func TestExecutorMoEExpertsUseBlockDots(t *testing.T) {
	// Serial: device fixtures share the one card.
	const (
		// A launch record is four header words, then eight bytes a pointer and four a scalar.
		headerBytes  = 4 * 8
		logitsRecord = headerBytes + 4*8 + 3*4
		routeRecord  = headerBytes + 5*8 + 5*4
		hiddenRecord = headerBytes + 7*8 + 13*4
		// The hidden record's storages follow its pointers and four scalars, the down record's its pointers and five.
		hiddenGateStorage = headerBytes + 7*8 + 4*4
		downStorage       = headerBytes + 6*8 + 5*4
	)
	for _, banks := range [][3]dtype.Type{
		// Flash-Next's routed layers pair IQ3_S gate and up with IQ4_NL down, and IQ4_XS with Q8_0.
		{dtype.IQ3S, dtype.IQ3S, dtype.IQ4NL},
		{dtype.IQ4XS, dtype.IQ4XS, dtype.Q8_0},
		// One storage in both projections; the host quantizes IQ3_S too slowly for a third bank.
		{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4XS},
		{dtype.IQ4NL, dtype.IQ4NL, dtype.IQ4NL},
		{dtype.Q8_0, dtype.Q8_0, dtype.Q8_0},
	} {
		name := fmt.Sprintf("%s-%s-%s", banks[0], banks[1], banks[2])
		t.Run(name, func(t *testing.T) {
			// Gate and up rows of two 256-weight blocks; three decode tokens stay below grouped matrix reuse.
			c := moePinnedCase{
				name: "block-dots-" + name, compute: tensor.MulMatComputeNativeTensorCore,
				hidden: 512, routerHidden: 512, intermediate: 256, experts: 4, tokens: 3, divisor: 1, topK: 2,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
				banks: banks, reference: true,
			}
			// The resident run sets the reference and the bits; it runs first.
			var want []float32
			var resident []byte
			for _, variant := range []struct {
				name             string
				offload, shifted bool
			}{{"resident", false, false}, {"shifted", false, true}, {"indexed", true, false}} {
				cuda := newFixtureExecutor(t)
				trace := &device.LaunchTrace{}
				if err := cuda.worker.Do(t.Context(), func(state *device.State) error { state.Trace = trace; return nil }); err != nil {
					t.Fatal(err)
				}
				c.offload = variant.offload
				output, host, pointers, bankValues := c.prepare(t, cuda, dtype.F32)
				feeds := maps.Clone(host)
				if variant.shifted {
					// One float ahead of an allocation leaves the input 4 bytes past 16-byte alignment.
					input := output.Inputs[0]
					shifted := append([]float32{0}, host[input].Data...)
					pointers[input] = copyFixtureDeviceBytes(t, cuda.worker, driver.Bytes(shifted)) + driver.DevicePtr(f32ScalarBytes)
					delete(feeds, input)
				}
				retained, err := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, feeds, pointers)
				if err != nil {
					t.Fatal(err)
				}
				value, copyErr := retained.CopyToHost(t.Context(), output)
				if err := errors.Join(copyErr, retained.Release(t.Context())); err != nil {
					t.Fatal(err)
				}
				raw := trace.Data()
				hidden, down := raw[logitsRecord+routeRecord:], raw[logitsRecord+routeRecord+hiddenRecord:]
				word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
				if word(hidden) != uint64(kernelMoeHiddenDotF32) || word(down) != uint64(kernelMoeDownDotF32) {
					t.Fatalf("%s: launches %d and %d, want the dot kernels %d and %d",
						variant.name, word(hidden), word(down), kernelMoeHiddenDotF32, kernelMoeDownDotF32)
				}
				if got := [3]dtype.Type{dtype.Type(scalar(hidden[hiddenGateStorage:])), dtype.Type(scalar(hidden[hiddenGateStorage+4:])), dtype.Type(scalar(down[downStorage:]))}; got != banks {
					t.Fatalf("%s: dot kernels received storages %v, want %v", variant.name, got, banks)
				}
				switch {
				case variant.offload:
					if !bytes.Equal(driver.Bytes(value.Data), resident) {
						t.Fatal("indexed banks changed the resident output bits")
					}
				case variant.shifted:
					// An unaligned input takes the per-weight dots, whose sums differ in their last bits.
					compare(t, value.Data, want, accuracyModel)
				default:
					resident = bytes.Clone(driver.Bytes(value.Data))
					maps.Copy(host, bankValues)
					for bank := range bankValues {
						bank.Type = dtype.F32
					}
					expected, err := reference.Execute([]*tensor.Tensor{output}, host)
					if err != nil {
						t.Fatal(err)
					}
					want = expected[output].Data
					compare(t, value.Data, want, accuracyModel)
				}
			}
		})
	}
}

// TestExecutorMoEMissedExpertsKeepResidentBits routes three decode tokens of
// Flash-Next's IQ4_XS and Q8_0 expert layer to six experts while the card
// holds only expert 0 of each bank, so no routed expert was ever on the card
// and the limit is below one layer's routed experts. The expert kernels read
// each where it is, a run waits for the stream no more often than the
// resident run, the next run finds routed experts moved to the card, and
// every run keeps the resident run's output bits.
func TestExecutorMoEMissedExpertsKeepResidentBits(t *testing.T) {
	banks := [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.Q8_0}
	c := moePinnedCase{
		name: "missed-experts", compute: tensor.MulMatComputeNativeTensorCore,
		hidden: 512, routerHidden: 512, intermediate: 256, experts: 8, tokens: 3, divisor: 1, topK: 2,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
		banks: banks, selected: true,
	}
	// The residency's tables hold a pointer an expert, and one
	// 256-byte-aligned slot of each bank holds expert 0 alone.
	const pointerBytes, slotAlignment = 8, 256
	c.limit = uint64(len(banks)) * c.experts * pointerBytes
	var smallest uint64
	for _, storage := range banks {
		expert, err := tensor.MustShape(c.hidden, c.intermediate).Bytes(storage)
		if err != nil {
			t.Fatal(err)
		}
		c.limit += (expert + slotAlignment - 1) / slotAlignment * slotAlignment
		smallest = min(cmp.Or(smallest, expert), expert)
	}
	// run executes the node once and returns its output bits and the
	// executor's stream waits and host-to-device bytes during the run.
	run := func(t *testing.T, cuda *Executor, output *tensor.Tensor, host map[*tensor.Tensor]reference.Value, pointers map[*tensor.Tensor]driver.DevicePtr) ([]byte, driver.ExecutionStats) {
		t.Helper()
		var before, after driver.ExecutionStats
		stats := func(into *driver.ExecutionStats) {
			if err := cuda.worker.Do(t.Context(), func(state *device.State) error { *into = state.Driver.ExecutionStats(); return nil }); err != nil {
				t.Fatal(err)
			}
		}
		stats(&before)
		retained, err := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, host, pointers)
		if err != nil {
			t.Fatal(err)
		}
		value, copyErr := retained.CopyToHost(t.Context(), output)
		if err := errors.Join(copyErr, retained.Release(t.Context())); err != nil {
			t.Fatal(err)
		}
		stats(&after)
		return bytes.Clone(driver.Bytes(value.Data)), driver.ExecutionStats{
			StreamSynchronizations: after.StreamSynchronizations - before.StreamSynchronizations,
			HostToDeviceBytes:      after.HostToDeviceBytes - before.HostToDeviceBytes,
		}
	}
	// Token t routes to experts t+1 and t+4, so none routes to expert 0.
	reroute := func(host map[*tensor.Tensor]reference.Value) {
		for node, value := range host {
			if node.Name == "selected" {
				for index := range value.Data {
					value.Data[index] = float32(index/int(c.topK) + 1 + index%int(c.topK)*3)
				}
			}
		}
	}
	cuda := newFixtureExecutor(t)
	output, host, pointers, _ := c.prepare(t, cuda, dtype.F32)
	reroute(host)
	resident, residentStats := run(t, cuda, output, host, pointers)
	c.offload = true
	cuda = newFixtureExecutor(t)
	output, host, pointers, _ = c.prepare(t, cuda, dtype.F32)
	reroute(host)
	for attempt := range 3 {
		got, stats := run(t, cuda, output, host, pointers)
		if !bytes.Equal(got, resident) {
			t.Fatalf("run %d: missed experts changed the resident output bits", attempt)
		}
		if stats.StreamSynchronizations != residentStats.StreamSynchronizations {
			t.Fatalf("run %d waited for the stream %d times, the resident run %d", attempt, stats.StreamSynchronizations, residentStats.StreamSynchronizations)
		}
		if attempt == 1 && stats.HostToDeviceBytes < smallest {
			t.Fatalf("the run after the routes moved %d bytes to the card, less than one expert", stats.HostToDeviceBytes)
		}
	}
}

// BenchmarkMoEGroupedMatrices measures resident execution including synchronization.
// Compilation, quantization, upload and warm-up precede the timed loop.
func BenchmarkMoEGroupedMatrices(b *testing.B) {
	library, err := driver.Open()
	if err != nil {
		b.Fatal(err)
	}
	defer library.Close()
	if _, err := library.ReserveDevice(0); err != nil {
		b.Fatal(err)
	}
	for _, tokens := range []uint64{1, 4, 128} {
		b.Run(fmt.Sprintf("tokens=%d", tokens), func(b *testing.B) {
			var c moePinnedCase
			for _, fixture := range moePinnedCases {
				if fixture.name == "mixed-iq" {
					c = fixture
				}
			}
			c.tokens, c.reference = tokens, false
			cuda := newFixtureExecutor(b)
			output, host, pointers, _ := c.prepare(b, cuda, dtype.F32)
			for node, value := range host {
				pointers[node] = copyFixtureDeviceBytes(b, cuda.worker, driver.Bytes(value.Data))
			}
			for _, mode := range []tensor.MulMatCompute{tensor.MulMatComputeExact, tensor.MulMatComputeNativeTensorCore} {
				b.Run(fmt.Sprintf("compute=%d", mode), func(b *testing.B) {
					a := output.Attrs.(tensor.MoEAttributes)
					a.Compute = mode
					output.Attrs = a
					compiled, inputs, err := compileDeviceFeeds([]*tensor.Tensor{output}, pointers)
					if err != nil {
						b.Fatal(err)
					}
					if _, err = cuda.ExecuteCompiled(b.Context(), compiled, nil, inputs); err != nil {
						b.Fatal(err)
					}
					b.ResetTimer()
					for b.Loop() {
						if _, err = cuda.ExecuteCompiled(b.Context(), compiled, nil, inputs); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportMetric(float64(compiled.matmulStagingBytes+compiled.attentionScoreBytes), "scratch-B")
				})
			}
		})
	}
}
