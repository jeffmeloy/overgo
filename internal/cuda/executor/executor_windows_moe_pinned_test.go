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
	"slices"
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
	"ocr":                  "9253e13814f673bce32b8bd0cbf98c34bcf3468f878d9576362f2000b6d259f2",
	"normalized-scaled":    "640ea20b02c7661d9652975cd2edc2af866d3baaa856172b434295b815c12749",
	"sigmoid-fused-biased": "45d83c6d13ad937bc48f0d610a381002eec7670beb805c83b55aeb972ff4f2d5",
	"selected-clamped":     "bfa6d790cc1f1ae664718c651a0efa865922f091e12b3274be2fb81d92186a32",
	"openai":               "4cc7a250477280ae72ea113dc262d2e0edd52bb061f86f1e63f6643e54463cbb",
	"grouped-relu-squared": "f2448f5f3fe386cce4e82486ab3e1c14ec2efbe8cde7010947fb7b7299173680",
	"gelu":                 "51c6670faa75e6c294ac195c72d092d8bfe1f3b74305e614c2cce4b2e4293430",
	"relu":                 "354125a8456ac723740d45746522ae7244b90b0940e7069ba0378760949fa835",
	"ungated":              "39402f6fb65ba43a17ec952c92ee4f9865db08da2052ad048815533938a08336",
	"many-experts":         "631b08ca5de2a032145799cf8cb391508e26e855bdadc311ea2ff0034a229527",
	"token-groups":         "8bd27bb1dec8dabefb79dc85c263d2e650b81b43c812e19ff1e7f4ac4ba8d7cb",
	"q8_0":                 "dae2e73fd8c5dab0a7bc6a2162bc5519064ccb6ab2d90c94cf956b2e83a073a8",
	"q4_0":                 "60edb9a269683abdef966b9ee10641270672792a44999cebb83d3bdd0edfe518",
	"q4_K":                 "a9cc720814154ede1118c99e0d80b83f78ecfef1fce2f698ca9a922afde13232",
	"iq4_xs":               "d0875e8f5f01b2f74882ff1420976e728f13741cac8a0e5e11b3aa88ad46dcfc",
	"tq2_0":                "3dfd1b21c7b4960d90c66634699a27b12fe4bd18023429059566f803f809a120",
	"mxfp4":                "a4e2408298fa943bedca343e29a2a352fb4aa04e35ed26bc4501dd0bb60ca7b1",
	"q1_0":                 "aae0a71f90ec9cc301dfc136aa71f7fe1d7059a3512d0ee507657df67aa9827c",
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
							if tokens == 1 {
								// A product of a q8 storage takes q8 activations.
								if slices.ContainsFunc(c.banks[:], moeQ8Storage) {
									compareRelativeRMS(t, got, want, accuracyQ8Experts)
									return
								}
								compare(t, got, want, accuracyModel)
								return
							}
							// The grouped products follow the router logits, one F32 product and the router's bias where
							// the case has one, and the routing; an F32 bank's weights multiply as f16.
							compare(t, got, want, accuracyHalfStaged)
							firstProduct := 8 + 6*8 + (32 + 5*8 + 5*4)
							if c.biases {
								firstProduct += 32 + 3*8 + 12*4
							}
							if len(trace.Data()) < firstProduct+8 || binary.LittleEndian.Uint64(trace.Data()[firstProduct:]) != uint64(kernelMoeGroupedHiddenF32) {
								t.Fatal("grouped expert products did not execute")
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
// down launches are the span's dot kernels with the banks' storages, indexed
// banks keep the output's bits, an input that starts off 16-byte alignment
// stays within the reference, and each token run alone, through the
// one-token kernels, computes the span's bits for it.
func TestExecutorMoEExpertsUseBlockDots(t *testing.T) {
	// Serial: device fixtures share the one card.
	const (
		hiddenRecord = moeRecordHeader + 7*8 + 13*4
		// The hidden record's storages follow its pointers and four scalars, the down record's its pointers and five.
		hiddenGateStorage = moeRecordHeader + 7*8 + 4*4
		downStorage       = moeRecordHeader + 6*8 + 5*4
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
			// Gate and up rows of two 256-weight blocks; four decode tokens, a span's most, stay below grouped matrix reuse,
			// and their eight routes over four experts share some.
			c := moePinnedCase{
				name: "block-dots-" + name, compute: tensor.MulMatComputeNativeTensorCore,
				hidden: 512, routerHidden: 512, intermediate: 256, experts: 4, tokens: moeSpanTokens, divisor: 1, topK: 2,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
				banks: banks, reference: true,
			}
			// The resident run sets the reference, the bits and the span's token feeds; it runs first.
			var want []float32
			var spanFeeds map[string][]float32
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
				input := output.Inputs[0]
				host[input] = reference.Value{Shape: input.Shape, Data: moeFullMantissas(host[input].Data)}
				feeds := maps.Clone(host)
				if variant.shifted {
					// One float ahead of an allocation leaves the input 4 bytes past 16-byte alignment.
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
				inputs, hidden := raw[moeDecodeRecords:], raw[moeDecodeRecords+moeQuantizeRecord:]
				activations, down := hidden[hiddenRecord:], hidden[hiddenRecord+moeQuantizeRecord:]
				word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
				if word(inputs) != uint64(kernelQuantizeQ80InputF32) || word(activations) != uint64(kernelQuantizeQ80InputF32) {
					t.Fatalf("%s: launches %d and %d, want the q8 quantizations %d", variant.name, word(inputs), word(activations), kernelQuantizeQ80InputF32)
				}
				if word(hidden) != uint64(kernelMoeHiddenSpanQ8F32) || word(down) != uint64(kernelMoeDownQ8F32) {
					t.Fatalf("%s: launches %d and %d, want the dot kernels %d and %d",
						variant.name, word(hidden), word(down), kernelMoeHiddenSpanQ8F32, kernelMoeDownQ8F32)
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
					// An unaligned input takes the router's per-weight dots, whose sums differ in their last bits.
					compareRelativeRMS(t, value.Data, want, accuracyQ8Experts)
				default:
					resident = bytes.Clone(driver.Bytes(value.Data))
					spanFeeds = moeTokenFeeds(host)
					maps.Copy(host, bankValues)
					for bank := range bankValues {
						bank.Type = dtype.F32
					}
					expected, err := reference.Execute([]*tensor.Tensor{output}, host)
					if err != nil {
						t.Fatal(err)
					}
					want = expected[output].Data
					compareRelativeRMS(t, value.Data, want, accuracyQ8Experts)
				}
			}
			if !bytes.Equal(moeTokensAlone(t, c, spanFeeds), resident) {
				t.Fatal("the span's shared expert reads changed a token's output bits")
			}
		})
	}
}

// TestExecutorMoESpanRepeatedRoutes runs a span whose tokens route twice to
// one bank expert through the index divisor, among experts they share, and
// holds each token's output to its bits run alone.
func TestExecutorMoESpanRepeatedRoutes(t *testing.T) {
	// Serial: device fixtures share the one card.
	// Token t selects experts t, t+3, t+6 and t+1 of eight, two a bank: an even token routes twice to bank t/2.
	c := moePinnedCase{
		name: "span-repeated", compute: tensor.MulMatComputeNativeTensorCore,
		hidden: 512, routerHidden: 512, intermediate: 256, experts: 8, tokens: moeSpanTokens, divisor: 2, topK: 4,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
		selected: true, banks: [3]dtype.Type{dtype.IQ3S, dtype.IQ3S, dtype.IQ4NL},
	}
	cuda := newFixtureExecutor(t)
	trace := &device.LaunchTrace{}
	if err := cuda.worker.Do(t.Context(), func(state *device.State) error { state.Trace = trace; return nil }); err != nil {
		t.Fatal(err)
	}
	output, host, pointers, _ := c.prepare(t, cuda, dtype.F32)
	input := output.Inputs[0]
	host[input] = reference.Value{Shape: input.Shape, Data: moeFullMantissas(host[input].Data)}
	retained, err := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, host, pointers)
	if err != nil {
		t.Fatal(err)
	}
	value, copyErr := retained.CopyToHost(t.Context(), output)
	if err := errors.Join(copyErr, retained.Release(t.Context())); err != nil {
		t.Fatal(err)
	}
	if kernel := binary.LittleEndian.Uint64(trace.Data()[moeDecodeRecords+moeQuantizeRecord:]); kernel != uint64(kernelMoeHiddenSpanQ8F32) {
		t.Fatalf("hidden launch %d, want the span kernel %d", kernel, kernelMoeHiddenSpanQ8F32)
	}
	if !bytes.Equal(moeTokensAlone(t, c, moeTokenFeeds(host)), driver.Bytes(value.Data)) {
		t.Fatal("a repeated route changed a token's output bits")
	}
}

const (
	// A launch record is four header words, then eight bytes a pointer and four a scalar.
	moeRecordHeader = 4 * 8
	// moeDecodeRecords: a decode group's router-logits and route records, which its hidden launch, or its q8
	// quantization of moeQuantizeRecord bytes, follows.
	moeDecodeRecords  = moeRecordHeader + 4*8 + 3*4 + moeRecordHeader + 5*8 + 5*4
	moeQuantizeRecord = moeRecordHeader + 2*8 + 4
)

// accuracyQ8Experts bounds the q8 routed products' relative RMS error against
// the F32 reference: int8 activations in the gate and up products and again
// in the down product measured 1.0% on every block storage of the block-dot
// case.
const accuracyQ8Experts = 2e-2

// compareRelativeRMS holds got to want by the RMS of their difference over
// the RMS of want.
func compareRelativeRMS(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	var difference, norm float64
	for index := range want {
		delta := float64(got[index] - want[index])
		difference += delta * delta
		norm += float64(want[index]) * float64(want[index])
	}
	if relative := math.Sqrt(difference / norm); !(relative <= tolerance) {
		t.Fatalf("relative RMS error %g, want at most %g", relative, tolerance)
	}
}

// moeFullMantissas scales BF16-representable values by 1.1, which fills their
// mantissas: a weight group's sum of them then depends on its order, where
// the cases' own values sum exactly in any.
func moeFullMantissas(values []float32) []float32 {
	full := make([]float32, len(values))
	for index, value := range values {
		full[index] = value * float32(1.1)
	}
	return full
}

// moeTokenFeeds returns a case's feeds that hold one column a token, by name.
func moeTokenFeeds(host map[*tensor.Tensor]reference.Value) map[string][]float32 {
	feeds := make(map[string][]float32)
	for node, value := range host {
		if node.Name == "input" || node.Name == "router_input" || node.Name == "selected" {
			feeds[node.Name] = value.Data
		}
	}
	return feeds
}

// moeTokensAlone runs each token of a case alone, its token feeds cut from
// the span's, and returns the outputs in token order.
func moeTokensAlone(t *testing.T, c moePinnedCase, span map[string][]float32) []byte {
	t.Helper()
	alone := c
	alone.tokens = 1
	var outputs []byte
	for token := range c.tokens {
		cuda := newFixtureExecutor(t)
		output, host, pointers, _ := alone.prepare(t, cuda, dtype.F32)
		for node, value := range host {
			if values, ok := span[node.Name]; ok {
				width := uint64(len(values)) / c.tokens
				host[node] = reference.Value{Shape: value.Shape, Data: values[token*width : (token+1)*width]}
			}
		}
		retained, err := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, host, pointers)
		if err != nil {
			t.Fatal(err)
		}
		value, copyErr := retained.CopyToHost(t.Context(), output)
		if err := errors.Join(copyErr, retained.Release(t.Context())); err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, driver.Bytes(value.Data)...)
	}
	return outputs
}

// groupedProductBits: SHA-256 of each fixture's grouped output by its tokens, on the card and from host memory alike,
// as the grouped kernels of 96c26fe39 produced them.
var groupedProductBits = map[string]map[uint64]string{
	"iq3_s-iq4_nl": {
		64:   "23d007dba6496d751f07f78f3a73c13b038e4622c6a6f6cf999b00531156301d",
		1561: "b1b58415fdcc92ae933b7f5521ed366fe5c808896bcae561d8050e43b3db865e",
	},
	"iq4_xs-q8_0": {
		64:   "96039559c245146fe93871befd04376253bbeead4f01198e66aeeda2e6715ef9",
		1561: "9c4996472e1feed6bc0f3812f9d4ab5129feb8fcf83e4322202eb3aa323bebee",
	},
	"fused-iq4_xs-iq4_nl": {
		64:   "220fd780786fd9b88c9ac3a3b5b11486862e64e5280972c21e12db0980ff0ef3",
		1561: "6dac3cb05cf1d5308fc7a80fc9066aae3d38bdf2d0f34b739d34fb98cdb0a8f5",
	},
	"bf16": {
		64:   "c711491e313fb45fe4ba5bcbe82db392c24bc29b4ebc2136f7f9d0a916ca35d7",
		1561: "1628bed78504dcf1dee872e6fae4751b305eaa7ed645dc9b3fd1f8c58723898c",
	},
}

// TestPromptExpertsMultiplyEncodedWeights holds a prompt's routed experts to
// the grouped products over their encoded weights:
//   - Flash-Next's routed layer takes the staging's tokens a group, so the 1,561-token document and the 4,019-token prompt are one group each and a 20,605-token prompt two of one size
//   - fixtures of Flash-Next's IQ3_S and IQ4_NL layer and its IQ4_XS and Q8_0 layer, fused IQ4_XS gate and up, and BF16 banks
//     of 96 weights a row, which decode a weight at a time; 16 tokens, under a tile, take the q8 products, and BF16 banks the row dots
//   - 64 tokens launch the grouped kernels with the banks' storages and stay within the model tolerance of the host reference,
//     with the experts on the card and in host memory alike: block formats' whole numbers and BF16 weights hold exactly in f16,
//     and each input splits into two f16 parts; three experts take 128 routed rows, so one of them takes two tiles;
//     experts in host memory are copied to the card and multiply in a batch after the card's
//   - the document's 1,561 tokens on experts in host memory keep the direct banks' bits
//   - every grouped run keeps the bits groupedProductBits pins
func TestPromptExpertsMultiplyEncodedWeights(t *testing.T) {
	// Serial: device fixtures share the one card.
	// documentTokens: the frozen Strata document's prompt tokens under Flash-Next's chat template.
	const documentTokens = 1561
	for _, c := range []struct {
		tokens uint64
		group  uint32
	}{{documentTokens, documentTokens}, {4019, 4019}, {20605, 10303}} {
		const hidden, intermediate, experts = 2560, 640, 512
		builder := tensor.NewBuilder()
		builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
		input := builder.Input("input", dtype.F32, tensor.MustShape(hidden, c.tokens))
		router := builder.Input("router", dtype.F32, tensor.MustShape(hidden, experts))
		up := builder.Input("up", dtype.IQ3S, tensor.MustShape(hidden, intermediate, experts))
		down := builder.Input("down", dtype.IQ4NL, tensor.MustShape(intermediate, hidden, experts))
		node := builder.MoEWithOptions(input, router, up, down, tensor.MoEOptions{
			Gate: builder.Input("gate", dtype.IQ3S, tensor.MustShape(hidden, intermediate, experts)),
			TopK: 10, Scale: 1, Routing: tensor.MoERoutingSoftmax, Activation: tensor.MoEActivationSiLU, ExpertIndexDivisor: 1,
		})
		if err := builder.Err(); err != nil {
			t.Fatal(err)
		}
		if group, _, err := moeScratchGroup(node); err != nil || group != c.group {
			t.Fatalf("%d Flash-Next tokens: group %d (%v), want %d", c.tokens, group, err, c.group)
		}
	}
	const (
		// A launch record is four header words, then eight bytes a pointer and four a scalar.
		headerBytes = 4 * 8
		// The hidden launch follows the router logits and the routing: a group of a tile or more takes
		// its logits as one product, an external call of six words, and a smaller group by the kernel.
		kernelLogits, productLogits, routeRecord = headerBytes + 4*8 + 3*4, 8 + 6*8, headerBytes + 5*8 + 5*4
		// The grouped hidden launch's storages follow its eight pointers and three scalars, the down launch's its six pointers and two of four.
		hiddenRecord = headerBytes + 8*8 + 10*4
		gateStorage  = headerBytes + 8*8 + 3*4
		downStorage  = headerBytes + 6*8 + 2*4
		downRecord   = headerBytes + 6*8 + 4*4
		gridWord     = 8
		// The card keeps expert 0 of each bank beside the tables of a pointer an expert.
		pointerBytes, slotAlignment = 8, 256
	)
	word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
	for _, fixture := range []struct {
		name   string
		banks  [3]dtype.Type
		fused  bool
		hidden uint64
	}{
		{"iq3_s-iq4_nl", [3]dtype.Type{dtype.IQ3S, dtype.IQ3S, dtype.IQ4NL}, false, 256},
		{"iq4_xs-q8_0", [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.Q8_0}, false, 512},
		{"fused-iq4_xs-iq4_nl", [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}, true, 512},
		{"bf16", [3]dtype.Type{dtype.BF16, dtype.BF16, dtype.BF16}, false, 96},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			c := moePinnedCase{
				name: "encoded-" + fixture.name, compute: tensor.MulMatComputeNativeTensorCore,
				hidden: fixture.hidden, routerHidden: fixture.hidden, intermediate: 128, experts: 3, divisor: 1, topK: 2,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
				banks: fixture.banks, fused: fixture.fused,
			}
			banks := []tensor.Shape{tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.intermediate, c.hidden)}
			if c.fused {
				banks = []tensor.Shape{tensor.MustShape(c.hidden, 2*c.intermediate), banks[2]}
			}
			hostLimit := uint64(len(banks)) * c.experts * pointerBytes
			for index, shape := range banks {
				expert, err := shape.Bytes(fixture.banks[3-len(banks)+index])
				if err != nil {
					t.Fatal(err)
				}
				hostLimit += (expert + slotAlignment - 1) / slotAlignment * slotAlignment
			}
			for _, run := range []struct {
				placement     string
				tokens, limit uint64
				grouped       bool
			}{{"card", 16, 0, false}, {"card", 64, 0, true}, {"host", 64, hostLimit, true}, {"host", documentTokens, hostLimit, true}} {
				t.Run(fmt.Sprintf("%s/tokens=%d", run.placement, run.tokens), func(t *testing.T) {
					// The direct banks' run sets the reference; over the document's tokens the host's takes minutes, so there it sets the bits.
					c.tokens, c.offload, c.reference = run.tokens, false, run.tokens < documentTokens
					direct, want := c.run(t, newFixtureExecutor(t), dtype.F32)
					cuda := newFixtureExecutor(t)
					trace := &device.LaunchTrace{}
					if err := cuda.worker.Do(t.Context(), func(state *device.State) error { state.Trace = trace; return nil }); err != nil {
						t.Fatal(err)
					}
					c.offload, c.limit = true, run.limit
					got, _ := c.run(t, cuda, dtype.F32)
					firstProduct := kernelLogits + routeRecord
					if run.tokens >= uint64(moeTileTokens) {
						firstProduct = productLogits + routeRecord
					}
					hidden := trace.Data()[firstProduct:]
					if !run.grouped {
						if !moeQ8Storage(fixture.banks[1]) {
							if word(hidden) != uint64(kernelMoeHiddenDotF32) {
								t.Fatalf("a group under a tile launched kernel %d, not the row dots", word(hidden))
							}
							compare(t, got, want, accuracyModel)
							return
						}
						// Block storages take the q8 products after the input's quantization.
						if word(hidden) != uint64(kernelQuantizeQ80InputF32) || word(hidden[moeQuantizeRecord:]) != uint64(kernelMoeHiddenQ8F32) {
							t.Fatalf("a group under a tile launched kernels %d and %d, not the q8 products", word(hidden), word(hidden[moeQuantizeRecord:]))
						}
						compareRelativeRMS(t, got, want, accuracyQ8Experts)
						return
					}
					down := hidden[hiddenRecord:]
					storages := [3]dtype.Type{dtype.Type(scalar(hidden[gateStorage:])), dtype.Type(scalar(hidden[gateStorage+4:])), dtype.Type(scalar(down[downStorage:]))}
					if word(hidden) != uint64(kernelMoeGroupedHiddenF32) || word(down) != uint64(kernelMoeGroupedDownF32) || storages != fixture.banks {
						t.Fatalf("launches %d and %d with storages %v, want the grouped kernels with %v", word(hidden), word(down), storages, fixture.banks)
					}
					// The card's experts, then each batch of experts copied from host memory, take a hidden and a down launch.
					tiles, batches := uint64(0), 0
					for raw := hidden; len(raw) >= hiddenRecord+downRecord && word(raw) == uint64(kernelMoeGroupedHiddenF32); raw = raw[hiddenRecord+downRecord:] {
						tiles += uint64(scalar(raw[gridWord:]))
						batches++
					}
					if tiles <= c.experts || (batches > 1) != (run.placement == "host") {
						t.Fatalf("%d tiles in %d batches for %d experts, want more tiles than experts and a copied batch in host memory", tiles, batches, c.experts)
					}
					digest := sha256.Sum256(driver.Bytes(got))
					if bits := hex.EncodeToString(digest[:]); bits != groupedProductBits[fixture.name][run.tokens] {
						t.Errorf("output %s, pinned %s", bits, groupedProductBits[fixture.name][run.tokens])
					}
					if want == nil {
						if !bytes.Equal(driver.Bytes(got), driver.Bytes(direct)) {
							t.Fatal("experts in host memory changed the direct banks' bits")
						}
						return
					}
					compare(t, got, want, accuracyModel)
				})
			}
		})
	}
}

// TestPromptExpertCopiesOverlapCompute runs a prompt group of 64 tokens over
// five BF16 experts of 24 MiB, each more than half a copy buffer, with only
// expert 0 on the card: the four experts in host memory copy in a batch
// each, so each copy buffer takes two batches in turn.
//   - the run waits for the stream as often as the run with every expert on the card:
//     no batch's copies hold the host until the kernels before them end
//   - every expert in host memory is copied, once
//   - the output keeps the card run's bits and the serial copy's, pinned from 7a396356b with the
//     router's logits as one F32 product since long-prompt/prefill-rate
func TestPromptExpertCopiesOverlapCompute(t *testing.T) {
	// serialCopyBits: SHA-256 of the output as the serial copy at 7a396356b produced it, with the router's
	// logits as one F32 product (long-prompt/prefill-rate), which sums them in another order.
	const serialCopyBits = "4c15ae63debccec8af99062e4163b780ef53dd3357eae3585dae8af4135567f0"
	banks := [3]dtype.Type{dtype.BF16, dtype.BF16, dtype.BF16}
	c := moePinnedCase{
		name: "copy-overlap", compute: tensor.MulMatComputeNativeTensorCore,
		hidden: 2048, routerHidden: 2048, intermediate: 2048, experts: 5, tokens: 64, divisor: 1, topK: 2,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
		banks: banks,
	}
	// The residency's tables hold a pointer an expert, and one 256-byte-aligned slot of each bank holds expert 0 alone.
	const pointerBytes, slotAlignment = 8, 256
	c.limit = uint64(len(banks)) * c.experts * pointerBytes
	var expertBytes uint64
	for _, storage := range banks {
		size, err := tensor.MustShape(c.hidden, c.intermediate).Bytes(storage)
		if err != nil {
			t.Fatal(err)
		}
		size = (size + slotAlignment - 1) / slotAlignment * slotAlignment
		c.limit += size
		expertBytes += size
	}
	if expertBytes > moeBatchBytes || 2*expertBytes <= moeBatchBytes {
		t.Fatalf("an expert of %d bytes shares a copy buffer of %d bytes or overflows it", expertBytes, moeBatchBytes)
	}
	// run executes the node once and returns its output bits and the
	// executor's stream waits and host-to-device bytes during the run.
	run := func(offload bool) ([]byte, driver.ExecutionStats) {
		t.Helper()
		c.offload = offload
		cuda := newFixtureExecutor(t)
		output, host, pointers, _ := c.prepare(t, cuda, dtype.F32)
		stats := func() (stats driver.ExecutionStats) {
			if err := cuda.worker.Do(t.Context(), func(state *device.State) error { stats = state.Driver.ExecutionStats(); return nil }); err != nil {
				t.Fatal(err)
			}
			return stats
		}
		before := stats()
		retained, err := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, host, pointers)
		if err != nil {
			t.Fatal(err)
		}
		value, copyErr := retained.CopyToHost(t.Context(), output)
		if err := errors.Join(copyErr, retained.Release(t.Context())); err != nil {
			t.Fatal(err)
		}
		after := stats()
		return bytes.Clone(driver.Bytes(value.Data)), driver.ExecutionStats{
			StreamSynchronizations: after.StreamSynchronizations - before.StreamSynchronizations,
			HostToDeviceBytes:      after.HostToDeviceBytes - before.HostToDeviceBytes,
		}
	}
	card, cardStats := run(false)
	host, hostStats := run(true)
	if hostStats.StreamSynchronizations != cardStats.StreamSynchronizations {
		t.Errorf("the run with experts in host memory waited for the stream %d times, the card's run %d",
			hostStats.StreamSynchronizations, cardStats.StreamSynchronizations)
	}
	if copied := hostStats.HostToDeviceBytes - cardStats.HostToDeviceBytes; copied < (c.experts-1)*expertBytes {
		t.Errorf("the run moved %d more bytes to the card than the card's run, less than its %d experts in host memory", copied, c.experts-1)
	}
	if !bytes.Equal(host, card) {
		t.Error("copied experts changed the card run's bits")
	}
	digest := sha256.Sum256(host)
	if got := hex.EncodeToString(digest[:]); got != serialCopyBits {
		t.Errorf("output bits %s, the serial copy's %s", got, serialCopyBits)
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
