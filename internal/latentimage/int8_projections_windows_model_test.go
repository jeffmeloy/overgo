//go:build modeltest && windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/torchrng"
)

// bf16Multiplies is what a MOSS pass at 2368 x 1344 spent in its BF16 matrix multiplies (moss-speed/do, Nsight
// Systems), and int8Multiplies what the int8 ConvRot products may spend (moss-int8): ComfyUI's whole pass takes 0.555 s.
const (
	bf16Multiplies = 760 * time.Millisecond
	int8Multiplies = 450 * time.Millisecond
)

// int8LayerCapture is comfy_kitchen's CUDA int8_linear product over the workflow's own projections, as
// scripts/python_golden_references/moss_int8_linear_golden.py captured it on the card (mode layer).
type int8LayerCapture struct {
	Seed     int64 `json:"seed"`
	Outliers struct {
		Period int     `json:"period"`
		Phase  int     `json:"phase"`
		Gain   float32 `json:"gain"`
	} `json:"outliers"`
	Cases []struct {
		Case          string   `json:"case"`
		Rows          int      `json:"rows"`
		OutFeatures   int      `json:"out_features"`
		InFeatures    int      `json:"in_features"`
		InputSHA256   string   `json:"input_sha256"`
		ProductSHA256 string   `json:"product_sha256"`
		RowSHA256     []string `json:"row_sha256"`
	} `json:"cases"`
}

// TestMOSSMultipliesInEightBitsAsComfyKitchen holds MOSS's sampling on the MOSS Ultra workflow's int8 ConvRot file to
// multiplying its joint projections as ComfyUI does (samplingInt8, moss-int8): the projections it binds from the file
// multiply the capture's seeded input to comfy_kitchen's CUDA product, bit for bit, over layer 0's four and layer 41's
// qkv and feed-forward output (fixtures/comfy/moss-ultra/int8_linear.json; a fused qkv as its q, k and v thirds); a
// pass at the sections' own size saves at least what brings its multiplies from bf16Multiplies to int8Multiplies; and
// each golden section's final latent keeps at least the agreement with its recording BF16 projections keep
// (holdInt8Sections).
func TestMOSSMultipliesInEightBitsAsComfyKitchen(t *testing.T) {
	cudatest.Require(t)
	root := ultraModelRoot(t)
	t.Run("layers", func(t *testing.T) { holdInt8Layers(t, filepath.Join(root, "transformer")) })
	t.Run("pass", func(t *testing.T) { holdInt8Pass(t, root) })
	t.Run("sections", func(t *testing.T) { holdInt8Sections(t, root) })
}

// holdInt8Sections reads each golden section's agreement with its recording with BF16 projections and then with int8,
// in one run, and holds each section's final latent, the denoiser's own output, to keeping at least BF16's agreement.
// The finished picture's is read beside it: on 2026-11-19 int8 kept text to image's latent at 2.09 dB where BF16 keeps
// 1.86, and its picture at 15.12 dB where BF16 keeps 15.39. A section whose adapters patch the projections multiplies
// them in BF16 either way (rotaryGraph.projection).
func holdInt8Sections(t *testing.T, root string) {
	profile := testcheck.Must(ResolveProfile(root))(t)
	served := samplingInt8
	t.Cleanup(func() { samplingInt8 = served })
	samplingInt8 = false
	bf16 := readGoldenCases(t, root, profile, goldenCases)
	samplingInt8 = true
	int8 := readGoldenCases(t, root, profile, goldenCases)
	for _, golden := range goldenCases {
		before, after := bf16[golden.name], int8[golden.name]
		t.Logf("%s: the final latent keeps %.2f dB with int8 projections, %.2f dB with BF16; the finished picture %.2f dB and %.2f dB",
			golden.name, after.latent, before.latent, after.picture, before.picture)
		if !(after.latent >= before.latent) {
			t.Errorf("%s keeps %.2f dB of its recording's latent with int8 projections, %.2f dB with BF16; want int8's at least BF16's",
				golden.name, after.latent, before.latent)
		}
	}
}

// holdInt8Layers multiplies the capture's seeded input by each captured projection as sampling binds it from the
// transformer's file (dtype.I8ConvRot, residentPayload) and holds each product row's BF16 bits to the capture's.
func holdInt8Layers(t *testing.T, transformer string) {
	ctx := t.Context()
	var capture int8LayerCapture
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(testutil.FixturePath(t, "comfy", "moss-ultra", "int8_linear.json")))(t), &capture))
	runtime := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	defer func() { testcheck.NoError(t, runtime.Close(context.WithoutCancel(ctx))) }()
	for _, captured := range capture.Cases {
		rows, in := captured.Rows, captured.InFeatures
		input := testcheck.Must(torchrng.CPUNormal(capture.Seed, rows*in))(t)
		for index := range input {
			if index%in%capture.Outliers.Period == capture.Outliers.Phase {
				input[index] *= capture.Outliers.Gain
			}
			if index/in == rows/2 {
				input[index] = 0
			}
			input[index] = dtype.RoundBF16(input[index])
		}
		if digest := bf16Bits(input); digest != captured.InputSHA256 {
			t.Fatalf("%s: the seeded input's BF16 bits read %s; the capture's %s", captured.Case, digest, captured.InputSHA256)
		}
		names := []string{captured.Case}
		if base, fused := strings.CutSuffix(captured.Case, ".qkv"); fused {
			names = []string{base + ".q", base + ".k", base + ".v"}
		}
		builder := tensor.NewBuilder()
		var weights tensor.WeightBindings
		bind := tensor.WeightInputs{Builder: builder, Bindings: &weights, MatrixType: dtype.BF16}
		x := builder.Input("x", dtype.F32, tensor.MustShape(uint64(in), uint64(rows)))
		var products []*tensor.Tensor
		part := uint64(captured.OutFeatures / len(names))
		for _, name := range names {
			products = append(products, builder.MulMat(bind.InputOf(dtype.I8ConvRot, name+".weight", uint64(in), part), x))
		}
		testcheck.NoError(t, builder.Err())
		program := testcheck.Must(compileResidentWeights(ctx, runtime, "int8 projections "+captured.Case, transformer, weights, []*tensor.Tensor{x}, products...))(t)
		got := testcheck.Must(runtime.Execute(ctx, program, map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: input}}))(t)
		product := make([]float32, 0, rows*captured.OutFeatures)
		for row := range rows {
			for _, node := range products {
				product = append(product, got[node].Data[row*int(part):(row+1)*int(part)]...)
			}
		}
		testcheck.NoError(t, runtime.Release(context.WithoutCancel(ctx), program))
		// A value whose low 16 bits are set is no BF16 value the next node reads as one, and bf16Bits would digest its
		// rounding (SQA-420).
		if index := slices.IndexFunc(product, func(value float32) bool { return math.Float32bits(value)&0xffff != 0 }); index >= 0 {
			t.Fatalf("%s: product element %d holds the F32 bits %#08x, no BF16 value", captured.Case, index, math.Float32bits(product[index]))
		}
		var differ []int
		for row := range rows {
			if bf16Bits(product[row*captured.OutFeatures:(row+1)*captured.OutFeatures]) != captured.RowSHA256[row] {
				differ = append(differ, row)
			}
		}
		if digest := bf16Bits(product); digest != captured.ProductSHA256 || len(differ) > 0 {
			t.Errorf("%s: the product's BF16 bits read %s, comfy_kitchen's %s; rows %v differ", captured.Case, digest, captured.ProductSHA256, differ)
		}
	}
}

// holdInt8Pass samples the own-size text section's request at guidance 1 from one generator at 2 and 6 steps with the
// BF16 and the int8 projections, a pass being the difference over four steps, and holds the int8 pass to saving at
// least bf16Multiplies less int8Multiplies.
func holdInt8Pass(t *testing.T, root string) {
	ctx := t.Context()
	profile := testcheck.Must(ResolveProfile(root))(t)
	recording := readComfySection(t, "own-size", "text-to-image")
	request := textSection(t, recording, "598", "436", "592", "599")
	request.Width, request.Height, request.Canvas = 0, 0, &media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}
	request = testcheck.Must(request.Sized(root))(t)
	one := float32(1)
	request.GuidanceScale = &one
	served := samplingInt8
	t.Cleanup(func() { samplingInt8 = served })
	asking := func(steps int) Request {
		asked := request
		asked.Steps = steps
		return asked
	}
	generator, loaded := testcheck.Must(LoadGenerator(ctx, root, profile, asking(2)))(t), true
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	sample := func(steps int, int8 bool) time.Duration {
		t.Helper()
		samplingInt8 = int8
		asked := asking(steps)
		if !loaded {
			testcheck.NoError(t, generator.Reset(ctx, asked))
		}
		loaded = false
		session := testcheck.Must(generator.prepare(ctx, asked))(t)
		start := time.Now()
		testcheck.Must(generator.integrate(ctx, session))(t)
		wall := time.Since(start)
		testcheck.Must(generator.decode(ctx, session))(t)
		return wall
	}
	sample(2, false)
	bf16 := (sample(6, false) - sample(2, false)) / 4
	sample(2, true)
	int8 := (sample(6, true) - sample(2, true)) / 4
	t.Logf("a conditional pass at %dx%d: %v with BF16 projections, %v with int8; the multiplies took %v in BF16",
		request.Width, request.Height, bf16.Round(time.Millisecond), int8.Round(time.Millisecond), bf16Multiplies)
	if saved := bf16 - int8; saved < bf16Multiplies-int8Multiplies {
		t.Errorf("with int8 projections a pass saves %v of %v; want at least %v, its multiplies at most %v",
			saved.Round(time.Millisecond), bf16.Round(time.Millisecond), bf16Multiplies-int8Multiplies, int8Multiplies)
	}
}

// bf16Bits is the SHA-256 of values' BF16 bits, little-endian, as the capture digests a bfloat16 tensor's bytes.
func bf16Bits(values []float32) string {
	bits := make([]byte, 2*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint16(bits[2*index:], dtype.Float32ToBF16(value))
	}
	sum := sha256.Sum256(bits)
	return hex.EncodeToString(sum[:])
}
