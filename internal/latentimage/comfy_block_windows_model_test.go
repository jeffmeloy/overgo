//go:build modeltest && windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// The block's update of its stream, its output less its input, is held to ComfyUI's while the block's attention and
// the arithmetic around its int8 products are still overgo's own (moss-comfy-block): ComfyUI's comfy_kitchen 8-bit
// attention quantizes q and k and the probabilities by blocks of its own, and overgo's 8-bit attention by its own;
// ComfyUI folds the norm's weight into one modulated norm (rms_adaln) and adds each gated residual with one rounding
// (addcmul_), where overgo rounds after each step, a rounding 2^-8 of a value; its rotary tables are F32, overgo's
// rounded to BF16. Declared before the first comparison: the block's update lies within 5% of ComfyUI's in relative
// L2, and each token's update within cosine 0.99 of ComfyUI's. A projection read under another name, an adapter
// merged wrong, a rotary table read at the wrong pair or a modulation field out of its order lies far outside both.
// Each pass row narrows them as it ports its kernel.
const (
	comfyBlockUpdateRelativeL2 = 0.05
	comfyBlockTokenCosine      = 0.99
)

// The feed's down projection is held to ComfyUI's product at a tolerance while its SiTU is still overgo's
// (moss-situ-prologue): ComfyUI's int8 path folds the SiTU into comfy_kitchen's quantizer, which computes it in F32
// (int8_linear.cu load_input_act: 4 * stable_tanh(gate / 4) / (1 + expf(-gate)) * 25 * stable_tanh(hidden / 25),
// under --use_fast_math) and rotates it unrounded, where overgo's prologue rounds each of the chain's steps to BF16 as
// MOSS's eager module does (moss_situ_value), 2^-8 of a value, which moves some entries of a row by one int8 step.
// Declared before its first comparison, after the first run found the product parting at 160,204 of its 196,608
// entries: the product lies within 2% of ComfyUI's in relative L2. Halves taken in the wrong order, a bound misread or
// the weight merged wrong lie far outside it.
const comfyBlockSiTUProductRelativeL2 = 0.02

// comfyCaptured is one tensor scripts/python_golden_references/comfy_moss_block_capture.py captured: its bytes'
// SHA-256, by which the references name them, their kind (bf16, f32 or i8) and its shape.
type comfyCaptured struct {
	SHA256 string `json:"sha256"`
	DType  string `json:"dtype"`
	Shape  []int  `json:"shape"`
}

// comfyBlockCapture is fixtures/comfy/moss-ultra/block.json: one joint block of MOSS's denoiser as ComfyUI ran it in
// the MOSS Ultra workflow's text to image Turbo section at 1024 x 576, its int8 projections and its sampler's steps.
type comfyBlockCapture struct {
	Layer       int                      `json:"layer"`
	Call        int                      `json:"call"`
	Rows        int                      `json:"rows"`
	Height      int                      `json:"height"`
	Width       int                      `json:"width"`
	TextTokens  int                      `json:"text_tokens"`
	FusedQKV    bool                     `json:"fused_qkv"`
	Tensors     map[string]comfyCaptured `json:"tensors"`
	Projections []struct {
		Order       int           `json:"order"`
		InFeatures  int           `json:"in_features"`
		OutFeatures int           `json:"out_features"`
		InputAct    *string       `json:"input_act"`
		Route       string        `json:"route"`
		Input       comfyCaptured `json:"input"`
		Output      comfyCaptured `json:"output"`
	} `json:"projections"`
}

// comfyValues are a captured tensor's values in F32, its bytes read from the references and checked against their
// hash: BF16 and F32 exactly.
func comfyValues(t *testing.T, references string, captured comfyCaptured) []float32 {
	t.Helper()
	data := testcheck.Must(os.ReadFile(filepath.Join(references, captured.SHA256+"."+captured.DType)))(t)
	if digest := sha256.Sum256(data); hex.EncodeToString(digest[:]) != captured.SHA256 {
		t.Fatalf("the captured %v tensor hashes %x; want %s", captured.Shape, digest, captured.SHA256)
	}
	width := map[string]int{"bf16": 2, "f32": 4}[captured.DType]
	if width == 0 {
		t.Fatalf("a captured %s tensor has no F32 values", captured.DType)
	}
	values := make([]float32, len(data)/width)
	for index := range values {
		if width == 2 {
			values[index] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[2*index:])) << 16)
		} else {
			values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*index:]))
		}
	}
	return values
}

// TestAMOSSBlockIsHeldToComfyUIs holds one joint block of MOSS's denoiser to ComfyUI's (moss-comfy-block), captured by
// scripts/python_golden_references/comfy_moss_block_capture.py into fixtures/comfy/moss-ultra/block.json as ComfyUI ran
// the MOSS Ultra workflow's text to image Turbo section at 1024 x 576 with the workflow's int8 ConvRot file and its
// adapters merged as ComfyUI merges them: block 20 of its fourth pass, its input, modulation and rotary tables fed to
// overgo's block compiled and merged as the section's denoiser compiles and merges it, whose update of its stream is
// held to ComfyUI's at the tolerance declared above; and each of the block's four int8 ConvRot projections (qkv, o,
// the feed's up and gate, the feed's down with its SiTU), its captured input fed to overgo's projection of the merged
// weight, held to ComfyUI's product bit for bit, as comfy_kitchen's activation quantizer, int8 product and scaling
// make it. The capture holds what the pass rows port beside (the quantizer's int8 rows and scales, the attention's
// output, the sampler's steps and noise), which each holds its kernel to as it lands.
func TestAMOSSBlockIsHeldToComfyUIs(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	repo := testutil.RepoRoot(t)
	var capture comfyBlockCapture
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "fixtures", "comfy", "moss-ultra", "block.json")))(t), &capture))
	references := filepath.Join(testcheck.Must(dataroot.Resolve(repo))(t).References(), "comfy", "moss-ultra", "block")
	root := ultraModelRoot(t)
	spec := testcheck.Must(ReadRotaryTransformer(root))(t)
	tokens := capture.Height*capture.Width + capture.TextTokens
	if !capture.FusedQKV || len(capture.Projections) != 4 || !slices.Equal(capture.Tensors["block_input"].Shape, []int{tokens, spec.Hidden}) {
		t.Fatalf("the capture holds %d projections over %v; want the fused qkv block's four over %d tokens of %d", len(capture.Projections), capture.Tensors["block_input"].Shape, tokens, spec.Hidden)
	}
	var request Request
	for _, section := range workflowSections(t) {
		if section.slug == "text-to-image-turbo" {
			request = section.request(t, readComfySection(t, "pass-2", section.slug))
		}
	}
	var adapters []*mediaAdapter
	var strengths []float32
	for _, applied := range request.applied() {
		adapters = append(adapters, testcheck.Must(modelAdapter(root, applied.Name))(t))
		strengths = append(strengths, applied.strength())
	}
	// The adapters' files close once the merges are made, as a sampling closes them (release).
	t.Cleanup(func() {
		for _, adapter := range adapters {
			testcheck.NoError(t, adapter.release())
		}
	})
	transformer := filepath.Join(root, "transformer")
	source := testcheck.Must(safetensors.OpenSource(transformer))(t)
	int8 := testcheck.Must(convRotProjections(source, spec.Layers))(t)
	testcheck.NoError(t, source.Close())
	options := rotaryOptions{eightBit: request.EightBitAttention, registers: samplingScoresInRegisters, int8: int8, adapters: adapters}
	session := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	t.Cleanup(func() { testcheck.NoError(t, session.Close(context.WithoutCancel(ctx))) })
	compile := func(label string, weights tensor.WeightBindings, merged []mergedProjection, bindings []tensor.WeightBindings, outputs ...*tensor.Tensor) *graphruntime.ResidentProgram {
		t.Helper()
		program := testcheck.Must(compileResidentWeights(ctx, session, label, transformer, weights, nil, outputs...))(t)
		testcheck.NoError(t, bindMerged(ctx, session, program, label, transformer, merged, adapters, strengths))
		for index, adapter := range adapters {
			testcheck.NoError(t, bindAdapterInputs(ctx, session, program, label, adapter, bindings[index]))
		}
		return program
	}
	feed := func(program *graphruntime.ResidentProgram, feeds map[*tensor.Tensor]reference.Value, adapterStrengths []*tensor.Tensor) map[*tensor.Tensor]reference.Value {
		for index, node := range adapterStrengths {
			if _, required := program.Graph.InputSlot(node); required {
				feeds[node] = reference.Value{Shape: node.Shape, Data: []float32{strengths[index]}}
			}
		}
		return feeds
	}

	t.Run("block", func(t *testing.T) {
		p := testcheck.Must(compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, capture.Layer, dtype.BF16, options))(t)
		program := compile("comfy block", p.weightInputs, p.merged, p.adapterInputs, p.output)
		input := comfyValues(t, references, capture.Tensors["block_input"])
		want := comfyValues(t, references, capture.Tensors["block_output"])
		pe := comfyValues(t, references, capture.Tensors["rope"])
		headWidth, half := spec.HeadDim, spec.HeadDim/2
		rope := make([]float32, headWidth*tokens)
		for token := range tokens {
			for pair := range half {
				at := (token*half + pair) * 4
				rope[token*headWidth+pair], rope[token*headWidth+half+pair] = pe[at+2], pe[at]
			}
		}
		unit := make([]float32, spec.Hidden)
		for index := range unit {
			unit[index] = tensor.UnitScale
		}
		got := testcheck.Must(session.Execute(ctx, program, feed(program, map[*tensor.Tensor]reference.Value{
			p.input: {Shape: p.input.Shape, Data: input}, p.rope: {Shape: p.rope.Shape, Data: rope}, p.unit: {Shape: p.unit.Shape, Data: unit},
			p.modulation: {Shape: p.modulation.Shape, Data: comfyValues(t, references, capture.Tensors["modulation"])},
		}, p.strengths)))(t)[p.output].Data
		var differenceSquares, wantSquares float64
		lowest, lowestToken := 1.0, 0
		for token := range tokens {
			var dot, gotSquares, referenceSquares float64
			for column := token * spec.Hidden; column < (token+1)*spec.Hidden; column++ {
				g, r := float64(got[column]-input[column]), float64(want[column]-input[column])
				dot += g * r
				gotSquares += g * g
				referenceSquares += r * r
				differenceSquares += (g - r) * (g - r)
			}
			wantSquares += referenceSquares
			if cosine := dot / math.Sqrt(gotSquares*referenceSquares); !(cosine >= lowest) {
				lowest, lowestToken = cosine, token
			}
		}
		relative := math.Sqrt(differenceSquares / wantSquares)
		t.Logf("block %d's update of %d tokens: relative L2 %.5f of ComfyUI's, lowest token cosine %.6f (token %d)", capture.Layer, tokens, relative, lowest, lowestToken)
		if !(relative <= comfyBlockUpdateRelativeL2) || !(lowest >= comfyBlockTokenCosine) {
			t.Errorf("block %d's update lies at relative L2 %.5f of ComfyUI's and token %d's at cosine %.6f; want at most %g and at least %g",
				capture.Layer, relative, lowestToken, lowest, comfyBlockUpdateRelativeL2, comfyBlockTokenCosine)
		}
	})

	for _, captured := range capture.Projections {
		t.Run(fmt.Sprintf("projection %d", captured.Order), func(t *testing.T) {
			if captured.Route != "fused_convrot" {
				t.Fatalf("ComfyUI's projection %d took the %s route; the capture holds no quantizer of it", captured.Order, captured.Route)
			}
			rows := capture.Rows
			var weights tensor.WeightBindings
			g := testcheck.Must(newRotaryGraph(spec, mossRefinementEpsilon, rows, dtype.BF16, &weights))(t)
			bindings := make([]tensor.WeightBindings, len(adapters))
			adapterStrengths := g.use(options, bindings)
			prefix := fmt.Sprintf("layers.%d.proj.", capture.Layer)
			width, intermediate := uint64(spec.Hidden), uint64(spec.Intermediate)
			x := g.b.Input("x", dtype.F32, tensor.MustShape(uint64(captured.InFeatures), uint64(rows)))
			var outputs []*tensor.Tensor
			switch captured.Order {
			case 0:
				outputs = []*tensor.Tensor{g.projection(prefix+"q", x, width, width), g.projection(prefix+"k", x, width, width), g.projection(prefix+"v", x, width, width)}
			case 1:
				outputs = []*tensor.Tensor{g.projection(prefix+"o", x, width, width)}
			case 2:
				outputs = []*tensor.Tensor{g.projection(prefix+"ffn_up_gate", x, width, tensor.PairedExtent*intermediate)}
			case 3:
				if captured.InputAct == nil || *captured.InputAct != "situ" {
					t.Fatalf("ComfyUI's feed down took input activation %v; want its SiTU", captured.InputAct)
				}
				up, gate := g.part(x, tensor.FirstOffset, intermediate), g.part(x, intermediate, intermediate)
				outputs = []*tensor.Tensor{g.projection(prefix+"ffn_down", g.situ(up, gate), intermediate, width)}
			}
			testcheck.NoError(t, g.b.Err())
			program := compile(fmt.Sprintf("comfy projection %d", captured.Order), weights, *g.merged, bindings, outputs...)
			got := testcheck.Must(session.Execute(ctx, program, feed(program, map[*tensor.Tensor]reference.Value{
				x: {Shape: x.Shape, Data: comfyValues(t, references, captured.Input)},
			}, adapterStrengths)))(t)
			want := comfyValues(t, references, captured.Output)
			parted, first, firstGot := 0, -1, float32(0)
			var differenceSquares, wantSquares float64
			for row := range rows {
				column := 0
				for _, output := range outputs {
					values, out := got[output].Data, int(output.Shape.Dims[0])
					for at := range out {
						index := row*captured.OutFeatures + column + at
						difference := float64(values[row*out+at]) - float64(want[index])
						differenceSquares, wantSquares = differenceSquares+difference*difference, wantSquares+float64(want[index])*float64(want[index])
						if math.Float32bits(values[row*out+at]) != math.Float32bits(want[index]) {
							if parted++; first < 0 {
								firstGot = values[row*out+at]
								first = index
							}
						}
					}
					column += out
				}
			}
			relative := math.Sqrt(differenceSquares / wantSquares)
			t.Logf("projection %d parts from ComfyUI's product at %d of its %d x %d entries, at relative L2 %.3g", captured.Order, parted, rows, captured.OutFeatures, relative)
			switch {
			case captured.InputAct != nil && !(relative <= comfyBlockSiTUProductRelativeL2):
				t.Errorf("projection %d, its SiTU still overgo's, lies at relative L2 %.3g of ComfyUI's product; want at most %g", captured.Order, relative, comfyBlockSiTUProductRelativeL2)
			case captured.InputAct == nil && parted != 0:
				t.Errorf("projection %d parts from ComfyUI's product at %d of its %d x %d entries, first at %d (%g against %g)", captured.Order, parted, rows,
					captured.OutFeatures, first, firstGot, want[first])
			}
		})
	}
}
