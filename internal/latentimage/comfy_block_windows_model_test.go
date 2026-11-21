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

// The block's update of its stream, its output less its input, is held to ComfyUI's while the arithmetic around its
// int8 products is still overgo's own (moss-comfy-block): ComfyUI folds the norm's weight into one modulated norm
// (rms_adaln) and adds each gated residual with one rounding (addcmul_), where overgo rounds after each step, a rounding
// 2^-8 of a value; and it normalizes and turns q and k in one fused kernel over F32 rotary tables (rms_rope_split_half),
// where overgo rounds its tables and each step to BF16. Its attention is ComfyUI's own since moss-attention-pass,
// comfy_kitchen's int8 attention ported bit for bit (TestTheInt8AttentionIsComfyKitchens), so the block's attention
// output differs from ComfyUI's only where its q and k do. Declared before the first comparison: the block's update lies
// within 5% of ComfyUI's in relative L2, and each token's update within cosine 0.99 of ComfyUI's; a projection read
// under another name, an adapter merged wrong, a rotary table read at the wrong pair or a modulation field out of its
// order lies far outside both. Narrowed by moss-attention-pass to 4% and its attention output held within 2.5%: its
// port took the update from 3.812% to 3.729% of ComfyUI's on the Turbo capture and from 2.016% to 1.943% on text to
// image's, whose lowest token cosine rose from 0.9847 to 0.9952, and the attention output from 1.705% to 1.537% and
// from 2.451% to 2.343%. Each pass row narrows them as it ports its kernel.
const (
	comfyBlockUpdateRelativeL2    = 0.04
	comfyBlockTokenCosine         = 0.99
	comfyBlockAttentionRelativeL2 = 0.025
)

// comfyBlockCaptures are the block captures, each a fixture of fixtures/comfy/moss-ultra and the section ComfyUI ran it
// in: block 20 of the fourth pass of text to image Turbo, whose projections the test also holds, and of text to image,
// both at 1024 x 576.
var comfyBlockCaptures = []struct{ fixture, section string }{
	{"block.json", "text-to-image-turbo"},
	{"block-text-to-image.json", "text-to-image"},
}

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
// scripts/python_golden_references/comfy_moss_block_capture.py into fixtures/comfy/moss-ultra (comfyBlockCaptures) as
// ComfyUI ran the MOSS Ultra workflow's text to image Turbo and text to image sections at 1024 x 576 with the workflow's
// int8 ConvRot file and each section's adapters merged as ComfyUI merges them: block 20 of its fourth pass, its input,
// modulation and rotary tables fed to overgo's block compiled and merged as the section's denoiser compiles and merges
// it, its attention the one the file declares for it, whose update of its stream and attention output are held to
// ComfyUI's at the tolerances declared above; and each of the Turbo block's four int8 ConvRot projections (qkv, o, the
// feed's up and gate, the feed's down with its SiTU), its captured input fed to overgo's projection of the merged
// weight, held to ComfyUI's product bit for bit, as comfy_kitchen's activation quantizer, int8 product and scaling make
// it. The captures hold what the pass rows port beside (the quantizer's int8 rows and scales, the sampler's steps and
// noise), which each holds its kernel to as it lands.
func TestAMOSSBlockIsHeldToComfyUIs(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	repo := testutil.RepoRoot(t)
	references := filepath.Join(testcheck.Must(dataroot.Resolve(repo))(t).References(), "comfy", "moss-ultra", "block")
	root := ultraModelRoot(t)
	spec := testcheck.Must(ReadRotaryTransformer(root))(t)
	readCapture := func(fixture string) comfyBlockCapture {
		var capture comfyBlockCapture
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "fixtures", "comfy", "moss-ultra", fixture)))(t), &capture))
		tokens := capture.Height*capture.Width + capture.TextTokens
		if !capture.FusedQKV || len(capture.Projections) != 4 || !slices.Equal(capture.Tensors["block_input"].Shape, []int{tokens, spec.Hidden}) {
			t.Fatalf("%s holds %d projections over %v; want the fused qkv block's four over %d tokens of %d", fixture, len(capture.Projections), capture.Tensors["block_input"].Shape, tokens, spec.Hidden)
		}
		return capture
	}
	sections := map[string]Request{}
	for _, section := range workflowSections(t) {
		sections[section.slug] = section.request(t, readComfySection(t, "pass-2", section.slug))
	}
	// A section's adapters, as its request applies them; their files close once the merges are made, as a sampling closes
	// them (release).
	sectionAdapters := func(slug string) ([]*mediaAdapter, []float32) {
		var adapters []*mediaAdapter
		var strengths []float32
		for _, applied := range sections[slug].applied() {
			adapters = append(adapters, testcheck.Must(modelAdapter(root, applied.Name))(t))
			strengths = append(strengths, applied.strength())
		}
		t.Cleanup(func() {
			for _, adapter := range adapters {
				testcheck.NoError(t, adapter.release())
			}
		})
		return adapters, strengths
	}
	transformer := filepath.Join(root, "transformer")
	source := testcheck.Must(safetensors.OpenSource(transformer))(t)
	int8 := testcheck.Must(convRotProjections(source, spec.Layers))(t)
	int8Attention := testcheck.Must(int8AttentionLayers(source, spec.Layers))(t)
	testcheck.NoError(t, source.Close())
	session := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	t.Cleanup(func() { testcheck.NoError(t, session.Close(context.WithoutCancel(ctx))) })
	compile := func(label string, adapters []*mediaAdapter, strengths []float32, weights tensor.WeightBindings, merged []mergedProjection, bindings []tensor.WeightBindings, outputs ...*tensor.Tensor) *graphruntime.ResidentProgram {
		t.Helper()
		program := testcheck.Must(compileResidentWeights(ctx, session, label, transformer, weights, nil, outputs...))(t)
		testcheck.NoError(t, bindMerged(ctx, session, program, label, transformer, merged, adapters, strengths))
		for index, adapter := range adapters {
			testcheck.NoError(t, bindAdapterInputs(ctx, session, program, label, adapter, bindings[index]))
		}
		return program
	}
	feed := func(program *graphruntime.ResidentProgram, strengths []float32, feeds map[*tensor.Tensor]reference.Value, adapterStrengths []*tensor.Tensor) map[*tensor.Tensor]reference.Value {
		for index, node := range adapterStrengths {
			if _, required := program.Graph.InputSlot(node); required {
				feeds[node] = reference.Value{Shape: node.Shape, Data: []float32{strengths[index]}}
			}
		}
		return feeds
	}

	for _, held := range comfyBlockCaptures {
		t.Run("block "+held.section, func(t *testing.T) {
			capture := readCapture(held.fixture)
			if !int8Attention[capture.Layer] {
				t.Fatalf("the file declares no int8 attention for layer %d, which ComfyUI ran in it", capture.Layer)
			}
			tokens := capture.Height*capture.Width + capture.TextTokens
			adapters, strengths := sectionAdapters(held.section)
			options := rotaryOptions{eightBit: sections[held.section].EightBitAttention, registers: samplingScoresInRegisters, int8: int8,
				int8Attention: int8Attention, adapters: adapters}
			p := testcheck.Must(compileRotaryJointProgram(spec, mossRefinementEpsilon, tokens, capture.Layer, dtype.BF16, options))(t)
			program := compile("comfy block "+held.section, adapters, strengths, p.weightInputs, p.merged, p.adapterInputs, p.output, p.attention)
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
			got := testcheck.Must(session.Execute(ctx, program, feed(program, strengths, map[*tensor.Tensor]reference.Value{
				p.input: {Shape: p.input.Shape, Data: input}, p.rope: {Shape: p.rope.Shape, Data: rope}, p.unit: {Shape: p.unit.Shape, Data: unit},
				p.modulation: {Shape: p.modulation.Shape, Data: comfyValues(t, references, capture.Tensors["modulation"])},
			}, p.strengths)))(t)
			relative, lowest, lowestToken := blockUpdateAgreement(input, got[p.output].Data, want, spec.Hidden)
			attention, equal := attentionAgreement(got[p.attention].Data, comfyValues(t, references, capture.Tensors["attention_output"]))
			t.Logf("block %d's update of %d tokens: relative L2 %.5f of ComfyUI's, lowest token cosine %.6f (token %d); its attention output at relative L2 %.5f, %.2f%% of its entries ComfyUI's",
				capture.Layer, tokens, relative, lowest, lowestToken, attention, 100*equal)
			if !(relative <= comfyBlockUpdateRelativeL2) || !(lowest >= comfyBlockTokenCosine) {
				t.Errorf("block %d's update lies at relative L2 %.5f of ComfyUI's and token %d's at cosine %.6f; want at most %g and at least %g",
					capture.Layer, relative, lowestToken, lowest, comfyBlockUpdateRelativeL2, comfyBlockTokenCosine)
			}
			if !(attention <= comfyBlockAttentionRelativeL2) {
				t.Errorf("block %d's attention output lies at relative L2 %.5f of ComfyUI's; want at most %g", capture.Layer, attention, comfyBlockAttentionRelativeL2)
			}
		})
	}

	capture := readCapture(comfyBlockCaptures[0].fixture)
	adapters, strengths := sectionAdapters(comfyBlockCaptures[0].section)
	options := rotaryOptions{eightBit: sections[comfyBlockCaptures[0].section].EightBitAttention, registers: samplingScoresInRegisters, int8: int8,
		int8Attention: int8Attention, adapters: adapters}

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
				outputs = []*tensor.Tensor{g.projection(prefix+"ffn_down", g.feedActivation(prefix+"ffn_down", x), intermediate, width)}
			}
			testcheck.NoError(t, g.b.Err())
			program := compile(fmt.Sprintf("comfy projection %d", captured.Order), adapters, strengths, weights, *g.merged, bindings, outputs...)
			got := testcheck.Must(session.Execute(ctx, program, feed(program, strengths, map[*tensor.Tensor]reference.Value{
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
			if parted != 0 {
				t.Errorf("projection %d parts from ComfyUI's product at %d of its %d x %d entries, first at %d (%g against %g)", captured.Order, parted, rows,
					captured.OutFeatures, first, firstGot, want[first])
			}
		})
	}
}
