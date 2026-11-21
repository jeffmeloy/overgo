//go:build windows

package executor

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/dataroot"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/torchrng"
)

// int8AttentionTensor is one tensor of the capture: the SHA-256 of its bytes as torch holds them, its kind and shape.
// The bytes are in the data root's references/comfy/int8-attention, named by the digest and the kind.
type int8AttentionTensor struct {
	SHA256 string `json:"sha256"`
	DType  string `json:"dtype"`
	Shape  []int  `json:"shape"`
}

// int8AttentionCapture is comfy_kitchen's int8 attention as scripts/python_golden_references/
// comfy_int8_attention_golden.py captured it on the card (fixtures/comfy/int8-attention/golden.json): for each case,
// seeded Q, K and V of heads x length x width in BF16, every stage the CUDA backend leaves (the INT8 codes and scales of
// Q, K and V) and the attention's BF16 output.
type int8AttentionCapture struct {
	Seed  int64 `json:"seed"`
	Heads int   `json:"heads"`
	Width int   `json:"width"`
	Cases []struct {
		Length int     `json:"length"`
		Offset bool    `json:"offset"`
		CTAK   int     `json:"cta_k"`
		Scale  float64 `json:"scale"`
		Inputs struct {
			Q, K, V int8AttentionTensor
		} `json:"inputs"`
		QInt8  int8AttentionTensor `json:"q_int8"`
		QScale int8AttentionTensor `json:"q_scale"`
		KInt8  int8AttentionTensor `json:"k_int8"`
		KScale int8AttentionTensor `json:"k_scale"`
		VInt8  int8AttentionTensor `json:"v_int8"`
		VScale int8AttentionTensor `json:"v_scale"`
		Output int8AttentionTensor `json:"output"`
	} `json:"cases"`
}

// int8AttentionKernels are the int8 attention's launches at more than 256 keys.
var int8AttentionKernels = []string{
	"int8_attention_anchor_f32", "int8_attention_quantize_qk_f32", "int8_attention_quantize_v_f32", "int8_attention_bf16_f32",
}

// TestTheInt8AttentionIsComfyKitchens holds the int8 attention (tensor.AttentionAttributes.Int8, moss-attention-pass)
// to comfy_kitchen's CUDA int8_attention, ComfyUI's attention over the MOSS Ultra workflow, as captured on the card at
// MOSS's shapes (fixtures/comfy/int8-attention/golden.json): from the capture's seeded heads, interleaved as overgo
// holds them, each stage's bytes are the capture's, Q's, K's and V's INT8 codes and scales, and so are the output's BF16
// bits, through the four kernels, and again where the graph rounds the heads, which the attention reads before their
// rounds, as MOSS's blocks feed it; the case whose keys carry an offset takes an anchor key in some head.
func TestTheInt8AttentionIsComfyKitchens(t *testing.T) {
	var capture int8AttentionCapture
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join("..", "..", "..", "fixtures", "comfy", "int8-attention", "golden.json")))(t), &capture))
	if len(capture.Cases) == 0 || capture.Width != int(int8AttentionWidth) {
		t.Fatalf("the capture holds %d cases of width %d; want cases of width %d", len(capture.Cases), capture.Width, int8AttentionWidth)
	}
	for _, captured := range capture.Cases {
		name := fmt.Sprint(captured.Length)
		if captured.Offset {
			name += "-offset"
		}
		t.Run(name, func(t *testing.T) {
			if captured.CTAK != int(int8AttentionKeyTile) {
				t.Fatalf("the capture ran %d-key tiles; the kernel takes %d", captured.CTAK, int8AttentionKeyTile)
			}
			heads, width, length := capture.Heads, capture.Width, captured.Length
			count := heads * length * width
			drawn := testcheck.Must(torchrng.CPUNormal(capture.Seed, 3*count))(t)
			heads3 := [3][]float32{drawn[:count], drawn[count : 2*count], drawn[2*count:]}
			if captured.Offset {
				for index := range heads3[1] {
					offset := float32(3 * (float32(index%width)/float32(width) - 0.5))
					heads3[1][index] += offset
				}
			}
			var unrounded [3][]float32
			for head, values := range heads3 {
				unrounded[head] = slices.Clone(values)
				for index, value := range values {
					values[index] = dtype.RoundBF16(value)
				}
			}
			for index, want := range []int8AttentionTensor{captured.Inputs.Q, captured.Inputs.K, captured.Inputs.V} {
				if digest := bf16Digest(heads3[index]); digest != want.SHA256 {
					t.Fatalf("seeded head %d's BF16 bits read %s; the capture's %s", index, digest, want.SHA256)
				}
			}
			stages, output, ran := runInt8Attention(t, heads3, heads, length, float32(captured.Scale), false)
			for _, kernel := range int8AttentionKernels {
				if ran[kernel] != 1 {
					t.Fatalf("%s ran %d times: %v", kernel, ran[kernel], ran)
				}
			}
			for index, want := range []int8AttentionTensor{captured.QInt8, captured.KInt8, captured.VInt8, captured.QScale, captured.KScale, captured.VScale} {
				checkInt8AttentionStage(t, []string{"Q's codes", "K's codes", "V's codes", "Q's scales", "K's scales", "V's scales"}[index], stages[index], want)
			}
			anchors := 0
			for head := range heads {
				if int32(binary.LittleEndian.Uint32(stages[6][4*head:])) >= 0 {
					anchors++
				}
			}
			if captured.Offset && anchors == 0 {
				t.Fatal("no key head took an anchor though its keys carry an offset")
			}
			t.Logf("%d of %d key heads took an anchor", anchors, heads)
			if index := slices.IndexFunc(output, notBF16); index >= 0 {
				t.Fatalf("output element %d holds the F32 bits %#08x, no BF16 value", index, math.Float32bits(output[index]))
			}
			ordered := make([]float32, count)
			for token := range length {
				for head := range heads {
					copy(ordered[(head*length+token)*width:][:width], output[(token*heads+head)*width:][:width])
				}
			}
			bits := make([]byte, 2*count)
			for index, value := range ordered {
				binary.LittleEndian.PutUint16(bits[2*index:], dtype.Float32ToBF16(value))
			}
			checkInt8AttentionStage(t, "the output", bits, captured.Output)
			// The heads rounded in the graph, as MOSS's blocks round them: the attention reads the pre-round heads, its
			// kernels rounding them as they load, and its output is the same bits, the rounds launching nothing.
			_, fused, fusedRan := runInt8Attention(t, unrounded, heads, length, float32(captured.Scale), true)
			if !slices.EqualFunc(fused, output, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
				t.Error("the attention over heads the graph rounds parts from the attention over the rounded heads")
			}
			for kernel := range fusedRan {
				if !slices.Contains(int8AttentionKernels, kernel) {
					t.Errorf("the attention over heads the graph rounds launched %s beside its own kernels", kernel)
				}
			}
		})
	}
}

// runInt8Attention runs the int8 attention over the heads Q, K and V, each heads x length x width as torch holds them,
// interleaved [token][head][channel] as overgo holds them, the graph rounding each to BF16 first where round says so (as
// MOSS's blocks round them), and returns its staging's parts (as int8StagingParts orders them), its output in overgo's
// layout, and how often each kernel ran.
func runInt8Attention(t *testing.T, heads3 [3][]float32, heads, length int, scale float32, round bool) ([][]byte, []float32, map[string]int) {
	t.Helper()
	width := int(int8AttentionWidth)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(uint64(width), uint64(heads), uint64(length))
	var inputs [3]*tensor.Tensor
	feeds := map[*tensor.Tensor]reference.Value{}
	for index, name := range []string{"query", "key", "value"} {
		inputs[index] = builder.Input(name, dtype.F32, shape)
		interleaved := make([]float32, len(heads3[index]))
		for head := range heads {
			for token := range length {
				copy(interleaved[(token*heads+head)*width:][:width], heads3[index][(head*length+token)*width:][:width])
			}
		}
		feeds[inputs[index]] = reference.Value{Shape: shape, Data: interleaved}
	}
	operands := inputs
	if round {
		for index, input := range inputs {
			operands[index] = builder.BF16RoundNative(input)
		}
	}
	attention := builder.AttentionWithOptions(operands[0], operands[1], operands[2], tensor.AttentionOptions{Scale: scale, Int8: true})
	testcheck.NoError(t, builder.Err())
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	census := map[string]int{}
	testcheck.NoError(t, cuda.worker.Do(t.Context(), func(state *device.State) error {
		state.Trace = &device.LaunchTrace{Census: census}
		return nil
	}))
	got := testcheck.Must(cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), []*tensor.Tensor{attention}, feeds, nil))(t)
	testcheck.NoError(t, cuda.worker.Do(t.Context(), func(state *device.State) error {
		state.Trace = nil
		return nil
	}))
	geometry := eightBitGeometry{queryHeads: uint32(heads), keyValueHeads: uint32(heads), queryTokens: uint32(length),
		keyValueTokens: uint32(length), sequences: 1}
	parts := geometry.int8StagingParts()
	stages := make([][]byte, len(parts))
	testcheck.NoError(t, cuda.worker.Do(t.Context(), func(state *device.State) error {
		if cuda.resources.blas.scores == 0 {
			return fmt.Errorf("the executor holds no staging after the run")
		}
		offsets, _ := alignedParts(cuda.resources.blas.scores, parts[:])
		for index, bytes := range parts {
			stages[index] = make([]byte, bytes)
			if err := state.Driver.MemcpyDtoH(stages[index], offsets[index]); err != nil {
				return err
			}
		}
		return nil
	}))
	return stages, got[attention].Data, maps.Clone(census)
}

// checkInt8AttentionStage fails where a stage's bytes are not the capture's, saying how many elements differ where the
// data root holds the capture's bytes.
func checkInt8AttentionStage(t *testing.T, stage string, got []byte, want int8AttentionTensor) {
	t.Helper()
	sum := sha256.Sum256(got)
	if digest := hex.EncodeToString(sum[:]); digest == want.SHA256 {
		return
	} else {
		detail := "the data root holds no copy of the capture's bytes"
		if roots, err := dataroot.ResolveCurrent(); err == nil {
			path := filepath.Join(roots.References(), "comfy", "int8-attention", want.SHA256+"."+want.DType)
			if captured, err := os.ReadFile(path); err == nil {
				detail = describeInt8AttentionDifference(got, captured, want.DType)
			}
		}
		t.Errorf("%s read %s, the capture's %s: %s", stage, digest, want.SHA256, detail)
	}
}

// describeInt8AttentionDifference counts the elements two copies of a stage differ in and names the first.
func describeInt8AttentionDifference(got, captured []byte, kind string) string {
	if len(got) != len(captured) {
		return fmt.Sprintf("%d bytes against the capture's %d", len(got), len(captured))
	}
	size := map[string]int{"i8": 1, "bf16": 2, "f32": 4}[kind]
	if size == 0 {
		return "of an unknown kind " + kind
	}
	differ, first := 0, -1
	for index := 0; index < len(got); index += size {
		if string(got[index:index+size]) != string(captured[index:index+size]) {
			if first < 0 {
				first = index / size
			}
			differ++
		}
	}
	return fmt.Sprintf("%d of %d elements differ, the first at %d", differ, len(got)/size, first)
}
