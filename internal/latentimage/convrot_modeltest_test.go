//go:build modeltest

package latentimage

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// convRotNoiseShare bounds how far, beyond one bfloat16 step, comfy_kitchen's decode of an element may lie from the
// exact one, as a share of its row's scale: its float32 sums over a group of 256 round, and where the exact value
// cancels toward zero that rounding is most of the element's value. Read 2026-11-11 over the golden's three tensors, at
// most 7.7e-5; a decode wrong by one quantization step, a scale / 16, would lie 256 times past the bound.
const convRotNoiseShare = 1.0 / 4096

// TestConvRotDecodesAsComfyKitchenDoes holds the ConvRot decode the MOSS loader runs at load (moss-ultra-import/convrot)
// to comfy_kitchen's, read by scripts/python_golden_references/convrot_decode_golden.py from the imported Ultra files:
// three tensors decoded here give the exact decode's bytes, and with the elements where comfy_kitchen's float32 sums
// rounded otherwise put in (0.05% to 0.09% of each), comfy_kitchen's bytes, each such element within one bfloat16 step
// of the exact value or within convRotNoiseShare of its row's scale. A q, k or v projection is its third of the fused
// qkv, in that order.
func TestConvRotDecodesAsComfyKitchenDoes(t *testing.T) {
	var golden struct {
		Readings []struct {
			File, Tensor string
			Shape        []uint64
			Comfy        string     `json:"comfy_sha256"`
			Exact        string     `json:"exact_sha256"`
			Differ       [][2]int64 `json:"differ"`
		} `json:"readings"`
	}
	root := testutil.RepoRoot(t)
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(root, "fixtures", "comfy", "moss-ultra", "convrot_decode.json")))(t), &golden))
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	ultra := filepath.Join(roots.StoreModels, "moss-ultra")
	if len(golden.Readings) == 0 {
		t.Fatal("the golden reads no tensor")
	}
	for _, reading := range golden.Readings {
		path := filepath.Join(ultra, "transformer", reading.File)
		if _, err := os.Stat(path); err != nil {
			path = filepath.Join(ultra, "text_encoder", reading.File)
		}
		source := testcheck.Must(safetensors.OpenFile(path))(t)
		decoded, shape, quantized, err := convRotWeight(source, reading.Tensor+".weight")
		if err != nil || !quantized || !slices.Equal(shape, reading.Shape) {
			t.Fatalf("%s decoded as %v of shape %v (quantized %t); want %v", reading.Tensor, err, shape, quantized, reading.Shape)
		}
		if got := bitsHash(decoded); got != reading.Exact {
			t.Errorf("%s decodes to %s; want the exact decode, %s", reading.Tensor, got, reading.Exact)
		}
		scales := source.Tensors[reading.Tensor+".weight_scale"]
		scaleBytes := make([]byte, scales.Size())
		testcheck.Must(scales.ReadAt(scaleBytes, 0))(t)
		patched, largest := slices.Clone(decoded), 0.0
		for _, item := range reading.Differ {
			index, comfy := item[0], uint16(item[1])
			scale := float64(math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[4*(uint64(index)/shape[1]):])))
			c, e := float64(dtype.BF16ToFloat32(comfy)), float64(dtype.BF16ToFloat32(patched[index]))
			if step := bf16Step(max(math.Abs(c), math.Abs(e))); math.Abs(c-e) > step {
				largest = max(largest, math.Abs(c-e)/scale)
			}
			patched[index] = comfy
		}
		if got := bitsHash(patched); got != reading.Comfy {
			t.Errorf("%s with its %d listed elements put in hashes %s; want comfy_kitchen's %s", reading.Tensor, len(reading.Differ), got, reading.Comfy)
		}
		if largest > convRotNoiseShare {
			t.Errorf("%s: comfy_kitchen's decode lies %g of a row's scale from the exact one; want at most %g", reading.Tensor, largest, convRotNoiseShare)
		}
		t.Logf("%s: %d of %d elements differ from comfy_kitchen's, the farthest %g of its row's scale", reading.Tensor, len(reading.Differ), len(decoded), largest)
		if reading.Tensor == "layers.0.proj.qkv" {
			key, keyShape, _, err := convRotWeight(source, "layers.0.proj.k.weight")
			third := reading.Shape[0] / 3
			if err != nil || !slices.Equal(keyShape, []uint64{third, reading.Shape[1]}) || !slices.Equal(key, decoded[third*reading.Shape[1]:2*third*reading.Shape[1]]) {
				t.Errorf("the k projection decodes as %v of shape %v; want the fused qkv's middle third", err, keyShape)
			}
		}
		testcheck.NoError(t, source.Close())
	}
}

// bitsHash is the SHA-256 of bfloat16 bits as little-endian bytes.
func bitsHash(bits []uint16) string {
	data := make([]byte, 2*len(bits))
	for index, value := range bits {
		binary.LittleEndian.PutUint16(data[2*index:], value)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// bf16Step is the gap between bfloat16 values at a magnitude: eight bits of mantissa.
func bf16Step(magnitude float64) float64 {
	if magnitude == 0 {
		return 0
	}
	_, exponent := math.Frexp(magnitude)
	return math.Ldexp(1, exponent-8)
}
