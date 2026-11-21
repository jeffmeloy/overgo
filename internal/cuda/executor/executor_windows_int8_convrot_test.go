//go:build windows

package executor

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/torchrng"
)

// int8ConvRotCapture is comfy_kitchen's CUDA int8_linear product as scripts/python_golden_references/
// moss_int8_linear_golden.py captured it on the card (mode pattern): the seeded input's and each product's BF16 bits.
type int8ConvRotCapture struct {
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
		Backend       string   `json:"backend"`
		InputSHA256   string   `json:"input_sha256"`
		ProductSHA256 string   `json:"product_sha256"`
		RowSHA256     []string `json:"row_sha256"`
	} `json:"cases"`
}

// TestAnInt8ConvRotProductIsComfyKitchens holds the int8 ConvRot product (moss-int8/product) to comfy_kitchen's CUDA
// int8_linear, the op ComfyUI's linear over the MOSS Ultra workflow's int8 ConvRot weights runs, as captured on the card
// (testdata/int8_convrot_product.json): over each captured shape's patterned weight, the seeded input's BF16 bits are
// the capture's and the product's are too, every row of it, run by the int8 kernels. comfy_kitchen's eager fallback,
// which its Python source shows, departs from its CUDA product in 223,621 of 589,824 elements of MOSS's layer 0 qkv.
func TestAnInt8ConvRotProductIsComfyKitchens(t *testing.T) {
	var capture int8ConvRotCapture
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join("testdata", "int8_convrot_product.json")))(t), &capture))
	if len(capture.Cases) == 0 {
		t.Fatal("the capture holds no product")
	}
	for _, captured := range capture.Cases {
		t.Run(captured.Case, func(t *testing.T) {
			if captured.Backend != "comfy_kitchen.backends.cuda.int8_linear" {
				t.Fatalf("the capture ran %s; want comfy_kitchen's CUDA backend", captured.Backend)
			}
			rows, out, in := captured.Rows, captured.OutFeatures, captured.InFeatures
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
			if digest := bf16Digest(input); digest != captured.InputSHA256 {
				t.Fatalf("the seeded input's BF16 bits read %s; the capture's %s", digest, captured.InputSHA256)
			}
			storage := make([]byte, out*in+out*4)
			for index := range out * in {
				storage[index] = byte(int8((index*61+17)%255 - 127))
			}
			for row := range out {
				binary.LittleEndian.PutUint32(storage[out*in+row*4:], math.Float32bits(float32(1+row%7)/1024))
			}
			product, ran := runInt8ConvRotProduct(t, storage, input, uint64(in), uint64(out), uint64(rows))
			for _, kernel := range []string{"quantize_int8_convrot_columns_f32", "scale_int8_product_bf16_f32"} {
				if !slices.Contains(ran, kernel) {
					t.Fatalf("%s did not run: %v", kernel, ran)
				}
			}
			if index := slices.IndexFunc(product, notBF16); index >= 0 {
				t.Fatalf("product element %d holds the F32 bits %#08x, no BF16 value the next node can read as one", index, math.Float32bits(product[index]))
			}
			var differ []int
			for row := range rows {
				if bf16Digest(product[row*out:(row+1)*out]) != captured.RowSHA256[row] {
					differ = append(differ, row)
				}
			}
			if digest := bf16Digest(product); digest != captured.ProductSHA256 || len(differ) > 0 {
				t.Fatalf("the product's BF16 bits read %s, the capture's %s; rows %v differ", digest, captured.ProductSHA256, differ)
			}
		})
	}
}

// runInt8ConvRotProduct multiplies an int8 ConvRot weight of out rows over in columns, its resident storage the rows'
// bytes and then their F32 scales, by rows token columns of input, and returns the product and the kernels it ran.
func runInt8ConvRotProduct(t *testing.T, storage []byte, input []float32, in, out, rows uint64) ([]float32, []string) {
	t.Helper()
	builder := tensor.NewBuilder()
	left := builder.Input("weight", dtype.I8ConvRot, tensor.MustShape(in, out))
	right := builder.Input("input", dtype.F32, tensor.MustShape(in, rows))
	product := builder.MulMat(left, right)
	testcheck.NoError(t, builder.Err())
	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	census := map[string]int{}
	testcheck.NoError(t, cuda.worker.Do(t.Context(), func(state *device.State) error {
		state.Trace = &device.LaunchTrace{Census: census}
		return nil
	}))
	t.Cleanup(func() {
		_ = cuda.worker.Do(context.WithoutCancel(t.Context()), func(state *device.State) error {
			state.Trace = nil
			return nil
		})
	})
	got := testcheck.Must(cuda.executeWithDeviceFeeds(
		context.WithoutCancel(t.Context()), []*tensor.Tensor{product},
		map[*tensor.Tensor]reference.Value{right: {Shape: right.Shape, Data: input}},
		map[*tensor.Tensor]driver.DevicePtr{left: pointer},
	))(t)
	return got[product].Data, slices.Collect(maps.Keys(census))
}

// notBF16 says an F32 value is no BF16 value, its low 16 bits set (SQA-420): bf16Digest would digest its rounding.
func notBF16(value float32) bool {
	return math.Float32bits(value)&0xffff != 0
}

// bf16Digest is the SHA-256 of values' BF16 bits, little-endian, as the capture digests a bfloat16 tensor's bytes; a
// value BF16 does not hold exactly digests as its rounding, so a product is held to notBF16 first.
func bf16Digest(values []float32) string {
	bits := make([]byte, 2*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint16(bits[2*index:], dtype.Float32ToBF16(value))
	}
	sum := sha256.Sum256(bits)
	return hex.EncodeToString(sum[:])
}
