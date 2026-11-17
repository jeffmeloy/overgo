package comfyquant

import (
	"bytes"
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestADecodedRowIsComfyKitchensRotation holds DecodeRow to comfy_kitchen's construction of the rotation
// (int8_utils._build_hadamard): H4's Kronecker powers up to the group, over its square root, applied to each group of a
// row's scaled values, for the 256 the language and diffusion weights use and the 64 the Ultra encoder's vision tower
// uses, each value rounded to float32 and then to bfloat16.
func TestADecodedRowIsComfyKitchensRotation(t *testing.T) {
	h4 := [][]float64{{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}}
	for _, size := range []int{64, Group} {
		matrix := h4
		for len(matrix) < size {
			next := make([][]float64, len(matrix)*4)
			for row := range next {
				next[row] = make([]float64, len(matrix)*4)
				for column := range next[row] {
					next[row][column] = matrix[row/4][column/4] * h4[row%4][column%4]
				}
			}
			matrix = next
		}
		root := math.Sqrt(float64(size))
		quantized := make([]byte, 2*size)
		for index := range quantized {
			quantized[index] = byte(int8(index*37%255 - 127))
		}
		scale := float32(0.0371)
		got := make([]uint16, len(quantized))
		DecodeRow(got, quantized, scale, uint64(size))
		for start := 0; start < len(quantized); start += size {
			for column := range size {
				// The matrix's entries are +-1, so the sum of a column's products is an integer, summed exactly.
				sum := 0.0
				for row := range size {
					sum += float64(int8(quantized[start+row])) * matrix[row][column]
				}
				if want := dtype.Float32ToBF16(float32(float64(scale) * sum / root)); got[start+column] != want {
					t.Fatalf("group %d, element %d decodes to %#04x, comfy_kitchen's rotation to %#04x", size, start+column, got[start+column], want)
				}
			}
		}
	}
}

// TestAQuantizedCheckpointReadsAsTheOneItDecodesTo holds the view a repository reads a ComfyUI ConvRot checkpoint
// through (moss-ultra-sections/enhancer-chat): each quantized weight under its own name as its decoded bfloat16 rows,
// read whole or a row at a time alike, every other tensor as stored, neither records nor scales, and a checkpoint that
// holds no quantized weight returned as it is.
func TestAQuantizedCheckpointReadsAsTheOneItDecodesTo(t *testing.T) {
	const rows, columns = 3, Group
	record := []byte(`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`)
	quantized := make([]byte, rows*columns)
	for index := range quantized {
		quantized[index] = byte(int8(index%11 - 5))
	}
	scales := []float32{0.5, 0.25, 4}
	scaleBytes := make([]byte, 4*rows)
	for index, scale := range scales {
		binary.LittleEndian.PutUint32(scaleBytes[4*index:], math.Float32bits(scale))
	}
	plain := []byte{1, 2, 3, 4}
	tensors := map[string]safetensors.Tensor{}
	add := func(name, dataType string, shape []uint64, data []byte) {
		tensors[name] = testcheck.Must(safetensors.NewTensor(name, dataType, shape, bytes.NewReader(data), 0, int64(len(data))))(t)
	}
	add("head.weight", "I8", []uint64{rows, columns}, quantized)
	add("head.weight_scale", "F32", []uint64{rows, 1}, scaleBytes)
	add("head.comfy_quant", "U8", []uint64{uint64(len(record))}, record)
	add("norm.weight", "BF16", []uint64{2}, plain)
	view := testcheck.Must(Decoded(&safetensors.Source{Tensors: tensors}))(t)
	if names := slices.Sorted(func(yield func(string) bool) {
		for name := range view.Tensors {
			if !yield(name) {
				return
			}
		}
	}); !slices.Equal(names, []string{"head.weight", "norm.weight"}) {
		t.Fatalf("the view holds %v, want the decoded weight and the plain tensor alone", names)
	}
	head := view.Tensors["head.weight"]
	if head.DType != "BF16" || !slices.Equal(head.Shape, []uint64{rows, columns}) {
		t.Fatalf("the quantized weight reads as %s %v, want BF16 [%d %d]", head.DType, head.Shape, rows, columns)
	}
	want := make([]byte, rows*columns*2)
	bits := make([]uint16, columns)
	for row := range rows {
		DecodeRow(bits, quantized[row*columns:(row+1)*columns], scales[row], Group)
		for column, value := range bits {
			binary.LittleEndian.PutUint16(want[(row*columns+column)*2:], value)
		}
	}
	whole := make([]byte, len(want))
	testcheck.Must(head.ReadAt(whole, 0))(t)
	second := make([]byte, columns*2)
	testcheck.Must(head.ReadAt(second, int64(columns*2)))(t)
	if !bytes.Equal(whole, want) || !bytes.Equal(second, want[columns*2:2*columns*2]) {
		t.Fatal("the quantized weight does not read as its decoded rows")
	}
	stored := make([]byte, len(plain))
	testcheck.Must(view.Tensors["norm.weight"].ReadAt(stored, 0))(t)
	if !bytes.Equal(stored, plain) {
		t.Fatal("a plain tensor does not read as stored")
	}
	unquantized := &safetensors.Source{Tensors: map[string]safetensors.Tensor{"norm.weight": tensors["norm.weight"]}}
	if testcheck.Must(Decoded(unquantized))(t) != unquantized {
		t.Fatal("a checkpoint holding no quantized weight was not returned as it is")
	}
}
