package latentimage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// convRotGroup is the width of the Hadamard rotation ConvRot quantizes a weight's rows in: 256 columns, the only group
// comfy_kitchen's regular Hadamard matrix (a power of 4) and the MOSS Ultra files use.
const convRotGroup = 256

// convRotRecord is the comfy_quant record a ConvRot weight carries beside its int8 values and per-row scales.
type convRotRecord struct {
	Format    string `json:"format"`
	ConvRot   bool   `json:"convrot"`
	GroupSize int    `json:"convrot_groupsize"`
}

// fusedProjections are the projections a ConvRot file fuses into one qkv weight, in its row order.
var fusedProjections = []string{"q", "k", "v"}

// convRotStored is a weight a ComfyUI ConvRot file holds quantized, as its header and comfy_quant record admit it: its
// int8 values and F32 row scales, and the rows of them that are the weight's, first on.
type convRotStored struct {
	values, scales       safetensors.Tensor
	first, rows, columns uint64
}

// convRotView resolves the weight a graph names (<base>.weight) in a ComfyUI ConvRot file, where the file holds it
// quantized: as <base>.weight in int8 with <base>.weight_scale and <base>.comfy_quant, or, for a q, k or v projection,
// as its third of the fused <prefix>.qkv weight, whose rows are q's, then k's, then v's. It reads the comfy_quant record
// alone and admits the values' and scales' shapes from the header, an F32 scale a row ([rows, 1]), so a weight it
// refuses has no payload read. It reports whether the file holds the weight so.
func convRotView(source *safetensors.Source, name string) (convRotStored, bool, error) {
	base, weight := strings.CutSuffix(name, ".weight")
	if !weight {
		return convRotStored{}, false, nil
	}
	stored, part, parts := base, 0, 1
	if _, quantized := source.Tensors[base+".comfy_quant"]; !quantized {
		at := strings.LastIndex(base, ".")
		prefix, index := base[:max(at, 0)], slices.Index(fusedProjections, base[at+1:])
		if _, fused := source.Tensors[prefix+".qkv.comfy_quant"]; index < 0 || !fused {
			return convRotStored{}, false, nil
		}
		stored, part, parts = prefix+".qkv", index, len(fusedProjections)
	}
	values, scales, recordTensor := source.Tensors[stored+".weight"], source.Tensors[stored+".weight_scale"], source.Tensors[stored+".comfy_quant"]
	record := make([]byte, recordTensor.Size())
	if _, err := recordTensor.ReadAt(record, 0); err != nil {
		return convRotStored{}, true, err
	}
	var declared convRotRecord
	if err := json.Unmarshal(record, &declared); err != nil || declared != (convRotRecord{Format: "int8_tensorwise", ConvRot: true, GroupSize: convRotGroup}) {
		return convRotStored{}, true, fmt.Errorf("convrot: %s declares %q, not int8 ConvRot over %d columns (%v)", stored, record, convRotGroup, err)
	}
	if values.DType != "I8" || len(values.Shape) != 2 || scales.DType != "F32" || !slices.Equal(scales.Shape, []uint64{values.Shape[0], 1}) ||
		values.Shape[1]%convRotGroup != 0 || values.Shape[0]%uint64(parts) != 0 {
		return convRotStored{}, true, fmt.Errorf("convrot: %s is %s %v with scales %s %v, not int8 rows in groups of %d with an F32 scale a row", stored, values.DType, values.Shape, scales.DType, scales.Shape, convRotGroup)
	}
	rows := values.Shape[0] / uint64(parts)
	return convRotStored{values: values, scales: scales, first: uint64(part) * rows, rows: rows, columns: values.Shape[1]}, true, nil
}

// shape is the weight's shape in the file's order (rows, columns).
func (w convRotStored) shape() []uint64 {
	return []uint64{w.rows, w.columns}
}

// decode reads the weight's int8 rows and their scales and decodes them as bfloat16 bits (decodeConvRotRow).
func (w convRotStored) decode() ([]uint16, error) {
	quantized := make([]byte, w.rows*w.columns)
	if _, err := w.values.ReadAt(quantized, int64(w.first*w.columns)); err != nil {
		return nil, err
	}
	scaleBytes := make([]byte, w.rows*4)
	if _, err := w.scales.ReadAt(scaleBytes, int64(w.first*4)); err != nil {
		return nil, err
	}
	decoded := make([]uint16, w.rows*w.columns)
	for row := range w.rows {
		scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[4*row:]))
		decodeConvRotRow(decoded[row*w.columns:(row+1)*w.columns], quantized[row*w.columns:(row+1)*w.columns], scale)
	}
	return decoded, nil
}

// decodeConvRotRow decodes one row of an int8 ConvRot weight as comfy_kitchen's decode means it, (q * scale) times the
// normalized regular Hadamard matrix over each group of 256 columns, computed exactly: the matrix is H4 x H4 x H4 x H4
// over 16, H4 symmetric with entries +-1, so each value is scale * n / 16 for an integer n the transform gives, which a
// float64 holds exactly. Each is rounded to float32 and then to bfloat16, as comfy_kitchen's cast rounds; its own
// float32 sums differ from these on a few elements by their rounding (fixtures/comfy/moss-ultra/convrot_decode.json).
func decodeConvRotRow(out []uint16, quantized []byte, scale float32) {
	var group [convRotGroup]int32
	for start := 0; start < len(quantized); start += convRotGroup {
		for index := range group {
			group[index] = int32(int8(quantized[start+index]))
		}
		for stride := 1; stride < convRotGroup; stride *= 4 {
			for block := 0; block < convRotGroup; block += 4 * stride {
				for offset := block; offset < block+stride; offset++ {
					a, b, c, d := group[offset], group[offset+stride], group[offset+2*stride], group[offset+3*stride]
					group[offset], group[offset+stride], group[offset+2*stride], group[offset+3*stride] = a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d
				}
			}
		}
		for index, n := range group {
			out[start+index] = dtype.Float32ToBF16(float32(float64(scale) * float64(n) / 16))
		}
	}
}
