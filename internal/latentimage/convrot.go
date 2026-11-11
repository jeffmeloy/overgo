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

// convRotWeight reads the weight a graph names (<base>.weight) from a ComfyUI ConvRot file as bfloat16 bits, where the
// file holds it quantized: as <base>.weight in int8 with <base>.weight_scale and <base>.comfy_quant, or, for a q, k or
// v projection, as its third of the fused <prefix>.qkv weight, whose rows are q's, then k's, then v's. It reports
// whether the file holds the weight so, its shape in the file's order (rows, columns) and the decoded rows.
func convRotWeight(source *safetensors.Source, name string) ([]uint16, []uint64, bool, error) {
	base, weight := strings.CutSuffix(name, ".weight")
	if !weight {
		return nil, nil, false, nil
	}
	stored, part, parts := base, 0, 1
	if _, quantized := source.Tensors[base+".comfy_quant"]; !quantized {
		at := strings.LastIndex(base, ".")
		prefix, index := base[:max(at, 0)], slices.Index(fusedProjections, base[at+1:])
		if _, fused := source.Tensors[prefix+".qkv.comfy_quant"]; index < 0 || !fused {
			return nil, nil, false, nil
		}
		stored, part, parts = prefix+".qkv", index, len(fusedProjections)
	}
	values, scales, recordTensor := source.Tensors[stored+".weight"], source.Tensors[stored+".weight_scale"], source.Tensors[stored+".comfy_quant"]
	record := make([]byte, recordTensor.Size())
	if _, err := recordTensor.ReadAt(record, 0); err != nil {
		return nil, nil, true, err
	}
	var declared convRotRecord
	if err := json.Unmarshal(record, &declared); err != nil || declared != (convRotRecord{Format: "int8_tensorwise", ConvRot: true, GroupSize: convRotGroup}) {
		return nil, nil, true, fmt.Errorf("convrot: %s declares %q, not int8 ConvRot over %d columns (%v)", stored, record, convRotGroup, err)
	}
	if values.DType != "I8" || len(values.Shape) != 2 || scales.DType != "F32" || len(scales.Shape) != 2 || scales.Shape[0] != values.Shape[0] ||
		values.Shape[1]%convRotGroup != 0 || values.Shape[0]%uint64(parts) != 0 {
		return nil, nil, true, fmt.Errorf("convrot: %s is %s %v with scales %s %v, not int8 rows in groups of %d with an F32 scale a row", stored, values.DType, values.Shape, scales.DType, scales.Shape, convRotGroup)
	}
	rows, columns := values.Shape[0]/uint64(parts), values.Shape[1]
	first := uint64(part) * rows
	quantized := make([]byte, rows*columns)
	if _, err := values.ReadAt(quantized, int64(first*columns)); err != nil {
		return nil, nil, true, err
	}
	scaleBytes := make([]byte, rows*4)
	if _, err := scales.ReadAt(scaleBytes, int64(first*4)); err != nil {
		return nil, nil, true, err
	}
	decoded := make([]uint16, rows*columns)
	for row := range rows {
		scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[4*row:]))
		decodeConvRotRow(decoded[row*columns:(row+1)*columns], quantized[row*columns:(row+1)*columns], scale)
	}
	return decoded, []uint64{rows, columns}, true, nil
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
