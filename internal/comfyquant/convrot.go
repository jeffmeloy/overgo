// Package comfyquant reads ComfyUI's quantized checkpoints: a weight held as int8 ConvRot (comfy_kitchen's
// int8_tensorwise with a Hadamard rotation), with an F32 scale a row and a comfy_quant record beside it, read as the
// bfloat16 weight it decodes to. The MOSS Ultra workflow's files hold their diffusion model and their text encoder so.
package comfyquant

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"sync"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// Group is the width of the Hadamard rotation ConvRot quantizes most of a weight's rows in: 256 columns, the group the
// MOSS Ultra files' language and diffusion weights use. comfy_kitchen builds its regular Hadamard matrix for any power
// of 4 (int8_utils._build_hadamard), and the Ultra encoder's vision tower is rotated over 64.
const Group = 256

// hadamardBase is the order of the matrix whose Kronecker powers a group's rotation is.
const hadamardBase = 4

// The names a quantized weight's companions take beside its base: its comfy_quant record and its row scales.
const (
	recordSuffix = ".comfy_quant"
	scaleSuffix  = ".weight_scale"
)

// record is the comfy_quant record a ConvRot weight carries beside its int8 values and per-row scales.
type record struct {
	Format    string `json:"format"`
	ConvRot   bool   `json:"convrot"`
	GroupSize int    `json:"convrot_groupsize"`
}

// fusedProjections are the projections a ConvRot file fuses into one qkv weight, in its row order.
var fusedProjections = []string{"q", "k", "v"}

// Weight is a weight a ComfyUI ConvRot file holds quantized, as its header and comfy_quant record admit it: its int8
// values and F32 row scales, the rows of them that are the weight's, first on, and the group its rows are rotated in.
type Weight struct {
	values, scales              safetensors.Tensor
	first, rows, columns, group uint64
}

// rotationGroup is whether a declared group is one comfy_kitchen rotates in: a power of 4.
func rotationGroup(size int) bool {
	if size < hadamardBase {
		return false
	}
	for size > 1 {
		if size%hadamardBase != 0 {
			return false
		}
		size /= hadamardBase
	}
	return true
}

// Find resolves the weight a graph names (<base>.weight) in a ComfyUI ConvRot file, where the file holds it quantized:
// as <base>.weight in int8 with <base>.weight_scale and <base>.comfy_quant, or, for a q, k or v projection, as its
// third of the fused <prefix>.qkv weight, whose rows are q's, then k's, then v's. It reads the comfy_quant record alone
// and admits the values' and scales' shapes from the header, an F32 scale a row ([rows, 1]), so a weight it refuses
// has no payload read. It reports whether the file holds the weight so.
func Find(source *safetensors.Source, name string) (Weight, bool, error) {
	base, weight := strings.CutSuffix(name, ".weight")
	if !weight {
		return Weight{}, false, nil
	}
	stored, part, parts := base, 0, 1
	if _, quantized := source.Tensors[base+recordSuffix]; !quantized {
		at := strings.LastIndex(base, ".")
		prefix, index := base[:max(at, 0)], slices.Index(fusedProjections, base[at+1:])
		if _, fused := source.Tensors[prefix+".qkv"+recordSuffix]; index < 0 || !fused {
			return Weight{}, false, nil
		}
		stored, part, parts = prefix+".qkv", index, len(fusedProjections)
	}
	values, scales, recordTensor := source.Tensors[stored+".weight"], source.Tensors[stored+scaleSuffix], source.Tensors[stored+recordSuffix]
	declaration := make([]byte, recordTensor.Size())
	if _, err := recordTensor.ReadAt(declaration, 0); err != nil {
		return Weight{}, true, err
	}
	var declared record
	if err := json.Unmarshal(declaration, &declared); err != nil || declared.Format != "int8_tensorwise" || !declared.ConvRot || !rotationGroup(declared.GroupSize) {
		return Weight{}, true, fmt.Errorf("convrot: %s declares %q, not int8 ConvRot over a power of 4 columns (%v)", stored, declaration, err)
	}
	group := uint64(declared.GroupSize)
	if values.DType != "I8" || len(values.Shape) != 2 || scales.DType != "F32" || !slices.Equal(scales.Shape, []uint64{values.Shape[0], 1}) ||
		values.Shape[1]%group != 0 || values.Shape[0]%uint64(parts) != 0 {
		return Weight{}, true, fmt.Errorf("convrot: %s is %s %v with scales %s %v, not int8 rows in groups of %d with an F32 scale a row", stored, values.DType, values.Shape, scales.DType, scales.Shape, group)
	}
	rows := values.Shape[0] / uint64(parts)
	return Weight{values: values, scales: scales, first: uint64(part) * rows, rows: rows, columns: values.Shape[1], group: group}, true, nil
}

// Shape is the weight's shape in the file's order (rows, columns).
func (w Weight) Shape() []uint64 {
	return []uint64{w.rows, w.columns}
}

// Decode reads the weight's int8 rows and their scales and decodes them as bfloat16 bits (DecodeRow).
func (w Weight) Decode() ([]uint16, error) {
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
		DecodeRow(decoded[row*w.columns:(row+1)*w.columns], quantized[row*w.columns:(row+1)*w.columns], scale, w.group)
	}
	return decoded, nil
}

// DecodeRow decodes one row of an int8 ConvRot weight as comfy_kitchen's decode means it, (q * scale) times the
// normalized regular Hadamard matrix over each group of size columns (a power of 4), computed exactly: the matrix is
// H4's Kronecker power over the square root of size, 16 for 256, H4 symmetric with entries +-1, so each value is
// scale * n over that root for an integer n the transform gives, which a float64 holds exactly. Each is rounded to
// float32 and then to bfloat16, as comfy_kitchen's cast rounds; its own float32 sums differ from these on a few elements
// by their rounding (fixtures/comfy/moss-ultra/convrot_decode.json).
func DecodeRow(out []uint16, quantized []byte, scale float32, size uint64) {
	group := make([]int32, size)
	root := 1.0
	for stride := uint64(1); stride < size; stride *= hadamardBase {
		root *= 2
	}
	for start := uint64(0); start < uint64(len(quantized)); start += size {
		for index := range group {
			group[index] = int32(int8(quantized[start+uint64(index)]))
		}
		for stride := uint64(1); stride < size; stride *= hadamardBase {
			for block := uint64(0); block < size; block += hadamardBase * stride {
				for offset := block; offset < block+stride; offset++ {
					a, b, c, d := group[offset], group[offset+stride], group[offset+2*stride], group[offset+3*stride]
					group[offset], group[offset+stride], group[offset+2*stride], group[offset+3*stride] = a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d
				}
			}
		}
		for index, n := range group {
			out[start+uint64(index)] = dtype.Float32ToBF16(float32(float64(scale) * float64(n) / root))
		}
	}
}

// Decoded is a source's tensors with each weight a ComfyUI ConvRot file holds quantized read as the bfloat16 weight it
// decodes to, under its own name, decoded as its rows are read, and neither its record nor its scales; the source
// returned as it is where it holds no quantized weight. The view takes the source's files over (Retensored).
func Decoded(source *safetensors.Source) (*safetensors.Source, error) {
	quantized := false
	for name := range source.Tensors {
		if strings.HasSuffix(name, recordSuffix) {
			quantized = true
			break
		}
	}
	if !quantized {
		return source, nil
	}
	tensors := map[string]safetensors.Tensor{}
	for name, value := range source.Tensors {
		if strings.HasSuffix(name, recordSuffix) || strings.HasSuffix(name, scaleSuffix) {
			continue
		}
		weight, held, err := Find(source, name)
		if err != nil {
			return nil, err
		}
		if !held {
			tensors[name] = value
			continue
		}
		decoded, err := safetensors.NewTensor(name, "BF16", weight.Shape(), &rowReader{weight: weight}, 0, int64(weight.rows*weight.columns*2))
		if err != nil {
			return nil, fmt.Errorf("convrot: %s: %w", name, err)
		}
		tensors[name] = decoded
	}
	return source.Retensored(tensors), nil
}

// blockRows is how many rows a weight's reader decodes at once, so a reader reading a weight whole reads its values and
// scales in blocks.
const blockRows = 256

// rowReader reads a ConvRot weight as the bfloat16 rows it decodes to, little-endian, decoding a block of rows at a
// time and keeping the last block, so a weight read whole or row by row decodes each row once.
type rowReader struct {
	weight Weight
	mu     sync.Mutex
	first  uint64
	block  []byte
}

// ReadAt reads the decoded weight's bytes from offset, as many as destination holds or the weight has left.
func (r *rowReader) ReadAt(destination []byte, offset int64) (int, error) {
	rowBytes := int64(r.weight.columns * 2)
	read := 0
	for read < len(destination) {
		position := offset + int64(read)
		row := uint64(position / rowBytes)
		if row >= r.weight.rows {
			return read, io.EOF
		}
		block, first, err := r.blockOf(row)
		if err != nil {
			return read, err
		}
		read += copy(destination[read:], block[position-int64(first)*rowBytes:])
	}
	return read, nil
}

// blockOf is the decoded block that holds row, and the block's first row.
func (r *rowReader) blockOf(row uint64) ([]byte, uint64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.block != nil && row >= r.first && row < r.first+uint64(len(r.block))/(r.weight.columns*2) {
		return r.block, r.first, nil
	}
	first := row - row%blockRows
	rows := min(blockRows, r.weight.rows-first)
	columns := r.weight.columns
	quantized := make([]byte, rows*columns)
	if _, err := r.weight.values.ReadAt(quantized, int64((r.weight.first+first)*columns)); err != nil {
		return nil, 0, err
	}
	scales := make([]byte, rows*4)
	if _, err := r.weight.scales.ReadAt(scales, int64((r.weight.first+first)*4)); err != nil {
		return nil, 0, err
	}
	bits := make([]uint16, columns)
	block := make([]byte, rows*columns*2)
	for index := range rows {
		DecodeRow(bits, quantized[index*columns:(index+1)*columns], math.Float32frombits(binary.LittleEndian.Uint32(scales[4*index:])), r.weight.group)
		for column, value := range bits {
			binary.LittleEndian.PutUint16(block[(index*columns+uint64(column))*2:], value)
		}
	}
	r.first, r.block = first, block
	return block, first, nil
}
