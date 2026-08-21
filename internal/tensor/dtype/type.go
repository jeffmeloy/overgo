package dtype

import (
	"errors"
	"fmt"
	"math"
)

// Type: ggml-compatible tensor storage type
type Type uint32

const (
	F32    Type = 0
	F16    Type = 1
	Q4_0   Type = 2
	Q4_1   Type = 3
	Q5_0   Type = 6
	Q5_1   Type = 7
	Q8_0   Type = 8
	Q8_1   Type = 9
	Q2K    Type = 10
	Q3K    Type = 11
	Q4K    Type = 12
	Q5K    Type = 13
	Q6K    Type = 14
	Q8K    Type = 15
	IQ2XXS Type = 16
	IQ2XS  Type = 17
	IQ3XXS Type = 18
	IQ1S   Type = 19
	IQ4NL  Type = 20
	IQ3S   Type = 21
	IQ2S   Type = 22
	IQ4XS  Type = 23
	I8     Type = 24
	I16    Type = 25
	I32    Type = 26
	I64    Type = 27
	F64    Type = 28
	IQ1M   Type = 29
	BF16   Type = 30
	TQ1_0  Type = 34
	TQ2_0  Type = 35
	MXFP4  Type = 39
	NVFP4  Type = 40
	Q1_0   Type = 41
	Q2_0   Type = 42
	// F8E4M3: OCP F8_E4M3FN payload with a per-output-row F32 scale carried
	// alongside (native-dtype residency). Not a self-contained GGUF block type;
	// this id is internal to the native-dtype matmul path. 1 byte/element.
	F8E4M3 Type = 43
	Count       = F8E4M3 + 1
)

// Traits defines type's physical block layout
type Traits struct {
	Name      string
	BlockSize uint64
	TypeSize  uint64
	Quantized bool
}

var traits = map[Type]Traits{
	F32:    {"f32", 1, 4, false},
	F16:    {"f16", 1, 2, false},
	Q4_0:   {"q4_0", 32, 18, true},
	Q4_1:   {"q4_1", 32, 20, true},
	Q5_0:   {"q5_0", 32, 22, true},
	Q5_1:   {"q5_1", 32, 24, true},
	Q8_0:   {"q8_0", 32, 34, true},
	Q8_1:   {"q8_1", 32, 36, true},
	Q2K:    {"q2_K", 256, 84, true},
	Q3K:    {"q3_K", 256, 110, true},
	Q4K:    {"q4_K", 256, 144, true},
	Q5K:    {"q5_K", 256, 176, true},
	Q6K:    {"q6_K", 256, 210, true},
	Q8K:    {"q8_K", 256, 292, true},
	IQ2XXS: {"iq2_xxs", 256, 66, true},
	IQ2XS:  {"iq2_xs", 256, 74, true},
	IQ3XXS: {"iq3_xxs", 256, 98, true},
	IQ1S:   {"iq1_s", 256, 50, true},
	IQ4NL:  {"iq4_nl", 32, 18, true},
	IQ3S:   {"iq3_s", 256, 110, true},
	IQ2S:   {"iq2_s", 256, 82, true},
	IQ4XS:  {"iq4_xs", 256, 136, true},
	I8:     {"i8", 1, 1, false},
	I16:    {"i16", 1, 2, false},
	I32:    {"i32", 1, 4, false},
	I64:    {"i64", 1, 8, false},
	F64:    {"f64", 1, 8, false},
	IQ1M:   {"iq1_m", 256, 56, true},
	BF16:   {"bf16", 1, 2, false},
	TQ1_0:  {"tq1_0", 256, 54, true},
	TQ2_0:  {"tq2_0", 256, 66, true},
	MXFP4:  {"mxfp4", 32, 17, true},
	NVFP4:  {"nvfp4", 64, 36, true},
	Q1_0:   {"q1_0", 128, 18, true},
	Q2_0:   {"q2_0", 64, 18, true},
	F8E4M3: {"f8_e4m3", 1, 1, false},
}

func (t Type) String() string {
	if value, ok := traits[t]; ok {
		return value.Name
	}
	return fmt.Sprintf("dtype_%d", t)
}

func (t Type) Traits() (Traits, bool) {
	value, ok := traits[t]
	return value, ok
}

// IsQuantized reports the physical block-layout class.
func (t Type) IsQuantized() bool {
	value, ok := traits[t]
	return ok && value.Quantized
}

// StorageBytes: physical byte size of a tensor of this type whose logical shape
// has `elements` elements laid out as rows of width `rowWidth` (dims[0]). This
// is the single owner of tensor storage-byte accounting shared by Shape.Bytes,
// the GGUF reader, and the GGUF writer. Block types occupy
// (elements/BlockSize)*TypeSize. F8E4M3 native residency additionally carries a
// per-output-row F32 scale after the packed e4m3 payload (elements*1 + rows*4),
// matching the resident matmul buffer layout [rows*inner e4m3 | rows*4 scale]
// the fp8 kernel expects. rowWidth must divide into whole blocks.
func (t Type) StorageBytes(elements, rowWidth uint64) (uint64, error) {
	traitsValue, ok := traits[t]
	if !ok || traitsValue.BlockSize == 0 || traitsValue.TypeSize == 0 {
		return 0, fmt.Errorf("unsupported type %d", uint32(t))
	}
	if rowWidth == 0 || rowWidth%traitsValue.BlockSize != 0 {
		return 0, fmt.Errorf(
			"row width %d is not divisible by %s block size %d",
			rowWidth, traitsValue.Name, traitsValue.BlockSize,
		)
	}
	if t == F8E4M3 {
		// packed e4m3 (1 byte/element) followed by one F32 scale per output row
		rows := elements / rowWidth
		if rows > (math.MaxUint64-elements)/4 {
			return 0, errors.New("fp8 tensor byte size overflows uint64")
		}
		return elements + rows*4, nil
	}
	blocks := elements / traitsValue.BlockSize
	if blocks > math.MaxUint64/traitsValue.TypeSize {
		return 0, errors.New("tensor byte size overflows uint64")
	}
	return blocks * traitsValue.TypeSize, nil
}
