package quant

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"testing"

	"llamacpp2go/internal/tensor/dtype"
)

func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, float32(math.Copysign(0, -1))},
		{0x3c00, 1},
		{0xc000, -2},
		{0x7bff, 65504},
		{0x0001, float32(math.Ldexp(1, -24))},
	}
	for _, tt := range tests {
		got := Float16ToFloat32(tt.bits)
		if math.Float32bits(got) != math.Float32bits(tt.want) {
			t.Errorf("Float16ToFloat32(%04x) = %g (%08x), want %g (%08x)",
				tt.bits, got, math.Float32bits(got), tt.want, math.Float32bits(tt.want))
		}
	}
	if !math.IsInf(float64(Float16ToFloat32(0x7c00)), 1) {
		t.Fatal("positive infinity was not preserved")
	}
	if !math.IsNaN(float64(Float16ToFloat32(0x7e00))) {
		t.Fatal("NaN was not preserved")
	}
}

func TestDequantizeScalarStorageTypes(t *testing.T) {
	tests := []struct {
		name     string
		dataType dtype.Type
		source   []byte
		want     []float32
	}{
		{
			name:     "i8",
			dataType: dtype.I8,
			source:   []byte{0x80, 0xff, 0x00, 0x7f},
			want:     []float32{-128, -1, 0, 127},
		},
		{
			name:     "i16",
			dataType: dtype.I16,
			source:   littleEndianValues(uint16(0x8000), uint16(0xffff), uint16(0x7fff)),
			want:     []float32{-32768, -1, 32767},
		},
		{
			name:     "i32",
			dataType: dtype.I32,
			source:   littleEndianValues(uint32(0x80000000), uint32(0xffffffff), uint32(123456)),
			want:     []float32{-2147483648, -1, 123456},
		},
		{
			name:     "i64",
			dataType: dtype.I64,
			source:   littleEndianValues(uint64(0x8000000000000000), uint64(0xffffffffffffffff), uint64(123456)),
			want:     []float32{-9223372036854775808, -1, 123456},
		},
		{
			name:     "f64",
			dataType: dtype.F64,
			source: littleEndianValues(
				math.Float64bits(-1.5),
				math.Float64bits(math.Inf(1)),
				math.Float64bits(math.SmallestNonzeroFloat64),
			),
			want: []float32{-1.5, float32(math.Inf(1)), 0},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			output, err := Dequantize(
				test.dataType,
				test.source,
				uint64(len(test.want)),
			)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(output, test.want) {
				t.Fatalf("output = %v, want %v", output, test.want)
			}
		})
	}
}

func littleEndianValues[T ~uint16 | ~uint32 | ~uint64](values ...T) []byte {
	var width int
	var zero T
	switch any(zero).(type) {
	case uint16:
		width = 2
	case uint32:
		width = 4
	case uint64:
		width = 8
	}
	result := make([]byte, len(values)*width)
	for index, value := range values {
		switch width {
		case 2:
			binary.LittleEndian.PutUint16(result[index*width:], uint16(value))
		case 4:
			binary.LittleEndian.PutUint32(result[index*width:], uint32(value))
		case 8:
			binary.LittleEndian.PutUint64(result[index*width:], uint64(value))
		}
	}
	return result
}

func TestDequantizeQ8_0(t *testing.T) {
	source := make([]byte, 34)
	binary.LittleEndian.PutUint16(source, 0x3800) // 0.5
	for index := 0; index < 32; index++ {
		source[2+index] = byte(int8(index - 16))
	}
	output, err := Dequantize(dtype.Q8_0, source, 32)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range output {
		want := float32(index-16) * 0.5
		if value != want {
			t.Fatalf("output[%d] = %v, want %v", index, value, want)
		}
	}
}

func TestDequantizeRejectsWrongSize(t *testing.T) {
	if _, err := Dequantize(dtype.Q8_0, make([]byte, 33), 32); err == nil {
		t.Fatal("short Q8_0 block was accepted")
	}
}

func TestDequantizeQ4(t *testing.T) {
	q40 := make([]byte, 18)
	binary.LittleEndian.PutUint16(q40, 0x3c00) // 1
	for index := 0; index < 16; index++ {
		q40[2+index] = byte(index) | byte(15-index)<<4
	}
	output, err := Dequantize(dtype.Q4_0, q40, 32)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 16; index++ {
		if output[index] != float32(index-8) {
			t.Fatalf("low output[%d] = %v", index, output[index])
		}
		if output[index+16] != float32(7-index) {
			t.Fatalf("high output[%d] = %v", index, output[index+16])
		}
	}

	q41 := make([]byte, 20)
	binary.LittleEndian.PutUint16(q41, 0x3800)     // 0.5
	binary.LittleEndian.PutUint16(q41[2:], 0xc000) // -2
	copy(q41[4:], q40[2:])
	output, err = Dequantize(dtype.Q4_1, q41, 32)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -2 || output[15] != 5.5 || output[16] != 5.5 {
		t.Fatalf("unexpected Q4_1 values: %v %v %v", output[0], output[15], output[16])
	}
}

func TestDequantizeQ8_1(t *testing.T) {
	source := make([]byte, 36)
	binary.LittleEndian.PutUint16(source, 0x3800) // 0.5
	binary.LittleEndian.PutUint16(source[2:], 0x4200)
	for index := 0; index < 32; index++ {
		source[4+index] = byte(int8(index - 16))
	}
	output, err := Dequantize(dtype.Q8_1, source, 32)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -8 || output[16] != 0 || output[31] != 7.5 {
		t.Fatalf("Q8_1 values = %v/%v/%v, want -8/0/7.5", output[0], output[16], output[31])
	}
}

func TestDequantizeQ8K(t *testing.T) {
	source := make([]byte, 292)
	binary.LittleEndian.PutUint32(source, math.Float32bits(0.25))
	for index := 0; index < 256; index++ {
		source[4+index] = byte(int8(index%127 - 63))
	}
	output, err := Dequantize(dtype.Q8K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -15.75 || output[63] != 0 || output[126] != 15.75 {
		t.Fatalf("Q8_K values = %v/%v/%v", output[0], output[63], output[126])
	}
}

func TestDequantizeQ1AndQ2(t *testing.T) {
	q1 := make([]byte, 18)
	binary.LittleEndian.PutUint16(q1, 0x4000) // 2
	q1[2] = 0b00000101
	output, err := Dequantize(dtype.Q1_0, q1, 128)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, -2, 2, -2}
	for index := range want {
		if output[index] != want[index] {
			t.Fatalf("Q1 output[%d] = %v, want %v", index, output[index], want[index])
		}
	}

	q2 := make([]byte, 18)
	binary.LittleEndian.PutUint16(q2, 0x3c00) // 1
	q2[2] = 0b11100100
	output, err = Dequantize(dtype.Q2_0, q2, 64)
	if err != nil {
		t.Fatal(err)
	}
	want = []float32{-1, 0, 1, 2}
	for index := range want {
		if output[index] != want[index] {
			t.Fatalf("Q2 output[%d] = %v, want %v", index, output[index], want[index])
		}
	}
}

func TestDequantizeQ6K(t *testing.T) {
	source := make([]byte, 210)
	source[0] = 0x0f
	source[128] = 0xe4
	for index := 0; index < 16; index++ {
		source[192+index] = 1
	}
	binary.LittleEndian.PutUint16(source[208:], 0x3c00) // 1
	output, err := Dequantize(dtype.Q6K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range map[int]float32{
		0:  -17,
		32: -16,
		64: 0,
		96: 16,
	} {
		if output[index] != want {
			t.Fatalf("Q6_K output[%d] = %v, want %v", index, output[index], want)
		}
	}
	if output[1] != -32 || output[255] != -32 {
		t.Fatalf("Q6_K untouched values = %v/%v, want -32/-32", output[1], output[255])
	}
}

func TestDequantizeQ6KScaleGroups(t *testing.T) {
	source := make([]byte, 210)
	for index := 0; index < 16; index++ {
		source[192+index] = byte(int8(index + 1))
	}
	binary.LittleEndian.PutUint16(source[208:], 0x3800) // 0.5
	output, err := Dequantize(dtype.Q6K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range map[int]float32{
		0:   -16,
		16:  -32,
		32:  -48,
		64:  -80,
		96:  -112,
		128: -144,
		255: -256,
	} {
		if output[index] != want {
			t.Fatalf("Q6_K scale output[%d] = %v, want %v", index, output[index], want)
		}
	}
}

func TestDequantizeQ2K(t *testing.T) {
	source := make([]byte, 84)
	for index := 0; index < 16; index++ {
		source[index] = 0x21
	}
	source[16] = 0x03
	binary.LittleEndian.PutUint16(source[80:], 0x3c00)
	binary.LittleEndian.PutUint16(source[82:], 0x3c00)
	output, err := Dequantize(dtype.Q2K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 1 || output[1] != -2 || output[255] != -2 {
		t.Fatalf("Q2_K values = %v/%v/%v, want 1/-2/-2", output[0], output[1], output[255])
	}
}

func TestDequantizeQ3K(t *testing.T) {
	source := make([]byte, 110)
	for index := 0; index < 8; index++ {
		source[96+index] = 0x11
	}
	for index := 0; index < 4; index++ {
		source[104+index] = 0xaa
	}
	source[0] = 0x01
	binary.LittleEndian.PutUint16(source[108:], 0x3c00)
	output, err := Dequantize(dtype.Q3K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 0 || output[1] != -4 || output[255] != -4 {
		t.Fatalf("Q3_K values = %v/%v/%v, want 0/-4/-4", output[0], output[1], output[255])
	}
}

func TestDequantizeQ4K(t *testing.T) {
	source := make([]byte, 144)
	binary.LittleEndian.PutUint16(source[0:], 0x3c00)
	binary.LittleEndian.PutUint16(source[2:], 0x3c00)
	source[4] = 1
	source[8] = 2
	source[16] = 3
	output, err := Dequantize(dtype.Q4K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 1 || output[1] != -2 || output[32] != 0 {
		t.Fatalf("Q4_K values = %v/%v/%v, want 1/-2/0", output[0], output[1], output[32])
	}
}

func TestDequantizeQ5K(t *testing.T) {
	source := make([]byte, 176)
	binary.LittleEndian.PutUint16(source[0:], 0x3c00)
	binary.LittleEndian.PutUint16(source[2:], 0x3c00)
	source[4] = 1
	source[8] = 2
	source[16] = 1
	source[48] = 3
	output, err := Dequantize(dtype.Q5K, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 17 || output[1] != -2 || output[32] != 0 {
		t.Fatalf("Q5_K values = %v/%v/%v, want 17/-2/0", output[0], output[1], output[32])
	}
}

func TestDequantizeIQ4XS(t *testing.T) {
	source := make([]byte, 136)
	binary.LittleEndian.PutUint16(source[0:], 0x3800) // 0.5
	// Group 0 scale: low 1 plus high 2 << 4 = 33, signed scale = 1.
	binary.LittleEndian.PutUint16(source[2:], 0x0002)
	source[4] = 0x01
	source[8] = 0xf0
	output, err := Dequantize(dtype.IQ4XS, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -63.5 || output[16] != 56.5 || output[32] != 2032 {
		t.Fatalf("IQ4_XS values = %v/%v/%v, want -63.5/56.5/2032", output[0], output[16], output[32])
	}
}

func TestDequantizeIQ4NL(t *testing.T) {
	source := make([]byte, 18)
	binary.LittleEndian.PutUint16(source[0:], 0x3800) // 0.5
	source[2] = 0xf0
	output, err := Dequantize(dtype.IQ4NL, source, 32)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -63.5 || output[16] != 56.5 ||
		output[15] != -63.5 || output[31] != -63.5 {
		t.Fatalf(
			"IQ4_NL values = %v/%v/%v/%v, want -63.5/56.5/-63.5/-63.5",
			output[0],
			output[16],
			output[15],
			output[31],
		)
	}
}

func TestDequantizeTQ2_0(t *testing.T) {
	source := make([]byte, 66)
	source[0] = 0xe4
	binary.LittleEndian.PutUint16(source[64:], 0x3800) // 0.5
	output, err := Dequantize(dtype.TQ2_0, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -0.5 || output[32] != 0 ||
		output[64] != 0.5 || output[96] != 1 ||
		output[128] != -0.5 {
		t.Fatalf(
			"TQ2_0 values = %v/%v/%v/%v/%v, want -0.5/0/0.5/1/-0.5",
			output[0],
			output[32],
			output[64],
			output[96],
			output[128],
		)
	}
}

func TestDequantizeTQ1_0(t *testing.T) {
	source := make([]byte, 54)
	source[0] = 0
	source[1] = 86
	source[2] = 171
	source[32] = 171
	source[48] = 86
	binary.LittleEndian.PutUint16(source[52:], 0x3800) // 0.5
	output, err := Dequantize(dtype.TQ1_0, source, 256)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != -0.5 || output[1] != 0 || output[2] != 0.5 ||
		output[32] != -0.5 || output[160] != 0.5 ||
		output[240] != 0 || output[244] != -0.5 {
		t.Fatalf(
			"TQ1_0 values = %v/%v/%v/%v/%v/%v/%v, want -0.5/0/0.5/-0.5/0.5/0/-0.5",
			output[0],
			output[1],
			output[2],
			output[32],
			output[160],
			output[240],
			output[244],
		)
	}
}

func TestDequantizeMXFP4(t *testing.T) {
	source := make([]byte, 17)
	source[0] = 128 // E8M0 half-scale = 1.
	source[1] = 0xf1
	output, err := Dequantize(dtype.MXFP4, source, 32)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 1 || output[16] != -12 ||
		output[1] != 0 || output[17] != 0 {
		t.Fatalf(
			"MXFP4 values = %v/%v/%v/%v, want 1/-12/0/0",
			output[0],
			output[16],
			output[1],
			output[17],
		)
	}
	source[0] = 0
	output, err = Dequantize(dtype.MXFP4, source, 32)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != math.Float32frombits(0x00200000) {
		t.Fatalf("MXFP4 denormal scale value = %v", output[0])
	}
}

func TestDequantizeNVFP4(t *testing.T) {
	source := make([]byte, 36)
	source[0] = 64 // UE4M3 half-scale = 1.
	source[1] = 65 // UE4M3 half-scale = 1.125.
	source[2] = 0
	source[3] = 0x7f
	source[4] = 0xf1
	source[12] = 0x25
	output, err := Dequantize(dtype.NVFP4, source, 64)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 1 || output[8] != -12 ||
		output[16] != 6.75 || output[24] != 2.25 ||
		output[32] != 0 || output[48] != 0 {
		t.Fatalf(
			"NVFP4 values = %v/%v/%v/%v/%v/%v, want 1/-12/6.75/2.25/0/0",
			output[0],
			output[8],
			output[16],
			output[24],
			output[32],
			output[48],
		)
	}
}

func TestDequantizeIQCodebooksMatchPinnedLlamaCPP(t *testing.T) {
	tests := []struct {
		name         string
		dataType     dtype.Type
		typeSize     int
		leadingScale bool
		iq1MScale    bool
		wantSHA256   string
	}{
		{
			name:         "iq2_xxs",
			dataType:     dtype.IQ2XXS,
			typeSize:     66,
			leadingScale: true,
			wantSHA256:   "12f023709353de5345b42e8c9d0e64dbb19375f637a05ca6ca49f13502a0e497",
		},
		{
			name:         "iq2_xs",
			dataType:     dtype.IQ2XS,
			typeSize:     74,
			leadingScale: true,
			wantSHA256:   "9c3afe6567abb1bffbd3525c45ec5571ccca6529c86f997143de93406eb4a5a6",
		},
		{
			name:         "iq2_s",
			dataType:     dtype.IQ2S,
			typeSize:     82,
			leadingScale: true,
			wantSHA256:   "2ba9b952bb4b37d42e94ea25b9bd0a8bff2e30f80be35217150e8ffa73c43117",
		},
		{
			name:         "iq3_xxs",
			dataType:     dtype.IQ3XXS,
			typeSize:     98,
			leadingScale: true,
			wantSHA256:   "f6443744ee01e2a6a63863f98828c01e59690805dbe499ff2cbb603e01242bab",
		},
		{
			name:         "iq3_s",
			dataType:     dtype.IQ3S,
			typeSize:     110,
			leadingScale: true,
			wantSHA256:   "6b26378b630b46eb4eb9d60ad9860bb539525fa5377cf1b734fd0b6843d7f55e",
		},
		{
			name:         "iq1_s",
			dataType:     dtype.IQ1S,
			typeSize:     50,
			leadingScale: true,
			wantSHA256:   "9279f05ac5794989a3e5885516daadc2b0c5265255c8176cc0788932108f59bb",
		},
		{
			name:       "iq1_m",
			dataType:   dtype.IQ1M,
			typeSize:   56,
			iq1MScale:  true,
			wantSHA256: "17ebeaee301a7280c26072c4ab0e049e689c12e3b8e58d69622da98f639c5296",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := make([]byte, test.typeSize)
			for index := range source {
				source[index] = byte(index*73 + 19)
			}
			if test.leadingScale {
				source[0] = 0
				source[1] = 0x3a // IEEE binary16 0.75.
			}
			if test.iq1MScale {
				// IQ1_M stores the four nibbles of its binary16 scale in the
				// high nibbles of four packed uint16 scale words.
				source[49] &= 0x0f
				source[51] = source[51]&0x0f | 0xa0
				source[53] = source[53]&0x0f | 0x30
				source[55] &= 0x0f
			}
			output, err := Dequantize(test.dataType, source, 256)
			if err != nil {
				t.Fatal(err)
			}
			encoded := make([]byte, len(output)*4)
			for index, value := range output {
				binary.LittleEndian.PutUint32(
					encoded[index*4:],
					math.Float32bits(value),
				)
			}
			got := fmt.Sprintf("%x", sha256.Sum256(encoded))
			if got != test.wantSHA256 {
				t.Fatalf(
					"output SHA-256 = %s, want pinned llama.cpp oracle %s",
					got,
					test.wantSHA256,
				)
			}
		})
	}
}
