package quant

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestFloat32ToFloat16(t *testing.T) {
	tests := []struct {
		value float32
		want  uint16
	}{
		{0, 0x0000},
		{float32(math.Copysign(0, -1)), 0x8000},
		{1, 0x3c00},
		{-2, 0xc000},
		{65504, 0x7bff},
		{float32(math.Ldexp(1, -24)), 0x0001},
		{float32(math.Ldexp(1, -25)), 0x0000},
		{float32(math.Inf(1)), 0x7c00},
		{float32(math.Inf(-1)), 0xfc00},
	}
	for _, test := range tests {
		if got := Float32ToFloat16(test.value); got != test.want {
			t.Errorf("Float32ToFloat16(%g) = %04x, want %04x",
				test.value, got, test.want)
		}
	}
	nan := Float32ToFloat16(float32(math.NaN()))
	if nan&0x7c00 != 0x7c00 || nan&0x03ff == 0 {
		t.Fatalf("NaN encoded as %04x", nan)
	}
}

func TestQuantizeMatchesPinnedGGMLReference(t *testing.T) {
	// hashes: produced by corresponding quantize_row_*_ref
	// exports in ggml-base.dll from llama.cpp commit
	// 42fc243060709331ff9b158a9ed2cbe37219ae83
	tests := []struct {
		dataType dtype.Type
		elements int
		wantHash string
	}{
		{dtype.Q1_0, 128, "76d753a4e7f8d2ac4a5932d4911ed850f7fd8491f46cb2b935465cf86f2e20c4"},
		{dtype.Q2_0, 64, "e76744a6e16c7c937b37b8e4e00e7e87b1d3f11367b758fd8bd4505c4b553ac2"},
		{dtype.Q4_0, 32, "d2954485591e755f49198554d19bc83df7c8a64ef635548f811c3785ab1a81a0"},
		{dtype.Q4_1, 32, "6bf5180c7c7f2395a0fed4a9844e13d52c826967614fc190ba5b58dffa5c3c57"},
		{dtype.Q5_0, 32, "563cdc7ac378b2942a2d5c4418de4d8f9aceb78b44e526f060ba6fe642b7905a"},
		{dtype.Q5_1, 32, "25ae45e5f4520ab387f5599d1a796adedad8d91711c2ef37455d4a39651b19a8"},
		{dtype.Q8_0, 32, "b8086271ad25adef2446c28ad6f45015e98f546d25b71d607ab373ade6580180"},
		{dtype.Q8_1, 32, "80f3020364a06c196070ef2e00f7462f523d102ad4bb9ddbf3383ecc918491cb"},
		{dtype.Q2K, 256, "d331b04f67c13a96d2eba8d23aaf75cc9a163ded1dcf7005a88e4e55ee5c9836"},
		{dtype.Q3K, 256, "913e744141c1a6f9c68c6de8473b6f4f29aa1524cd802ba6897d40ce7a287d6f"},
		{dtype.Q4K, 256, "21065fe371b120180dd7294440c9fb51f0f695e0f9fd4ee7fdc5fbaf3f9cf003"},
		{dtype.Q5K, 256, "a984053156c16221a9f8c2989ab2b09a7a5daab9783bb472891a65955a3f0b05"},
		{dtype.Q6K, 256, "465c89fa26c492e4acfbe64065027e77c5b8d900b486e4137127cbd3c7ef9830"},
		{dtype.Q8K, 256, "3cf3c379be9b03c3d9c4705bca8a463d8c1d3f3fce4edb0eab40f28f6c7e5dd6"},
		{dtype.TQ1_0, 256, "bbcf80b112ba9979d7d5817012da7a07e889f371617162ff96f6da8d8a82e25c"},
		{dtype.TQ2_0, 256, "a2fcdf8494a4d73e19abc82af162251c5918f0b66cd8664904b07cec39e4c4d2"},
		{dtype.MXFP4, 32, "d5863453bda709912a07524993680afa229631493ae70c27adc7686e60838def"},
		{dtype.NVFP4, 64, "f58566060757af09890b181312c1190734a07dc80f2d38205a3b46d8c6233a5e"},
		{dtype.IQ4NL, 32, "432c6f82bac0b7b18efb47c35ac92210c403836612458682174aa90de68dc16a"},
		{dtype.IQ4XS, 256, "db0d6a21cbbe3768a5bfb0e6de70ea2521517c0d06112600953253fb067b3410"},
		{dtype.IQ2S, 256, "8d4411e4beda018b81236ddb8f3122ac20f8c982b471b3a79d9cf28d0f42d2f3"},
		{dtype.IQ3XXS, 256, "c07d08d3be921e0a8b6d33cdcbe7131632cff8b7f7b7dbf05c871cb4f9d6ebfc"},
		{dtype.IQ3S, 256, "c79fb1e428c7fe92592b79092d89f0a3d70599854e7e14d9a35cc3f7f323351e"},
	}
	for _, test := range tests {
		t.Run(test.dataType.String(), func(t *testing.T) {
			input := quantizeOracleValues(test.elements)
			output, err := Quantize(test.dataType, input)
			if err != nil {
				t.Fatal(err)
			}
			if got := hashHex(output); got != test.wantHash {
				t.Fatalf("SHA256 = %s, want %s\nbytes = %x",
					got, test.wantHash, output)
			}
		})
	}
}

func TestQuantizeWeightedMatchesPinnedGGMLHashes(t *testing.T) {
	values := quantizeOracleValues(256)
	weights := make([]float32, len(values))
	for index := range weights {
		weights[index] = 0.25 + float32((index*29)%37)/11
	}
	for _, test := range []struct {
		dataType dtype.Type
		wantHash string
	}{
		{dtype.IQ2XXS, "2ed52c0109f6a017e95e9d17f65cabde3db251b8e261a90ae698a056a29bc25c"},
		{dtype.IQ2XS, "46cf4456f536442a4f919b5a35a056462aa89b5ff387f5b1a7df1fa277b98e42"},
		{dtype.IQ1S, "99c2478dab6e056c02b459562c0f0c4462662c7f2761d2d51322c24d1c2b070c"},
		{dtype.IQ1M, "c4b106d3d0afb73bc8a07960a220946c8c4f44a87039074d8be9dc6a510dc6db"},
	} {
		t.Run(test.dataType.String(), func(t *testing.T) {
			output, err := QuantizeWeighted(test.dataType, values, weights)
			if err != nil {
				t.Fatal(err)
			}
			if got := hashHex(output); got != test.wantHash {
				t.Fatalf("SHA256 = %s, want %s", got, test.wantHash)
			}
		})
	}
}

func TestQuantizeScalarStorage(t *testing.T) {
	values := []float32{-2.5, 0, 1.25, float32(math.Inf(1))}
	f32, err := Quantize(dtype.F32, values)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range values {
		if got := binary.LittleEndian.Uint32(f32[index*4:]); got != math.Float32bits(value) {
			t.Fatalf("F32[%d] = %08x, want %08x",
				index, got, math.Float32bits(value))
		}
	}

	f16, err := Quantize(dtype.F16, values)
	if err != nil {
		t.Fatal(err)
	}
	wantF16 := []uint16{0xc100, 0x0000, 0x3d00, 0x7c00}
	for index, want := range wantF16 {
		if got := binary.LittleEndian.Uint16(f16[index*2:]); got != want {
			t.Fatalf("F16[%d] = %04x, want %04x", index, got, want)
		}
	}

	bf16, err := Quantize(dtype.BF16, values)
	if err != nil {
		t.Fatal(err)
	}
	wantBF16 := []uint16{0xc020, 0x0000, 0x3fa0, 0x7f80}
	for index, want := range wantBF16 {
		if got := binary.LittleEndian.Uint16(bf16[index*2:]); got != want {
			t.Fatalf("BF16[%d] = %04x, want %04x", index, got, want)
		}
	}
}

func TestQuantizeErrors(t *testing.T) {
	if _, err := Quantize(dtype.Q4_0, make([]float32, 31)); err == nil {
		t.Fatal("expected block alignment error")
	}
	values := make([]float32, 32)
	values[17] = float32(math.NaN())
	if _, err := Quantize(dtype.Q8_0, values); err == nil {
		t.Fatal("expected non-finite input error")
	}
	if _, err := Quantize(dtype.IQ2XXS, make([]float32, 256)); err == nil {
		t.Fatal("expected missing importance error")
	}
	if _, err := QuantizeWeighted(dtype.IQ2XXS, make([]float32, 256), make([]float32, 255)); err == nil {
		t.Fatal("expected importance length error")
	}
	weights := make([]float32, 256)
	weights[7] = -1
	if _, err := QuantizeWeighted(dtype.IQ2XXS, make([]float32, 256), weights); err == nil {
		t.Fatal("expected negative importance error")
	}
}

func TestQuantizeDequantize(t *testing.T) {
	for _, dataType := range Types() {
		traits, _ := dataType.Traits()
		input := quantizeOracleValues(int(traits.BlockSize))
		var encoded []byte
		var err error
		if RequiresImportance(dataType) {
			weights := make([]float32, len(input))
			for index := range weights {
				weights[index] = 1
			}
			encoded, err = QuantizeWeighted(dataType, input, weights)
		} else {
			encoded, err = Quantize(dataType, input)
		}
		if err != nil {
			t.Fatalf("%s encode: %v", dataType, err)
		}
		decoded, err := Dequantize(dataType, encoded, uint64(len(input)))
		if err != nil {
			t.Fatalf("%s decode: %v", dataType, err)
		}
		if len(decoded) != len(input) || slices.ContainsFunc(decoded, func(value float32) bool {
			return math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)
		}) {
			t.Fatalf("%s produced invalid decoded values", dataType)
		}
	}
}

func quantizeOracleValues(count int) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(
			float64(((index*37)%101)-50)*0.1375 +
				float64(index%3)*0.03125,
		)
	}
	return values
}

func hashHex(data []byte) string {
	return fmtHash(sha256.Sum256(data))
}

func fmtHash(value [sha256.Size]byte) string {
	const digits = "0123456789abcdef"
	result := make([]byte, len(value)*2)
	for index, item := range value {
		result[index*2] = digits[item>>4]
		result[index*2+1] = digits[item&0xf]
	}
	return string(result)
}
