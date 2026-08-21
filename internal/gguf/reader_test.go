package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func TestParseFixture(t *testing.T) {
	data := buildFixture(t)
	file, err := Parse(bytes.NewReader(data), uint64(len(data)), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	if file.Version != 3 || file.Alignment != 32 {
		t.Fatalf("unexpected header: version=%d alignment=%d", file.Version, file.Alignment)
	}
	if len(file.Metadata) != 3 || len(file.Tensors) != 2 {
		t.Fatalf("unexpected counts: metadata=%d tensors=%d", len(file.Metadata), len(file.Tensors))
	}
	name, ok := file.MetadataValue("general.name")
	if !ok || name.Data != "fixture" {
		t.Fatalf("general.name = %#v, %v", name, ok)
	}
	tokens, ok := file.MetadataValue("tokenizer.ggml.tokens")
	if !ok || tokens.Count() != 2 {
		t.Fatalf("tokens = %#v, %v", tokens, ok)
	}
	quant, ok := file.Tensor("quant")
	if !ok || quant.Type != DTypeQ8_0 || quant.Size != 34 || quant.Offset != 32 {
		t.Fatalf("quant tensor = %#v, %v", quant, ok)
	}
	destination := make([]byte, quant.Size)
	if err := file.ReadTensorData(quant, destination); err != nil {
		t.Fatal(err)
	}
	for i, value := range destination {
		if value != 0x5a {
			t.Fatalf("tensor byte %d = %x, want 5a", i, value)
		}
	}
	rangeData := make([]byte, 5)
	if err := file.ReadTensorRange(quant, 7, rangeData); err != nil {
		t.Fatal(err)
	}
	for i, value := range rangeData {
		if value != 0x5a {
			t.Fatalf("range byte %d = %x, want 5a", i, value)
		}
	}
	if err := file.ReadTensorRange(quant, quant.Size-1, make([]byte, 2)); err == nil {
		t.Fatal("out-of-range tensor read was accepted")
	}
}

func TestParseRejectsMalformedData(t *testing.T) {
	fixture := buildFixture(t)
	tests := []struct {
		name string
		data []byte
		want string
	}{
		{"truncated", fixture[:20], "unexpected EOF"},
		{"bad magic", append([]byte("NOPE"), fixture[4:]...), "invalid GGUF magic"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(bytes.NewReader(tt.data), uint64(len(tt.data)), DefaultOptions())
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestAdversarialGGUFCatalogCountsFailBeforeAllocation(t *testing.T) {
	fixture := buildFixture(t)
	for name, offset := range map[string]int{"tensors": 8, "metadata": 16} {
		t.Run(name, func(t *testing.T) {
			data := append([]byte(nil), fixture...)
			binary.LittleEndian.PutUint64(data[offset:], math.MaxUint64)
			if _, err := Parse(bytes.NewReader(data), uint64(len(data)), DefaultOptions()); err == nil ||
				!strings.Contains(err.Error(), "exceeds limit") {
				t.Fatalf("count error = %v", err)
			}
		})
	}
}

func TestBoundedBoundaryPolicyContract(t *testing.T) {
	testBoundedBoundaryPolicy(t)
}

func TestParseEnforcesLimits(t *testing.T) {
	testBoundedBoundaryPolicy(t)
}

func testBoundedBoundaryPolicy(t *testing.T) {
	t.Helper()
	data := buildFixture(t)
	tests := map[string]func(*Options){
		"string bytes":   func(options *Options) { options.MaxStringBytes = 1 },
		"array elements": func(options *Options) { options.MaxArrayElements = 1 },
		"metadata count": func(options *Options) { options.MaxMetadata = 1 },
		"tensor count":   func(options *Options) { options.MaxTensors = 1 },
		"alignment":      func(options *Options) { options.MaxAlignment = 1 },
	}
	for name, constrain := range tests {
		t.Run(name, func(t *testing.T) {
			options := DefaultOptions()
			constrain(&options)
			if _, err := Parse(bytes.NewReader(data), uint64(len(data)), options); err == nil {
				t.Fatal("file-controlled allocation exceeded its configured bound")
			}
		})
	}
}

func FuzzParseNeverPanics(f *testing.F) {
	f.Add(buildFixture(f))
	f.Add([]byte("GGUF"))
	f.Fuzz(func(t *testing.T, data []byte) {
		options := DefaultOptions()
		options.MaxStringBytes = 1 << 20
		options.MaxArrayElements = 1 << 16
		options.MaxMetadata = 1 << 12
		options.MaxTensors = 1 << 12
		_, _ = Parse(bytes.NewReader(data), uint64(len(data)), options)
	})
}

type testingT interface {
	Helper()
	Fatalf(string, ...any)
}

func buildFixture(t testingT) []byte {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatalf("write fixture: %v", err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}

	_, _ = buffer.WriteString(Magic)
	write(uint32(3))
	write(uint64(2))
	write(uint64(3))

	writeString("general.alignment")
	write(uint32(ValueTypeUint32))
	write(uint32(32))

	writeString("general.name")
	write(uint32(ValueTypeString))
	writeString("fixture")

	writeString("tokenizer.ggml.tokens")
	write(uint32(ValueTypeArray))
	write(uint32(ValueTypeString))
	write(uint64(2))
	writeString("a")
	writeString("b")

	writeString("weight")
	write(uint32(1))
	write(uint64(4))
	write(uint32(DTypeF32))
	write(uint64(0))

	writeString("quant")
	write(uint32(1))
	write(uint64(32))
	write(uint32(DTypeQ8_0))
	write(uint64(32))

	for buffer.Len()%32 != 0 {
		_ = buffer.WriteByte(0)
	}
	_, _ = buffer.Write(bytes.Repeat([]byte{0x11}, 16))
	_, _ = buffer.Write(make([]byte, 16))
	_, _ = buffer.Write(bytes.Repeat([]byte{0x5a}, 34))
	_, _ = buffer.Write(make([]byte, 30))
	return buffer.Bytes()
}
