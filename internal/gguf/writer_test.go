package gguf

import (
	"bytes"
	"errors"
	"io"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestTensorInfoElementCount(t *testing.T) {
	const width, height = uint64(3), uint64(5)
	info := TensorInfo{Name: "weight", Dimensions: 2, Shape: [MaxDimensions]uint64{width, height}}
	if got, err := info.ElementCount(); err != nil || got != width*height {
		t.Fatalf("element count = %d, %v", got, err)
	}
	for _, shape := range [][MaxDimensions]uint64{{}, {math.MaxUint64, 2}} {
		info.Shape = shape
		if _, err := info.ElementCount(); err == nil {
			t.Fatalf("invalid shape accepted: %v", shape)
		}
	}
}

func TestWriteRoundTripsMetadataAndTensors(t *testing.T) {
	metadata := []Metadata{
		{Key: "u8", Value: Value{Type: ValueTypeUint8, Data: uint8(255)}},
		{Key: "i8", Value: Value{Type: ValueTypeInt8, Data: int8(-12)}},
		{Key: "u16", Value: Value{Type: ValueTypeUint16, Data: uint16(65530)}},
		{Key: "i16", Value: Value{Type: ValueTypeInt16, Data: int16(-1234)}},
		{Key: "u32", Value: Value{Type: ValueTypeUint32, Data: uint32(4_000_000_000)}},
		{Key: "i32", Value: Value{Type: ValueTypeInt32, Data: int32(-2_000_000_000)}},
		{Key: "f32", Value: Value{Type: ValueTypeFloat32, Data: float32(-1.25)}},
		{Key: "bool", Value: Value{Type: ValueTypeBool, Data: true}},
		{Key: "string", Value: Value{Type: ValueTypeString, Data: "hello\x00世界"}},
		{Key: "u64", Value: Value{Type: ValueTypeUint64, Data: uint64(math.MaxUint64 - 1)}},
		{Key: "i64", Value: Value{Type: ValueTypeInt64, Data: int64(math.MinInt64 + 1)}},
		{Key: "f64", Value: Value{Type: ValueTypeFloat64, Data: math.Pi}},
		{Key: "a_u8", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeUint8, Data: []uint8{0, 255}}},
		{Key: "a_i8", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeInt8, Data: []int8{-128, 127}}},
		{Key: "a_u16", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeUint16, Data: []uint16{0, 65535}}},
		{Key: "a_i16", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeInt16, Data: []int16{-32768, 32767}}},
		{Key: "a_u32", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeUint32, Data: []uint32{0, math.MaxUint32}}},
		{Key: "a_i32", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeInt32, Data: []int32{math.MinInt32, math.MaxInt32}}},
		{Key: "a_f32", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeFloat32, Data: []float32{-0, 1.5}}},
		{Key: "a_bool", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeBool, Data: []bool{true, false}}},
		{Key: "a_string", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeString, Data: []string{"", "x"}}},
		{Key: "a_u64", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeUint64, Data: []uint64{0, math.MaxUint64}}},
		{Key: "a_i64", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeInt64, Data: []int64{math.MinInt64, math.MaxInt64}}},
		{Key: "a_f64", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeFloat64, Data: []float64{math.SmallestNonzeroFloat64, math.Inf(1)}}},
	}
	f32Data := []byte{
		0, 0, 0x80, 0x3f,
		0, 0, 0, 0x40,
		0, 0, 0x40, 0x40,
		0, 0, 0x80, 0x40,
	}
	q4Data := make([]byte, 18)
	for index := range q4Data {
		q4Data[index] = byte(index*17 + 3)
	}
	var encoded bytes.Buffer
	err := Write(
		&encoded,
		metadata,
		[]TensorData{
			{Name: "f32.weight", Shape: []uint64{2, 2}, Type: DTypeF32, Data: bytes.NewReader(f32Data)},
			{Name: "q4.weight", Shape: []uint64{32}, Type: DTypeQ4_0, Data: bytes.NewReader(q4Data)},
		},
		WriteOptions{Alignment: 64},
	)
	if err != nil {
		t.Fatal(err)
	}

	file, err := Parse(bytes.NewReader(encoded.Bytes()), uint64(encoded.Len()), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	if file.Version != CurrentVersion || file.Alignment != 64 {
		t.Fatalf("version/alignment = %d/%d, want %d/64", file.Version, file.Alignment, CurrentVersion)
	}
	if file.DataOffset%64 != 0 || file.DataSize != 128 {
		t.Fatalf("data offset/size = %d/%d, want aligned/128", file.DataOffset, file.DataSize)
	}
	if len(file.Metadata) != len(metadata)+1 {
		t.Fatalf("metadata count = %d, want %d", len(file.Metadata), len(metadata)+1)
	}
	for _, item := range metadata {
		got, ok := file.MetadataValue(item.Key)
		if !ok {
			t.Fatalf("metadata %q is missing", item.Key)
		}
		if !reflect.DeepEqual(got, item.Value) {
			t.Fatalf("metadata %q = %#v, want %#v", item.Key, got, item.Value)
		}
	}
	alignment, ok := file.MetadataValue("general.alignment")
	if !ok || alignment.Type != ValueTypeUint32 || alignment.Data != uint32(64) {
		t.Fatalf("generated alignment metadata = %#v", alignment)
	}
	if len(file.Tensors) != 2 ||
		file.Tensors[0].Offset != 0 ||
		file.Tensors[1].Offset != 64 {
		t.Fatalf("tensor directory = %#v", file.Tensors)
	}
	for index, want := range [][]byte{f32Data, q4Data} {
		got := make([]byte, len(want))
		if err := file.ReadTensorData(file.Tensors[index], got); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("tensor %d data = %x, want %x", index, got, want)
		}
	}
}

func TestWriteEmptyFileRoundTrips(t *testing.T) {
	var encoded bytes.Buffer
	if err := Write(&encoded, nil, nil, WriteOptions{Version: 2}); err != nil {
		t.Fatal(err)
	}
	file, err := Parse(bytes.NewReader(encoded.Bytes()), uint64(encoded.Len()), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	if file.Version != 2 || file.DataOffset != uint64(encoded.Len()) ||
		len(file.Metadata) != 0 || len(file.Tensors) != 0 {
		t.Fatalf("empty file = %#v", file)
	}
}

func TestWriteRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name     string
		metadata []Metadata
		tensors  []TensorData
		options  WriteOptions
		want     string
	}{
		{
			name: "metadata data type",
			metadata: []Metadata{{
				Key:   "bad",
				Value: Value{Type: ValueTypeUint32, Data: int32(1)},
			}},
			want: "does not have type uint32",
		},
		{
			name: "duplicate metadata",
			metadata: []Metadata{
				{Key: "same", Value: Value{Type: ValueTypeBool, Data: true}},
				{Key: "same", Value: Value{Type: ValueTypeBool, Data: false}},
			},
			want: "duplicate metadata",
		},
		{
			name: "alignment conflict",
			metadata: []Metadata{{
				Key:   "general.alignment",
				Value: Value{Type: ValueTypeUint32, Data: uint32(32)},
			}},
			options: WriteOptions{Alignment: 64},
			want:    "write alignment is 64",
		},
		{
			name: "quant block alignment",
			tensors: []TensorData{{
				Name:  "bad",
				Shape: []uint64{31},
				Type:  DTypeQ4_0,
				Data:  bytes.NewReader(nil),
			}},
			want: "not divisible",
		},
		{
			name: "nil tensor data",
			tensors: []TensorData{{
				Name:  "bad",
				Shape: []uint64{1},
				Type:  DTypeF32,
			}},
			want: "data is nil",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := Write(io.Discard, test.metadata, test.tensors, test.options)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestWriteRejectsShortTensorData(t *testing.T) {
	err := Write(
		io.Discard,
		nil,
		[]TensorData{{
			Name:  "short",
			Shape: []uint64{2},
			Type:  DTypeF32,
			Data:  bytes.NewReader(make([]byte, 7)),
		}},
		WriteOptions{},
	)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("error = %v, want EOF", err)
	}
}

func TestFileWriteToMergesSplitSources(t *testing.T) {
	directory := t.TempDir()
	prefix := filepath.Join(directory, "source")
	firstPath := formatSplitPath(prefix, 0, 2)
	writeSplitFixture(t, firstPath, 0, 2, 3, []splitFixtureTensor{
		{name: "first", fill: 0x11},
		{name: "second", fill: 0x22},
	})
	writeSplitFixture(
		t,
		formatSplitPath(prefix, 1, 2),
		1,
		2,
		3,
		[]splitFixtureTensor{{name: "third", fill: 0x33}},
	)
	source, err := Open(firstPath)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	var encoded bytes.Buffer
	if err := source.WriteTo(&encoded, WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	merged, err := Parse(
		bytes.NewReader(encoded.Bytes()),
		uint64(encoded.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if merged.SplitCount != 1 || len(merged.Tensors) != 3 {
		t.Fatalf("merged split/tensors = %d/%d, want 1/3", merged.SplitCount, len(merged.Tensors))
	}
	for _, key := range []string{"split.no", "split.count", "split.tensors.count"} {
		if _, ok := merged.MetadataValue(key); ok {
			t.Fatalf("merged metadata still contains %q", key)
		}
	}
	for index, fill := range []byte{0x11, 0x22, 0x33} {
		got := make([]byte, merged.Tensors[index].Size)
		if err := merged.ReadTensorData(merged.Tensors[index], got); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, bytes.Repeat([]byte{fill}, len(got))) {
			t.Fatalf("merged tensor %d data = %x", index, got)
		}
	}
}
