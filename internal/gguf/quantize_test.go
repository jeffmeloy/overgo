package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/quant"
)

func TestQuantizeToConvertsMatricesAndPreservesOtherTensors(t *testing.T) {
	matrixValues := quantizeFixtureValues(64)
	vectorValues := quantizeFixtureValues(32)
	matrixData := float32Bytes(matrixValues)
	vectorData := float32Bytes(vectorValues)
	alreadyQuantized, err := quant.Quantize(DTypeQ4_0, vectorValues)
	if err != nil {
		t.Fatal(err)
	}
	var sourceData bytes.Buffer
	err = Write(
		&sourceData,
		[]Metadata{
			{
				Key:   "general.file_type",
				Value: Value{Type: ValueTypeUint32, Data: uint32(0)},
			},
			{
				Key:   "split.no",
				Value: Value{Type: ValueTypeUint16, Data: uint16(0)},
			},
		},
		[]TensorData{
			{
				Name:  "matrix.weight",
				Shape: []uint64{32, 2},
				Type:  DTypeF32,
				Data:  bytes.NewReader(matrixData),
			},
			{
				Name:  "matrix.bias",
				Shape: []uint64{32},
				Type:  DTypeF32,
				Data:  bytes.NewReader(vectorData),
			},
			{
				Name:  "already.weight",
				Shape: []uint64{32, 1},
				Type:  DTypeQ4_0,
				Data:  bytes.NewReader(alreadyQuantized),
			},
		},
		WriteOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	source, err := Parse(
		bytes.NewReader(sourceData.Bytes()),
		uint64(sourceData.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}

	var outputData bytes.Buffer
	report, err := source.QuantizeTo(
		&outputData,
		DTypeQ4_0,
		QuantizeOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if report.Converted != 1 || report.Preserved != 2 ||
		report.InputBytes != 402 || report.OutputBytes != 182 {
		t.Fatalf("report = %#v", report)
	}
	output, err := Parse(
		bytes.NewReader(outputData.Bytes()),
		uint64(outputData.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	fileType, ok := output.MetadataValue("general.file_type")
	if !ok || fileType.Type != ValueTypeUint32 || fileType.Data != uint32(2) {
		t.Fatalf("general.file_type = %#v, present=%v", fileType, ok)
	}
	version, ok := output.MetadataValue("general.quantization_version")
	if !ok || version.Data != uint32(quantizationVersion) {
		t.Fatalf("general.quantization_version = %#v, present=%v", version, ok)
	}
	if _, ok := output.MetadataValue("split.no"); ok {
		t.Fatal("split metadata was retained")
	}

	wantMatrix, err := quant.Quantize(DTypeQ4_0, matrixValues)
	if err != nil {
		t.Fatal(err)
	}
	assertTensorData(t, output, "matrix.weight", DTypeQ4_0, wantMatrix)
	assertTensorData(t, output, "matrix.bias", DTypeF32, vectorData)
	assertTensorData(t, output, "already.weight", DTypeQ4_0, alreadyQuantized)
}

func TestQuantizeToStreamsRequantization(t *testing.T) {
	const elements = 256 * 2049
	values := quantizeFixtureValues(elements)
	sourceTensor, err := quant.Quantize(DTypeQ8_0, values)
	if err != nil {
		t.Fatal(err)
	}
	var sourceData bytes.Buffer
	if err := Write(
		&sourceData,
		nil,
		[]TensorData{{
			Name:  "large.weight",
			Shape: []uint64{256, 2049},
			Type:  DTypeQ8_0,
			Data:  bytes.NewReader(sourceTensor),
		}},
		WriteOptions{},
	); err != nil {
		t.Fatal(err)
	}
	source, err := Parse(
		bytes.NewReader(sourceData.Bytes()),
		uint64(sourceData.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}

	var outputData bytes.Buffer
	report, err := source.QuantizeTo(
		&outputData,
		DTypeQ5_1,
		QuantizeOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if report.Converted != 1 {
		t.Fatalf("report = %#v", report)
	}
	output, err := Parse(
		bytes.NewReader(outputData.Bytes()),
		uint64(outputData.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(
		DTypeQ8_0,
		sourceTensor,
		elements,
	)
	if err != nil {
		t.Fatal(err)
	}
	want, err := quant.Quantize(DTypeQ5_1, dequantized)
	if err != nil {
		t.Fatal(err)
	}
	assertTensorData(t, output, "large.weight", DTypeQ5_1, want)
}

func TestQuantizeToRejectsUnsupportedOrForcedMisalignedTargets(t *testing.T) {
	values := quantizeFixtureValues(62)
	var encoded bytes.Buffer
	if err := Write(
		&encoded,
		nil,
		[]TensorData{{
			Name:  "odd.weight",
			Shape: []uint64{31, 2},
			Type:  DTypeF32,
			Data:  bytes.NewReader(float32Bytes(values)),
		}},
		WriteOptions{},
	); err != nil {
		t.Fatal(err)
	}
	source, err := Parse(
		bytes.NewReader(encoded.Bytes()),
		uint64(encoded.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := source.QuantizeTo(
		&bytes.Buffer{},
		DTypeIQ2XXS,
		QuantizeOptions{},
	); err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("unsupported target error = %v", err)
	}
	if _, err := source.QuantizeTo(
		&bytes.Buffer{},
		DTypeQ4_0,
		QuantizeOptions{
			ShouldQuantize: func(TensorInfo) bool { return true },
		},
	); err == nil || !strings.Contains(err.Error(), "row size") {
		t.Fatalf("misaligned target error = %v", err)
	}
}

func quantizeFixtureValues(count int) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(math.Sin(float64(index)*0.071)*4.5) +
			float32((index%11)-5)*0.015625
	}
	return values
}

func float32Bytes(values []float32) []byte {
	result := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(result[index*4:], math.Float32bits(value))
	}
	return result
}

func assertTensorData(
	t *testing.T,
	file *File,
	name string,
	dataType DType,
	want []byte,
) {
	t.Helper()
	tensor, ok := file.Tensor(name)
	if !ok {
		t.Fatalf("tensor %q is missing", name)
	}
	if tensor.Type != dataType {
		t.Fatalf("tensor %q type = %s, want %s", name, tensor.Type, dataType)
	}
	got := make([]byte, tensor.Size)
	if err := file.ReadTensorData(tensor, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("tensor %q data mismatch", name)
	}
}
