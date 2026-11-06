package modelartifact

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestMeasureGGUFUsesBoundedDeterministicSamples(t *testing.T) {
	values := make([]byte, 512*4)
	for index := range 512 {
		binary.LittleEndian.PutUint32(values[index*4:], math.Float32bits(float32(index-256)))
	}
	opened := openGGUF(t, []gguf.TensorData{{
		Name: "weight", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(values),
	}})
	inventory := testcheck.Must(FromGGUF(opened, artifact.KindModel))(t)
	policy := MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1024}
	first := testcheck.Must(MeasureGGUF(inventory.TensorInventory, opened, policy))(t)
	second := testcheck.Must(MeasureGGUF(inventory.TensorInventory, opened, policy))(t)
	measurement := first.Measurements[0]
	if first.ID != second.ID || first.ReadBytes != 1024 || measurement.Samples != 256 ||
		measurement.Elements != 512 || float64(measurement.Samples)/float64(measurement.Elements) != 0.5 ||
		measurement.InterquartileRange <= 0 {
		t.Fatalf("GGUF measurement = %+v", first)
	}
	content := testcheck.Must(first.Content())(t)
	parsed, err := tensorMeasurementCodec.Parse(content.Data)
	if err != nil || parsed.ID != first.ID {
		t.Fatalf("measurement round trip = (%+v, %v)", parsed, err)
	}
}

func TestMeasureSafetensorsUsesElementReads(t *testing.T) {
	directory := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "config.json"), []byte(fixtureConfig), 0o600))
	header := `{"weight":{"dtype":"F32","shape":[512],"data_offsets":[0,2048]}}`
	data := make([]byte, 8, 8+len(header)+2048)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	for index := range 512 {
		var scalar [4]byte
		binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(float32(index)))
		data = append(data, scalar[:]...)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "model.safetensors"), data, 0o600))
	repository := testcheck.Must(hfrepo.Open(directory))(t)
	defer repository.Close()
	inventory := testcheck.Must(FromHFRepository(repository))(t)
	document := testcheck.Must(MeasureSafetensors(
		inventory.TensorInventory, repository.Tensors,
		MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1024},
	))(t)
	if document.ReadBytes != 1024 || math.Abs(document.Measurements[0].Median-255) > 1 {
		t.Fatalf("Safetensors measurement = %+v", document)
	}
}

func TestTensorMeasurementEnforcesReadBudget(t *testing.T) {
	policy := MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1}
	if _, err := newTensorMeasurementDocument(
		fixtureInventoryID(t), policy, 2,
		[]TensorMeasurement{{
			Name:             "weight",
			Characterization: tensorstats.Characterization{Elements: 1, Samples: 1, FiniteSamples: 1},
		}},
	); err == nil {
		t.Fatal("measurement beyond read budget accepted")
	}
}

func TestEvenlySpacedIndexAvoidsIntermediateOverflow(t *testing.T) {
	const sampleCount = uint64(4096)
	previous := uint64(0)
	for sample := range sampleCount {
		index := evenlySpacedIndex(sample, sampleCount, math.MaxUint64)
		if sample > 0 && index <= previous {
			t.Fatalf("sample %d index %d follows %d", sample, index, previous)
		}
		previous = index
	}
	if previous >= math.MaxUint64 {
		t.Fatalf("final sample index = %d", previous)
	}
}

func fixtureInventoryID(t *testing.T) artifact.ID {
	return testutil.ArtifactID(t, artifact.KindTensorInventory, "inventory")
}
