package modelartifact

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
)

func TestMeasureAtLocationGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	values := make([]byte, 512*4)
	for i := 0; i < 512; i++ {
		binary.LittleEndian.PutUint32(values[i*4:], math.Float32bits(float32(i)))
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(f, nil, []gguf.TensorData{
		{Name: "w", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(values)},
	}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	f.Close()
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := FromGGUF(file)
	if err != nil {
		t.Fatal(err)
	}

	doc, err := MeasureAtLocation(inventory.TensorInventory, path, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Measurements) != 1 || doc.Measurements[0].Name != "w" {
		t.Fatalf("gguf measurements = %+v", doc.Measurements)
	}
}

func TestMeasureAtLocationSafetensors(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	header := `{"weight":{"dtype":"F32","shape":[512],"data_offsets":[0,2048]}}`
	data := make([]byte, 8, 8+len(header)+2048)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	for i := 0; i < 512; i++ {
		var scalar [4]byte
		binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(float32(i)))
		data = append(data, scalar[:]...)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	repository, err := hfrepo.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	inventory, err := FromHFRepository(repository)
	if err != nil {
		t.Fatal(err)
	}

	// Location is the repository directory; MeasureAtLocation dispatches on the
	// inventory's safetensors format.
	doc, err := MeasureAtLocation(inventory.TensorInventory, dir, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Measurements) != 1 || doc.Measurements[0].Name != "weight" {
		t.Fatalf("safetensors measurements = %+v", doc.Measurements)
	}
}
