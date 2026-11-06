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
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestMeasureAtLocationGGUF(t *testing.T) {
	values := make([]float32, 512)
	for i := range values {
		values[i] = float32(i)
	}
	path := testutil.TempGGUF(t, "model.gguf", nil, []gguf.TensorData{
		{Name: "w", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(values))},
	})
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	inventory := testcheck.Must(FromGGUF(file, artifact.KindModel))(t)

	doc := testcheck.Must(MeasureAtLocation(inventory.TensorInventory, path, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20}))(t)
	if len(doc.Measurements) != 1 || doc.Measurements[0].Name != "w" {
		t.Fatalf("gguf measurements = %+v", doc.Measurements)
	}
}

func TestMeasureAtLocationSafetensors(t *testing.T) {
	dir := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte(fixtureConfig), 0o600))
	header := `{"weight":{"dtype":"F32","shape":[512],"data_offsets":[0,2048]}}`
	data := make([]byte, 8, 8+len(header)+2048)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	for i := range 512 {
		var scalar [4]byte
		binary.LittleEndian.PutUint32(scalar[:], math.Float32bits(float32(i)))
		data = append(data, scalar[:]...)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o600))
	repository := testcheck.Must(hfrepo.Open(dir))(t)
	defer repository.Close()
	inventory := testcheck.Must(FromHFRepository(repository))(t)

	// Location is the repository directory; MeasureAtLocation dispatches on the
	// inventory's safetensors format.
	doc := testcheck.Must(MeasureAtLocation(inventory.TensorInventory, dir, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20}))(t)
	if len(doc.Measurements) != 1 || doc.Measurements[0].Name != "weight" {
		t.Fatalf("safetensors measurements = %+v", doc.Measurements)
	}
}
