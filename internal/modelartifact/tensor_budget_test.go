package modelartifact

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMeasurementChargesReadBudget pins the audit's evidence-integrity P1:
// every byte the measurement reads -- sampled or spectral, either format --
// charges the recorded budget through the single read meter. Spectral
// collection can no longer read a full tensor for free, a budget that covers
// sampling but not the spectral read defers the spectrum without reading it, and
// a nonzero spectral policy refuses documents whose measurements carry no
// spectral verdict.
func TestMeasurementChargesReadBudget(t *testing.T) {
	values := make([]float32, 16)
	for i := range values {
		values[i] = float32(i%4) + 1
	}
	// One fixture serves every measurement through a counted parse of its bytes; the inventory, which names the
	// file's physical paths, comes from the file opened where it lies.
	path := testutil.TempGGUF(t, "counted.gguf", nil, []gguf.TensorData{
		{Name: "matrix", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(values))},
	})
	opened := testcheck.Must(gguf.Open(path))(t)
	defer opened.Close()
	inventory := testcheck.Must(FromGGUF(opened, artifact.KindModel))(t)
	encoded := testcheck.Must(os.ReadFile(path))(t)
	source := &countedReader{source: bytes.NewReader(encoded)}
	file := testcheck.Must(gguf.Parse(source, uint64(len(encoded)), gguf.DefaultOptions()))(t)

	// Sampling alone: 4 samples x 4 bytes = 16 bytes recorded.
	sampled := testcheck.Must(MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: 1 << 20,
	}))(t)
	// Sampling plus spectral: the full 64-byte tensor read must be charged on
	// top of the sampled bytes, so recorded ReadBytes strictly exceeds the
	// sampling-only document's.
	spectral := testcheck.Must(MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: 1 << 20, SpectralMaxDim: 8,
	}))(t)
	if spectral.Measurements[0].SpectralStatus != SpectralComputed {
		t.Fatalf("spectral status = %q, want computed", spectral.Measurements[0].SpectralStatus)
	}
	if spectral.ReadBytes < sampled.ReadBytes+64 {
		t.Fatalf("spectral read not charged: sampled=%d spectral=%d", sampled.ReadBytes, spectral.ReadBytes)
	}

	// A budget that fits sampling but not the full spectral read defers the
	// spectrum without reading it: recorded bytes are exactly the sampled ones.
	// (A refusal here once failed a served model's whole pass after minutes.)
	deferred, err := MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: sampled.ReadBytes + 8, SpectralMaxDim: 8,
	})
	if err != nil || deferred.Measurements[0].SpectralStatus != SpectralDeferred || deferred.ReadBytes != sampled.ReadBytes {
		t.Fatalf("under-budget spectral read: status=%q read=%d sampled=%d err=%v", deferred.Measurements[0].SpectralStatus, deferred.ReadBytes, sampled.ReadBytes, err)
	}
	// Sampling itself still refuses a budget it cannot fit, before it reads a byte of the tensor.
	source.reads = 0
	if _, err := MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: sampled.ReadBytes - 1,
	}); err == nil || !strings.Contains(err.Error(), "read budget") || source.reads != 0 {
		t.Fatalf("under-budget sampled read: %v after %d tensor reads, want a refusal before any", err, source.reads)
	}

	// A nonzero spectral policy refuses measurements without a verdict.
	bare := spectral.Measurements
	bare[0].SpectralStatus = ""
	bare[0].EffectiveRank = 0
	if _, err := newTensorMeasurementDocument(spectral.Inventory, spectral.Policy, spectral.ReadBytes, bare); err == nil {
		t.Fatal("empty spectral status accepted under a nonzero spectral policy")
	}
}

// countedReader counts the reads made of a parsed GGUF's bytes.
type countedReader struct {
	source io.ReaderAt
	reads  int
}

func (r *countedReader) ReadAt(buffer []byte, offset int64) (int, error) {
	r.reads++
	return r.source.ReadAt(buffer, offset)
}
