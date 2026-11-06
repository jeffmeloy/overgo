package server

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestTensorMeasurementLookupStopsAtItsMatch finds a model's committed
// measurement without reading the measurements behind it: the lookup read
// and decoded every stored measurement, each a whole document, after the
// newest one had already matched. An older document that cannot be decoded
// stands behind the match here, so a lookup that reads past it fails.
func TestTensorMeasurementLookupStopsAtItsMatch(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	older := []byte(`{"schema":"not a measurement"}`)
	olderID := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorInventory, older))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "server/tensor-lookup-older", Contents: []artifact.Content{{
		Descriptor: artifact.Descriptor{ID: olderID, Size: uint64(len(older)), MediaType: modelartifact.TensorMeasurementMediaType, Schema: modelartifact.TensorMeasurementSchema},
		Data:       older,
	}}}); err != nil {
		t.Fatal(err)
	}
	file := testcheck.Must(gguf.Open(writeTensorFixture(t)))(t)
	defer file.Close()
	model := testcheck.Must(modelartifact.FromGGUF(file, artifact.KindModel))(t)
	policy := modelartifact.MeasurementPolicy{MaxSamplesPerTensor: testAnalysisPolicy.TensorSamples, MaxReadBytes: testAnalysisPolicy.TensorReadBytes}
	measured := testcheck.Must(measureGGUF(model.TensorInventory, file, policy))(t)
	testcheck.NoError(t, publishTensorMeasurement(t.Context(), store, model.TensorInventory, measured))
	found, ok, err := findTensorMeasurement(t.Context(), store, model.TensorInventory.ID, policy)
	if err != nil || !ok || found.ID != measured.ID {
		t.Fatalf("lookup = %v, %v, %v; want the newest measurement %v", found.ID, ok, err, measured.ID)
	}
	other := policy
	other.MaxSamplesPerTensor++
	if _, ok, err := findTensorMeasurement(t.Context(), store, model.TensorInventory.ID, other); ok || err == nil {
		t.Fatalf("a policy nothing measured matched (%v) or read past the older document without error (%v)", ok, err)
	}
}
