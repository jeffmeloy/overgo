package server

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// TestTensorMeasurementLookupStopsAtItsMatch finds a model's committed
// measurement without reading the measurements behind it: the lookup read
// and decoded every stored measurement, each a whole document, after the
// newest one had already matched. An older document that cannot be decoded
// stands behind the match here, so a lookup that reads past it fails.
func TestTensorMeasurementLookupStopsAtItsMatch(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	older := []byte(`{"schema":"not a measurement"}`)
	olderID, err := artifact.IdentifyBytes(artifact.KindTensorInventory, older)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "server/tensor-lookup-older", Contents: []artifact.Content{{
		Descriptor: artifact.Descriptor{ID: olderID, Size: uint64(len(older)), MediaType: modelartifact.TensorMeasurementMediaType, Schema: modelartifact.TensorMeasurementSchema},
		Data:       older,
	}}}); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Open(writeTensorFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	model, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	policy := modelartifact.MeasurementPolicy{MaxSamplesPerTensor: testAnalysisPolicy.TensorSamples, MaxReadBytes: testAnalysisPolicy.TensorReadBytes}
	measured, err := measureGGUF(model.TensorInventory, file, policy)
	if err != nil {
		t.Fatal(err)
	}
	if err := publishTensorMeasurement(t.Context(), store, model.TensorInventory, measured); err != nil {
		t.Fatal(err)
	}
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
