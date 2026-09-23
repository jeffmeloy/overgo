package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// TestAnalyzeTensorsIsBounded measures a served model once: a repeat request
// on the same server and a later server on the same store both answer the
// same inventory without another pass over its tensors, with the same
// profile. A full pass over a served model took minutes on every Tensors tab
// mount.
func TestAnalyzeTensorsIsBounded(t *testing.T) {
	passes := 0
	measure := measureGGUF
	measureGGUF = func(inventory modelartifact.TensorInventoryDocument, file *gguf.File, policy modelartifact.MeasurementPolicy) (modelartifact.TensorMeasurementDocument, error) {
		passes++
		return measure(inventory, file, policy)
	}
	t.Cleanup(func() { measureGGUF = measure })
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	path := writeTensorFixture(t)
	request := func(handler *Handler) analyzeTensorsResponse {
		t.Helper()
		response := serveTestRequest(handler, http.MethodGet, "/analyze/tensors", "")
		if response.Code != http.StatusOK {
			t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
		}
		var result analyzeTensorsResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	serving := newTestHandlerForRepository(t, store, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})
	defer serving.Close()
	first := request(serving)
	again := request(serving)
	restarted := newTestHandlerForRepository(t, store, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})
	defer restarted.Close()
	later := request(restarted)
	if passes != 1 {
		t.Fatalf("measuring passes = %d over a repeat request and a restart, want 1", passes)
	}
	for _, result := range []analyzeTensorsResponse{again, later} {
		if len(result.Tensors) != 1 || result.Tensors[0].LMoments != first.Tensors[0].LMoments {
			t.Fatalf("reused profile %+v differs from the measured %+v", result.Tensors, first.Tensors)
		}
	}
}
