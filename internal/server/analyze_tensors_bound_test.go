package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// countFullPasses counts the measuring passes that compute spectra (the
// minutes-long ones on a served model) and lets release hold each one back.
func countFullPasses(t *testing.T, release <-chan struct{}) *int {
	t.Helper()
	passes := 0
	measure := measureGGUF
	measureGGUF = func(inventory modelartifact.TensorInventoryDocument, file *gguf.File, policy modelartifact.MeasurementPolicy) (modelartifact.TensorMeasurementDocument, error) {
		if policy.SpectralMaxDim != 0 {
			passes++
			if release != nil {
				<-release
			}
		}
		return measure(inventory, file, policy)
	}
	t.Cleanup(func() { measureGGUF = measure })
	return &passes
}

func requestTensors(t *testing.T, handler *Handler, query string) analyzeTensorsResponse {
	t.Helper()
	response := serveTestRequest(handler, http.MethodGet, "/analyze/tensors"+query, "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result analyzeTensorsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	return result
}

// TestAnalyzeTensorsIsBounded measures a served model's spectra once: a
// repeat request on the same server and a later server on the same store both
// answer the same inventory without another full pass, with the same profile.
// A full pass over a served model took minutes on every Tensors tab mount.
func TestAnalyzeTensorsIsBounded(t *testing.T) {
	passes := countFullPasses(t, nil)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	path := writeTensorFixture(t)
	serving := newTestHandlerForRepository(t, store, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})
	defer serving.Close()
	first := requestTensors(t, serving, "?wait=spectra")
	again := requestTensors(t, serving, "")
	restarted := newTestHandlerForRepository(t, store, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})
	defer restarted.Close()
	later := requestTensors(t, restarted, "")
	if *passes != 1 {
		t.Fatalf("full passes = %d over a repeat request and a restart, want 1", *passes)
	}
	for _, result := range []analyzeTensorsResponse{first, again, later} {
		if result.Spectra != spectraComplete || len(result.Tensors) != 1 || result.Tensors[0].LMoments != first.Tensors[0].LMoments {
			t.Fatalf("answer %s %+v differs from the measured %+v", result.Spectra, result.Tensors, first.Tensors)
		}
	}
}

// TestAnalyzeTensorsProfilesBeforeSpectra answers a model's first view from
// the sampled pass while the full pass is still running, then the complete
// measurement once it ends: the Tensors tab showed nothing for minutes while
// every effective rank was computed.
func TestAnalyzeTensorsProfilesBeforeSpectra(t *testing.T) {
	release := make(chan struct{})
	countFullPasses(t, release)
	handler := newTestHandler(t, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: writeTensorFixture(t)})
	first := requestTensors(t, handler, "")
	if first.Spectra != spectraPending || first.Count != 1 || first.Tensors[0].Name != "blk.0.weight" {
		t.Fatalf("first view = %s with %d tensors, want pending sampled profiles", first.Spectra, first.Count)
	}
	close(release)
	complete := requestTensors(t, handler, "?wait=spectra")
	if complete.Spectra != spectraComplete || complete.Tensors[0].SpectralStatus != modelartifact.SpectralNotApplicable {
		t.Fatalf("after the full pass = %s, spectral %q; want complete", complete.Spectra, complete.Tensors[0].SpectralStatus)
	}
}
