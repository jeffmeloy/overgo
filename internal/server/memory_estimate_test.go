package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/runrecord"
)

// TestRunMemoryEstimateFromMeasuredRecords: before a model serves, the
// catalog states the device memory its runs measured -- the benchmark's
// bytes after load, peak, device total and residency, and the peak at each
// prompt length its long-form run climbed on that run's device -- a model no
// run measured states none, and the training capability declares that it
// runs on the host, needing no device memory.
func TestRunMemoryEstimateFromMeasuredRecords(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	ctx := t.Context()
	store := handler.config.Repository
	weightsPath := filepath.Join(t.TempDir(), "measured.gguf")
	if err := os.WriteFile(weightsPath, []byte("measured memory weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	weights, err := os.Open(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	if err != nil {
		t.Fatal(err)
	}
	commit := func(capability string, peak uint64, document string) {
		t.Helper()
		data := []byte(document)
		evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		record, err := runrecord.NewModelVerification(model, "fixture", []runrecord.CapabilityClaim{{
			Capability: capability, Tier: runrecord.TierCapabilityMeasured, PeakDeviceBytes: peak, WallNS: 1_000_000,
			Commit: "0123456789abcdef0123456789abcdef01234567", Evidence: []artifact.ID{evidence},
		}})
		if err != nil {
			t.Fatal(err)
		}
		if err := runrecord.CommitVerificationClaim(ctx, store, record, data, evidence, weightsPath); err != nil {
			t.Fatal(err)
		}
	}
	commit("benchmark", 6_000_000_000, `{"device_after_load_bytes":5000000000,"realized_residency":"device-native","device":{"totalMemoryBytes":24000000000},"summary":{"decode_tokens_per_second_p50":40}}`)
	commit("long-form", 0, `{"verdict":{"passed":true},"device":{"totalMemoryBytes":24000000000},"rungs":[
		{"measure":{"prompt_tokens":1024,"memory":{"peakBytes":6100000000}}},
		{"measure":{"prompt_tokens":4096,"memory":{"peakBytes":6900000000}}},
		{"measure":{"prompt_tokens":8192}}]}`)
	registered, err := filepath.Abs(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	rows := catalogListing([]discovery.CatalogEntry{
		{Model: model, Location: registered, Present: true},
		{Model: model, Location: "C:/models/unmeasured.gguf", Present: true},
	}, handler.catalogEvidence(ctx, nil))
	memory := rows[0].Memory
	if memory == nil || memory.LoadBytes != 5_000_000_000 || memory.PeakBytes != 6_000_000_000 || memory.DeviceBytes != 24_000_000_000 ||
		memory.Residency != "device-native" || memory.ContextDeviceBytes != 24_000_000_000 ||
		!slices.Equal(memory.Context, []contextPeak{{PromptTokens: 1024, PeakBytes: 6_100_000_000}, {PromptTokens: 4096, PeakBytes: 6_900_000_000}}) {
		t.Fatalf("measured memory = %+v", memory)
	}
	if rows[1].Memory != nil {
		t.Fatalf("an unmeasured model states memory %+v", rows[1].Memory)
	}

	fixture := newDPOTrainingFixture(t)
	training, err := New(Config{Repository: fixture.store}, &workspaceTestRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: fixture.workspace})
	if err != nil {
		t.Fatal(err)
	}
	defer training.Close()
	var capabilities []WorkflowCapability
	listed := serveTestRequest(training, http.MethodGet, "/training/capabilities", "")
	if err := json.Unmarshal(listed.Body.Bytes(), &capabilities); err != nil || len(capabilities) != 1 || !capabilities[0].Host {
		t.Fatalf("training capabilities = %s", listed.Body.String())
	}
}
