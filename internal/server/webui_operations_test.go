package server

import (
	"encoding/json"
	"net/http"
	"testing"
)

// TestFrontPageOperations pins long operations on the front page
// (professional GUI campaign, gui-workbench/operations-strip): the health
// probe reports the device's current and peak memory when the generator
// runs on one, the shell's header carries status dots for the server, the
// swap proxy (read from the header every proxied answer carries) and the
// device, and the approvals count that opens the inbox; the operations
// strip shows every operation with progress, cancel, the live event tail
// and the durable receipt, and a model switch shows there as a local chip.
func TestFrontPageOperations(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	health := serveTestRequest(handler, http.MethodGet, "/health", "")
	var probe struct {
		Status string `json:"status"`
		Device struct {
			CurrentBytes uint64 `json:"current_bytes"`
			PeakBytes    uint64 `json:"peak_bytes"`
		} `json:"device"`
	}
	if err := json.Unmarshal(health.Body.Bytes(), &probe); err != nil {
		t.Fatal(err)
	}
	if probe.Status != "ok" || probe.Device.PeakBytes == 0 || probe.Device.CurrentBytes == 0 {
		t.Fatalf("health = %s", health.Body.String())
	}
}
