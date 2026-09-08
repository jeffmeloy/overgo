package server

import (
	"encoding/json"
	"net/http"
	"strings"
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

	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }
	shell := get("/")
	for _, needle := range []string{`id="server-dot"`, `id="proxy-dot"`, `id="device-dot"`, `id="inbox-count"`} {
		if !strings.Contains(shell, needle) {
			t.Errorf("shell missing %q", needle)
		}
	}
	boot := get("/boot.js")
	for _, needle := range []string{`"X-Overgo-Swap-Proxy"`, "opts.onHeaders(response.headers)", "health.device", "peak_bytes", "overgo.localOperation("} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot missing %q", needle)
		}
	}
	strip := get("/operations_shell.js")
	for _, needle := range []string{"Live events", `"inbox-count"`, `location.hash = "inbox"`, "function localOperation(", "Download result", `"/operations/cancel"`, `"/operations/evidence?id="`} {
		if !strings.Contains(strip, needle) {
			t.Errorf("operations shell missing %q", needle)
		}
	}
	if !strings.Contains(get("/style.css"), ".dot.ok") {
		t.Error("style lacks the status dots")
	}
}
