package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMemoryTiers drives the memory bar beside the pill: the
// served model's owners, largest first, then another process's share, each
// sized against the device total with every owner and the free bytes in its
// label; a changed device reads on the next probe; a device reporting no
// total shows no bar.
func TestWebUIBrowserMemoryTiers(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the memory tiers leg uses Chromium through cmd/webui-lane")
	generator := &memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: testDeviceMemory()}
	owned := httptest.NewServer(newTestHandler(t, generator))
	defer owned.Close()
	plain := httptest.NewServer(newTestHandler(t, &fakeGenerator{}))
	defer plain.Close()
	ctx, browser, check, settle := openBrowserLeg(t, owned.URL)
	defer browser.Close()
	const parts = `[...document.querySelectorAll('#memory-bar .memory-part')].map(part => part.dataset.owner + ' ' + Math.round(part.getBoundingClientRect().width / document.getElementById('memory-bar').clientWidth * 24)).join()`

	// The owners, largest first, then the other process, each its share of 24 GiB.
	settle(`!document.getElementById('memory-bar').hidden && document.querySelectorAll('#memory-bar .memory-part').length === 4`)
	check(parts + ` === 'weights 8,KV cache and buffers 3,activations 1,other processes 2'`)
	check(`document.getElementById('memory-bar').getAttribute('aria-label') === 'device memory: weights 8.00 GB · KV cache and buffers 3.00 GB · activations 1.00 GB · other processes 2.00 GB · free 10.00 GB of 24.00 GB'`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "memory-bar")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// A changed device reads on the next probe.
	generator.set(inference.DeviceMemory{TotalBytes: 24 * gibibyte, FreeBytes: 4 * gibibyte, Owners: []inference.DeviceMemoryOwner{
		{Name: "KV cache and buffers", Bytes: 9 * gibibyte}, {Name: "weights", Bytes: 8 * gibibyte}, {Name: "activations", Bytes: 1 * gibibyte},
	}})
	check(`(() => { document.dispatchEvent(new Event('visibilitychange')); return true; })()`)
	settle(parts + ` === 'KV cache and buffers 9,weights 8,activations 1,other processes 2'`)

	// A device reporting no total shows no bar.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`" + ` && document.getElementById('device-dot').classList.contains('ok')`)
	check(`document.getElementById('memory-bar').hidden`)
	webuilane.Leg(t, "memory tiers leg", "the bar beside the pill showed the served model's owners largest first and another process's share, each sized against the device total with the free bytes in its label; a changed device read on the next probe; a device reporting no total showed no bar")
}
