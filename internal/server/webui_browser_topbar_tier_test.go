package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserPhoneTopBarHoldsALongSupportTier holds the top bar, at a
// phone's width, to the longest it is asked to show: a served model whose
// support tag names its tier and both rungs that stand beside it, with the
// device's memory bar shown. The model picker keeps a width a finger can
// press, nothing stands off the screen, and it is the tag that gives way, cut
// with an ellipsis. The full measurement run's first-run journey found the
// fault on a model that had gained a training capability: the picker was
// squeezed to no width and Settings stood two pixels off the screen.
func TestWebUIBrowserPhoneTopBarHoldsALongSupportTier(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the phone top bar leg uses Chromium through cmd/webui-lane")
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Tier answer"}}))
	var health map[string]any
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/health", "").Body.Bytes(), &health); err != nil || health["model"] == nil {
		t.Fatalf("health = %+v, %v", health, err)
	}
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil || manifest.Model == nil || manifest.Model.Model == "" {
		t.Fatalf("manifest = %+v, %v", manifest, err)
	}
	rung := func(name string, held bool) map[string]any {
		return map[string]any{"name": name, "held": held, "detail": "the top bar leg's " + name + " rung"}
	}
	catalog, err := json.Marshal(map[string]any{"truncated": false, "models": []map[string]any{{
		"model": manifest.Model.Model, "location": "C:/models/" + health["model"].(string), "present": true, "recipe": "recipe:served",
		"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:served", "tier": "verified"}},
		"support": map[string]any{"tier": "parity-checked", "rungs": []map[string]any{
			rung("registered", true), rung("self-consistent", true), rung("parity-checked", true), rung("guarded", true), rung("trainable", true),
		}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	// The health probe as a server with a device answers it, so the memory bar stands in the bar too.
	memory := testDeviceMemory()
	health["device"] = map[string]any{"current_bytes": memory.TotalBytes - memory.FreeBytes, "peak_bytes": memory.TotalBytes - memory.FreeBytes, "total_bytes": memory.TotalBytes, "free_bytes": memory.FreeBytes, "owners": memory.Owners}
	probed, err := json.Marshal(health)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/catalog/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(catalog)
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(probed)
		default:
			handler.ServeHTTP(w, r)
		}
	}))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)
	settle(`(() => { const tier = document.getElementById('model-tier'); return !tier.hidden && tier.textContent === 'parity-checked · guarded · trainable' && !document.getElementById('memory-bar').hidden; })()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "long-support-tier")
		if err != nil || len(findings) != 0 {
			t.Errorf("%s: the top bar with a long support tier: %v %v", viewport.Name, findings, err)
		}
	}
	// The last viewport captured is the phone's: there the picker keeps its width and the tag is the one cut.
	check(`(() => {
  const pill = document.getElementById('model-pill').getBoundingClientRect(), tier = document.getElementById('model-tier');
  const settings = document.getElementById('settings-toggle').getBoundingClientRect();
  if (innerWidth > 400) throw new Error('the page is ' + innerWidth + ' wide; want the phone viewport');
  if (pill.width < 44) throw new Error('the model picker is ' + pill.width + 'px wide at a phone width; want a width a finger can press');
  if (settings.right > innerWidth) throw new Error('Settings ends at ' + settings.right + 'px of ' + innerWidth);
  if (tier.scrollWidth <= tier.clientWidth || getComputedStyle(tier).textOverflow !== 'ellipsis') throw new Error('the tag shows ' + tier.clientWidth + ' of ' + tier.scrollWidth + 'px with text-overflow ' + getComputedStyle(tier).textOverflow + '; want it cut with an ellipsis');
  return true;
})()`)
	webuilane.Leg(t, "phone top bar leg", "with a support tag of a tier and two rungs and the memory bar shown, the model picker kept a finger's width at a phone's width, nothing stood off the screen and the tag was cut with an ellipsis")
}
