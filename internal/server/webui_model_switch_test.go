package server

import (
	"strings"
	"testing"
)

// TestFrontPageModelSwitch pins the model selector on the front page
// (professional GUI campaign, gui-shell/model-switch): the selector lists
// the catalog with each entry's evidence line (prompt and decode tok/s,
// evaluation accuracies) and declared task capabilities, serves a model
// live through the swap proxy with elapsed-time progress, re-reads the
// capability document and re-mounts the active surface on arrival so the
// composer re-derives, explains itself without the proxy, remembers the
// last choice per browser, and lists a stale activation with the loader's
// reason and no serve control.
func TestFrontPageModelSwitch(t *testing.T) {
	sources := webuiJavaScript(t)
	boot := sources["boot.js"]
	for _, needle := range []string{
		`"/catalog/models"`, `"/health?swap="`, "prompt_tokens_per_second", "decode_tokens_per_second",
		"item.capabilities", "item.stale", "MODEL_STORAGE", "elapsed", "remountActive(", "invalidateModel()",
		"overgo_gui.bat", "capabilityDocument = ",
	} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js selector lacks %s", needle)
		}
	}
	// A stale entry is never offered a serve control: the button is created
	// only for servable entries.
	if strings.Index(boot, "item.stale") > strings.Index(boot, `text: "serve"`) {
		t.Error("boot.js decides the serve control before reading the entry's staleness")
	}
}
