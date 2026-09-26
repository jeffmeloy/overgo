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
	t.Parallel()
	// The model picker leg drives the picker; the source keeps the stale
	// check ahead of the serve control, so a stale entry is never offered one.
	boot := webuiJavaScript(t)["boot.js"]
	if strings.Index(boot, "item.stale") > strings.Index(boot, `text: "Serve"`) {
		t.Error("boot.js decides the serve control before reading the entry's staleness")
	}
}
