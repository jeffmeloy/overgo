package webuilane

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestPageDialogEndsEveryWait holds a JavaScript dialog to what a crash
// already is: an event after which the page answers no evaluation, so every
// wait on it fails at once naming the dialog, instead of reading a reply
// that will not come until the lane's timeout ends the whole run.
func TestPageDialogEndsEveryWait(t *testing.T) {
	t.Parallel()
	dialog := cdpEvent{Method: "Page.javascriptDialogOpening", Params: json.RawMessage(`{"type":"beforeunload","message":"unsaved form"}`)}
	if err := targetEnded(dialog); err == nil || !strings.Contains(err.Error(), "JavaScript dialog") || !strings.Contains(err.Error(), "beforeunload") {
		t.Fatalf("dialog = %v, want a refusal naming the dialog", err)
	}
	for _, method := range []string{"Inspector.targetCrashed", "Inspector.detached"} {
		if err := targetEnded(cdpEvent{Method: method}); err == nil {
			t.Fatalf("%s ended no wait", method)
		}
	}
	if err := targetEnded(cdpEvent{Method: "Page.frameNavigated"}); err != nil {
		t.Fatalf("an ordinary page event ended a wait: %v", err)
	}
}
