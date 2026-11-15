package plan

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPlanSavedWithNoStatusReadsAsOpen holds the plan to reading a row with no status as open and to saving none
// (single-source/status-read): every row the plan holds is open, and the step that removes the field lands under the
// gate built from the trunk, which must read the plan the step writes; an empty status had been refused as invalid.
func TestAPlanSavedWithNoStatusReadsAsOpen(t *testing.T) {
	t.Parallel()
	read := testcheck.Must(Parse([]byte(`{"campaign":"x","items":[{"id":"row","title":"Row","steps":[{"id":"do","title":"Do","verify":"go test ./..."}]}]}`)))(t)
	if item, step, ok := Current(read, ""); !ok || item.ID != "row" || step.ID != "do" {
		t.Fatalf("a plan with no status dispatches %s/%s (%v); want row/do", item.ID, step.ID, ok)
	}
	saved := filepath.Join(t.TempDir(), "plan.json")
	testcheck.NoError(t, Save(saved, read))
	if text := string(testcheck.Must(os.ReadFile(saved))(t)); strings.Contains(text, `"status"`) {
		t.Fatalf("the plan is saved with a status: %s", text)
	}
	if again := testcheck.Must(Load(saved))(t); again.Items[0].Steps[0].Status != StatusOpen {
		t.Fatalf("the saved plan reads its step as %q; want it open", again.Items[0].Steps[0].Status)
	}
}
