package landing

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestARefusalKeepsItsCause holds the refusal record to the cause each plan review counts refusals by
// (refusal-causes/refusal-cause): a refusal no stage made, one of a stage and one of the rules stage are each kept with
// the first line of their text past the gate's prefix, a long one cut to the bound, and with their class and rules as
// before.
func TestARefusalKeepsItsCause(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	long := "land: " + strings.Repeat("a cause that runs on ", 40)
	for _, text := range []string{
		"land: the change removes row/old from the plan without landing it: retire it with go run ./cmd/plan -retire",
		"gate: land: verify: land: no check of row/do fails on an assertion\nthe rest of the output",
		"gate: land: rules: new sort.Slice (use slices.SortFunc) in internal/p/p.go",
		long,
	} {
		testcheck.Must(RecordRefusal(root, Classify("row/do", text)))(t)
	}
	data := testcheck.Must(os.ReadFile(filepath.Join(root, processlock.StateDirectory, refusalRecord)))(t)
	var kept []struct {
		Class string   `json:"class"`
		Rules []string `json:"rules"`
		Cause string   `json:"cause"`
	}
	testcheck.NoError(t, json.Unmarshal(data, &kept))
	if len(kept) != 4 {
		t.Fatalf("the record holds %d refusals; want 4", len(kept))
	}
	want := []struct{ class, cause string }{
		{"gate", "the change removes row/old from the plan without landing it: retire it with go run ./cmd/plan -retire"},
		{"verify", "verify: land: no check of row/do fails on an assertion"},
		{"rules", "rules: new sort.Slice (use slices.SortFunc) in internal/p/p.go"},
	}
	for index, refusal := range want {
		if kept[index].Class != refusal.class || kept[index].Cause != refusal.cause {
			t.Errorf("refusal %d is kept as %s by %q; want %s by %q", index, kept[index].Class, kept[index].Cause, refusal.class, refusal.cause)
		}
	}
	if !slices.Equal(kept[2].Rules, []string{"legacy-forms"}) {
		t.Errorf("the rules refusal names the rules %v; want legacy-forms", kept[2].Rules)
	}
	if cause := kept[3].Cause; len(cause) == 0 || len(cause) > 240 || !strings.HasPrefix(long, "land: "+cause) {
		t.Errorf("the long refusal's cause is %d bytes, %q; want its opening, at most 240 bytes", len(cause), cause)
	}
}
