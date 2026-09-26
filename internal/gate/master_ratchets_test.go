package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestMergeRefusesLooserRatchet holds a merge to master's ceilings and
// floors: a raised ceiling, an added exception or a lowered or dropped floor
// is named, a tightening is not, and every ratchet master owns is a tracked
// JSON document.
func TestMergeRefusesLooserRatchet(t *testing.T) {
	t.Parallel()
	decode := func(text string) any {
		var value any
		if err := json.Unmarshal([]byte(text), &value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	for _, tc := range []struct {
		name, before, after string
		ceiling             bool
		want                []string
	}{
		{"raised ceiling", `{"duplicate_excess_nodes": 7991}`, `{"duplicate_excess_nodes": 9970}`, true, []string{"c.duplicate_excess_nodes 7991 -> 9970"}},
		{"lowered ceiling", `{"duplicate_excess_nodes": 7991}`, `{"duplicate_excess_nodes": 7964}`, true, nil},
		{"added exception", `{"exceptions": [{"subject": "a", "limit": 2100}]}`, `{"exceptions": [{"subject": "a", "limit": 2100}, {"subject": "b", "limit": 2500}]}`, true,
			[]string{`c.exceptions entry {"limit":2500,"subject":"b"}`}},
		{"retired exception", `{"exceptions": [{"subject": "a", "limit": 2100}]}`, `{"exceptions": []}`, true, nil},
		{"lowered floor", `{"hermetic": {"./internal/server": 72.7}}`, `{"hermetic": {"./internal/server": 60}}`, false, []string{"c.hermetic../internal/server 72.7 -> 60"}},
		{"dropped floor", `{"hermetic": {"./internal/server": 72.7}}`, `{"hermetic": {}}`, false, []string{"c.hermetic../internal/server removed"}},
		{"raised floor", `{"hermetic": {"./internal/server": 72.7}}`, `{"hermetic": {"./internal/server": 80, "./internal/new": 50}}`, false, nil},
	} {
		if got := looserRatchet("c", decode(tc.before), decode(tc.after), tc.ceiling); !slices.Equal(got, tc.want) {
			t.Errorf("%s: loosened %q, want %q", tc.name, got, tc.want)
		}
	}
	for path := range masterRatchets {
		data, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(path)))
		if err != nil || !json.Valid(data) {
			t.Errorf("master ratchet %s is not a tracked JSON document: %v", path, err)
		}
	}
}
