package plan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestThePlanHoldsNoRuleText holds the plan to rows and decisions: its eleven standing paragraphs, 8 KB, were read by
// sessions as rules beside skill.md, and rules written in two places disagree. The repository's plan holds no
// paragraphs and no doctrine string, and a plan that holds them, as an earlier commit's does, is read and saved without
// them.
func TestThePlanHoldsNoRuleText(t *testing.T) {
	t.Parallel()
	var held map[string]json.RawMessage
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(Path))))(t), &held))
	for _, key := range []string{"paragraphs", "doctrine"} {
		if _, found := held[key]; found {
			t.Errorf("%s holds %q, rule text the plan keeps no more", Path, key)
		}
	}
	earlier := []byte(`{"campaign": "fixture", "paragraphs": [{"governs": "standing", "text": "a rule written in the plan"}],
		"items": [{"id": "row", "title": "row", "status": "open", "steps": [{"id": "do", "title": "do", "status": "open", "verify": "true"}]}]}`)
	document := testcheck.Must(Parse(earlier))(t)
	path := filepath.Join(t.TempDir(), "plan.json")
	testcheck.NoError(t, Save(path, document))
	if saved := string(testcheck.Must(os.ReadFile(path))(t)); strings.Contains(saved, "a rule written in the plan") || strings.Contains(saved, "paragraphs") {
		t.Errorf("an earlier plan saved again keeps its rule text:\n%s", saved)
	}
}
