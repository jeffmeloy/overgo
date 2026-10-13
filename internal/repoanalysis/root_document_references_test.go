package repoanalysis

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestRootDocumentsNameOnlyWhatTheTreeHolds holds the documents a session or a person opens first to the tree.
//   - root documents = skill.md, README.md and the figure definition docs/assets/overgo_graphic.json
//   - each names no test that no test file declares
//   - each names no path the tree lacks, except a path Git is told to ignore, which a checkout may lack
//   - skill.md and README.md state no promotion, because progress is each scorecard cell beside its previous measurement
//   - the figure definition is valid JSON
//   - a root document absent from the tree fails the test
func TestRootDocumentsNameOnlyWhatTheTreeHolds(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	const figure = "docs/assets/overgo_graphic.json"
	texts := map[string]string{}
	for _, document := range []string{"skill.md", "README.md", figure} {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(document)))
		if err != nil {
			t.Fatal(err)
		}
		texts[document] = string(data)
	}
	if !json.Valid([]byte(texts[figure])) {
		t.Errorf("%s is not valid JSON", figure)
	}
	// ignored reports a path Git is told to ignore, itself or as the directory of what it ignores.
	ignored := func(name string) bool {
		for _, candidate := range []string{name, name + "/entry"} {
			command := exec.CommandContext(t.Context(), "git", "check-ignore", "-q", candidate)
			command.Dir = root
			if command.Run() == nil {
				return true
			}
		}
		return false
	}
	for _, dead := range DeadReferences(root, texts) {
		if _, name, _ := strings.Cut(dead, ": "); !ignored(name) {
			t.Errorf("%s, which the tree no longer has", dead)
		}
	}
	promotion := regexp.MustCompile(`(?i)\bpromot(e|es|ed|ing|ion|ions)\b`)
	for _, document := range []string{"skill.md", "README.md"} {
		for number, line := range strings.Split(texts[document], "\n") {
			if promotion.MatchString(line) && !strings.Contains(line, "never a promotion") {
				t.Errorf("%s:%d states a promotion, which the loop does not make: %s", document, number+1, strings.TrimSpace(line))
			}
		}
	}
}
