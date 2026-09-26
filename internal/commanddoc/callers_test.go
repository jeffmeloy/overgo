package commanddoc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gitauthority"
)

// TestEveryCommandHasACaller holds every shipped command to a caller the tree
// shows -- code, a script, a document or a test that names cmd/<name> outside
// the command's own directory -- or to a declared identity in Declared (owner
// directive 2026-09-24: a command nothing refers to is declared, not deleted).
// The generated API manifest and the plan, which name every command or the
// ones under review, are not callers.
func TestEveryCommandHasACaller(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	listed, err := gitauthority.Query(t.Context(), root, "ls-files")
	if err != nil {
		t.Fatal(err)
	}
	texts := map[string]string{}
	for path := range strings.FieldsSeq(string(listed)) {
		if path == "docs/api_manifest.json" || path == "docs/plan.json" || !strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, ".md") &&
			!strings.HasSuffix(path, ".json") && !strings.HasSuffix(path, ".sh") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		texts[path] = string(data)
	}
	commands, err := os.ReadDir(filepath.Join(root, "cmd"))
	if err != nil {
		t.Fatal(err)
	}
	for _, command := range commands {
		name := command.Name()
		if !command.IsDir() || Declared[name].Command.Purpose != "" || namedOutside(texts, name) {
			continue
		}
		t.Errorf("cmd/%s has no caller the tree shows and no declared identity in commanddoc.Declared", name)
	}
	for name, descriptor := range Declared {
		if descriptor.Command.Name != name || descriptor.Command.Audience == "" {
			t.Errorf("declared command %s is incomplete: %+v", name, descriptor.Command)
		}
	}
}

// namedOutside reports whether any file outside cmd/<name>/ names the command
// as a whole path segment, so cmd/train does not count for cmd/train-lane.
func namedOutside(texts map[string]string, name string) bool {
	mention := "cmd/" + name
	for path, text := range texts {
		if strings.HasPrefix(path, mention+"/") {
			continue
		}
		for rest := text; ; {
			at := strings.Index(rest, mention)
			if at < 0 {
				break
			}
			rest = rest[at+len(mention):]
			if rest == "" || !strings.ContainsRune("abcdefghijklmnopqrstuvwxyz0123456789-", rune(rest[0])) {
				return true
			}
		}
	}
	return false
}
