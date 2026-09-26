package commanddoc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRootDocumentsNameExistingCommands holds the root documents to the tree:
// every cmd/<name> skill.md or README.md names is a command that exists, so a
// renamed or folded command cannot leave an instruction pointing nowhere.
func TestRootDocumentsNameExistingCommands(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	for _, document := range []string{"skill.md", "README.md"} {
		data, err := os.ReadFile(filepath.Join(root, document))
		if err != nil {
			t.Fatal(err)
		}
		named := 0
		for rest := string(data); ; {
			at := strings.Index(rest, "cmd/")
			if at < 0 {
				break
			}
			rest = rest[at+len("cmd/"):]
			end := strings.IndexFunc(rest, func(r rune) bool {
				return !strings.ContainsRune("abcdefghijklmnopqrstuvwxyz0123456789-", r)
			})
			if end < 0 {
				end = len(rest)
			}
			name := rest[:end]
			if name == "" {
				continue
			}
			named++
			if info, err := os.Stat(filepath.Join(root, "cmd", name)); err != nil || !info.IsDir() {
				t.Errorf("%s names cmd/%s, which does not exist", document, name)
			}
		}
		if named == 0 {
			t.Errorf("%s names no command; the check would hold nothing", document)
		}
	}
}
