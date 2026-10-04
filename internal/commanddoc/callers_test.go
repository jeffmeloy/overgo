package commanddoc

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"os/exec"
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
	texts, err := trackedTexts(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	commands, err := os.ReadDir(filepath.Join(root, "cmd"))
	if err != nil {
		t.Fatal(err)
	}
	for _, command := range commands {
		name := command.Name()
		// A command is a directory holding Go source; one left holding only
		// ignored files, a store a test wrote there, is none.
		sources, err := filepath.Glob(filepath.Join(root, "cmd", name, "*.go"))
		if err != nil {
			t.Fatal(err)
		}
		if !command.IsDir() || len(sources) == 0 || Declared[name].Command.Purpose != "" || namedOutside(texts, name) {
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

// trackedTexts reads the tracked code, document and script files of the
// checkout at root, except the generated manifest and the plan. A tracked file
// the working tree no longer holds, a deletion not staged yet, is absent: the
// tree shows no caller there, and a landing that deletes a file is not refused
// for it.
func trackedTexts(ctx context.Context, root string) (map[string]string, error) {
	listed, err := gitauthority.Query(ctx, root, "ls-files")
	if err != nil {
		return nil, err
	}
	texts := map[string]string{}
	for path := range strings.FieldsSeq(string(listed)) {
		if path == "docs/api_manifest.json" || path == "docs/plan.json" || !strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, ".md") &&
			!strings.HasSuffix(path, ".json") && !strings.HasSuffix(path, ".sh") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		texts[path] = string(data)
	}
	return texts, nil
}

// TestTrackedTextsSkipAFileDeletedBeforeItIsStaged holds a deletion in the
// working tree, which git still lists until it is staged, to being absent
// from the texts rather than an error.
func TestTrackedTextsSkipAFileDeletedBeforeItIsStaged(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	git := func(arguments ...string) {
		t.Helper()
		command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "-c", "commit.gpgsign=false"}, arguments...)...)
		command.Dir = root
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
	}
	for _, name := range []string{"kept.md", "deleted.md"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("see cmd/tool"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	git("add", "-A")
	git("commit", "-q", "-m", "tracked")
	if err := os.Remove(filepath.Join(root, "deleted.md")); err != nil {
		t.Fatal(err)
	}
	texts, err := trackedTexts(t.Context(), root)
	if err != nil || len(texts) != 1 || texts["kept.md"] == "" {
		t.Fatalf("texts of a tree with one file deleted before it is staged = %v, %v, want kept.md alone", texts, err)
	}
}

// namedOutside reports whether any file outside cmd/<name>/ names the command
// as a whole path segment, so cmd/benchmark-report does not count for cmd/benchmark.
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
