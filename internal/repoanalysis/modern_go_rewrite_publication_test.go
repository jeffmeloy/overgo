package repoanalysis

import (
	"cmp"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModernGoRewritePublication(t *testing.T) {
	const source = "package sample\n\nvar A = 1\nvar B = 2\n"
	first, second := strings.Index(source, "1"), strings.Index(source, "2")
	finishError := errors.New("import repair refused")
	for _, test := range []struct {
		name          string
		edits         []modernGoSourceEdit
		current, want string
		finish        func(string, []byte) ([]byte, error)
		refused       bool
	}{
		{name: "unordered edits", edits: []modernGoSourceEdit{{second, second + 1, "4"}, {first, first + 1, "3"}}, want: "package sample\n\nvar A = 3\nvar B = 4\n"},
		{name: "overlap", edits: []modernGoSourceEdit{{first, first + 1, "3"}, {first, first + 1, "4"}}, refused: true},
		{name: "negative offset", edits: []modernGoSourceEdit{{-1, first, "3"}}, refused: true},
		{name: "reversed interval", edits: []modernGoSourceEdit{{first + 1, first, "3"}}, refused: true},
		{name: "past source", edits: []modernGoSourceEdit{{first, len(source) + 1, "3"}}, refused: true},
		{name: "source changed", current: source + "// newer source\n", edits: []modernGoSourceEdit{{first, first + 1, "3"}}, refused: true},
		{name: "invalid output", edits: []modernGoSourceEdit{{first, first + 1, "}"}}, refused: true},
		{name: "finish failure", finish: func(string, []byte) ([]byte, error) { return nil, finishError }, refused: true},
		{name: "unchanged cleanup", finish: func(_ string, data []byte) ([]byte, error) { return data, nil }},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			path := filepath.Join(root, "sample.go")
			current := cmp.Or(test.current, source)
			if err := os.WriteFile(path, []byte(current), 0o600); err != nil {
				t.Fatal(err)
			}
			before, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			snapshot, err := (SourceSnapshot{}).Overlay(map[string][]byte{"sample.go": []byte(source)})
			if err != nil {
				t.Fatal(err)
			}
			changed, err := rewriteModernGoFile(root, snapshot.Files[0], test.edits, "fixture", test.finish)
			if (err != nil) != test.refused {
				t.Fatalf("refused=%v want %v: %v", err != nil, test.refused, err)
			}
			if test.name == "finish failure" && !errors.Is(err, finishError) {
				t.Fatalf("lost finish error: %v", err)
			}
			want := cmp.Or(test.want, current)
			if changed != (want != current) {
				t.Fatalf("changed=%v want %v", changed, want != current)
			}
			got, err := os.ReadFile(path)
			if err != nil || string(got) != want {
				t.Fatalf("published %q want %q: %v", got, want, err)
			}
			after, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if before.Mode() != after.Mode() {
				t.Fatalf("file mode changed: %v -> %v", before.Mode(), after.Mode())
			}
		})
	}
}
