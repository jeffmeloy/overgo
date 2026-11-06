package storepolicy

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestCheckoutRootsUnionDocumentsAndCommitMessages collects the identities a
// checkout pins without an alias -- from its committed JSON documents and
// from every commit message -- once each and sorted, reports each source's
// count, and ignores text that is not an identity and files that are not
// JSON.
func TestCheckoutRootsUnionDocumentsAndCommitMessages(t *testing.T) {
	checkout := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", checkout}, args...)...)
		command.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.invalid",
			"GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.invalid",
		)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
	}
	write := func(relative, text string) {
		t.Helper()
		path := filepath.Join(checkout, relative)
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(text), 0o644))
	}
	receipt := "evidence:sha256:" + strings.Repeat("a", 64)
	model := "model:sha256:" + strings.Repeat("b", 64)
	preparation := "evidence:sha256:" + strings.Repeat("c", 64)
	ignored := "file:sha256:" + strings.Repeat("d", 64)
	git("init", "-q")
	write("docs/verification/receipt.json", `{"evidence":["`+receipt+`","`+receipt+`"],"note":"not:an:id sha256:short"}`)
	write("compatibility.json", `{"model":"`+model+`"}`)
	write("docs/notes.md", ignored)
	git("add", ".")
	git("commit", "-q", "-m", "first\n\nOvergo-Gate-Preparation: "+preparation)
	write("docs/notes.md", ignored+" again")
	git("add", ".")
	git("commit", "-q", "-m", "second\n\nOvergo-Gate-Preparation: "+preparation+"\nOvergo-Evidence: "+receipt)

	roots, documents, messages, err := CheckoutRoots(t.Context(), checkout)
	testcheck.NoError(t, err)
	var want []artifact.ID
	for _, text := range []string{receipt, model, preparation} {
		id := testcheck.Must(artifact.ParseID(text))(t)
		want = append(want, id)
	}
	slices.SortFunc(want, artifact.CompareID)
	if !slices.Equal(roots, want) || documents != 2 || messages != 2 {
		t.Fatalf("roots = %v (documents=%d messages=%d), want %v", roots, documents, messages, want)
	}
}
