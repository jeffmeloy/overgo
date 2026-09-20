package storepolicy

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
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
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
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
	if err != nil {
		t.Fatal(err)
	}
	var want []artifact.ID
	for _, text := range []string{receipt, model, preparation} {
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, id)
	}
	slices.SortFunc(want, artifact.CompareID)
	if !slices.Equal(roots, want) || documents != 2 || messages != 2 {
		t.Fatalf("roots = %v (documents=%d messages=%d), want %v", roots, documents, messages, want)
	}
}

// TestCheckoutRootsHonorTheProofHorizon pins commit-message roots to the
// history after a committed proof horizon: an identity only a landing at or
// before the horizon names stops being a root, one a later landing names
// stays, a receipt that is only in the working tree releases nothing, and the
// receipt's own commit identity is not mistaken for a store identity.
func TestCheckoutRootsHonorTheProofHorizon(t *testing.T) {
	checkout := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", checkout}, args...)...)
		command.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.invalid",
			"GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.invalid",
		)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	write := func(relative, text string) {
		t.Helper()
		path := filepath.Join(checkout, relative)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	early := "evidence:sha256:" + strings.Repeat("e", 64)
	late := "evidence:sha256:" + strings.Repeat("f", 64)
	named := func() []string {
		t.Helper()
		roots, _, _, err := CheckoutRoots(t.Context(), checkout)
		if err != nil {
			t.Fatal(err)
		}
		var texts []string
		for _, root := range roots {
			texts = append(texts, root.String())
		}
		return texts
	}
	git("init", "-q")
	write("compatibility.json", `{}`)
	git("add", ".")
	git("commit", "-q", "-m", "landed early\n\nOvergo-Gate-Preparation: "+early)
	horizon := git("rev-parse", "HEAD")
	write("docs/plan_proof_horizon.json", `{"commit":"`+horizon+`","completions":1}`)
	if got := named(); !slices.Equal(got, []string{early}) {
		t.Fatalf("an uncommitted receipt changed the roots to %v", got)
	}
	git("add", ".")
	git("commit", "-q", "-m", "landed late\n\nOvergo-Gate-Preparation: "+late)
	if got := named(); !slices.Equal(got, []string{late}) {
		t.Fatalf("roots after the horizon = %v, want only %s", got, late)
	}
}
