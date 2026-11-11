package landing

import (
	"crypto/sha256"
	"encoding/hex"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestEveryRuleRefusalFixesOrPrintsItsEdit holds each refusal of the rules stage to the edit that clears it: from Oct
// 31 to Nov 7 a refusal said what was wrong and left the fix to be guessed. A refusal whose edit is a command is cleared
// by running it in the checkout; one whose edit is a change to write names the line and the words, and is cleared by
// writing them; one the fixture cannot apply names its edit. A feature row's growth past its count is recorded, not
// refused; a consolidation's is refused with what it must remove.
func TestEveryRuleRefusalFixesOrPrintsItsEdit(t *testing.T) {
	t.Parallel()
	shell := testcheck.Must(clioptions.POSIXShell())(t)
	// refused takes the fixture's dirty tree as row/do's landing and returns its rules refusal, which names an edit.
	refused := func(fixture *sliceFixture, want string) string {
		t.Helper()
		err := fixture.candidate().rules(t.Context())
		if err == nil || !strings.Contains(err.Error(), want) || !strings.Contains(err.Error(), "edit") {
			t.Fatalf("the rules stage ended as %v, want a refusal naming %q and its edit", err, want)
		}
		return err.Error()
	}
	// cleared runs the refusal's command in the checkout, the text from "edit: " up to stop, and holds the rules to it.
	cleared := func(fixture *sliceFixture, refusal, stop string) {
		t.Helper()
		_, edit, _ := strings.Cut(refusal, "edit: ")
		edit, _, _ = strings.Cut(edit, stop)
		command := exec.Command(shell, "-c", edit)
		command.Dir = fixture.repo
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("the edit %q: %v: %s", edit, err, output)
		}
		testcheck.NoError(t, fixture.candidate().rules(t.Context()))
	}

	pinned := newRowFixture(t, map[string]string{"a.txt": "base\n"})
	digest := sha256.Sum256([]byte("base\n"))
	pinned.write("docs/verification/a-coverage.json", `{"inputs": {"a.txt": "`+hex.EncodeToString(digest[:])+`"}}`+"\n")
	pinned.git("add", "-A")
	pinned.git("commit", "-q", "-m", "a pin")
	pinned.write("a.txt", "changed\n")
	cleared(pinned, refused(pinned, "pinned by docs/verification/a-coverage.json"), ", or else")

	text := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	text.write("internal/p/p.go", emptyPackage+"\nimport \"errors\"\n\nfunc failed() error { return errors.New(\"Failed\") }\n")
	if refusal := refused(text, "error-text"); !strings.Contains(refusal, "internal/p/p.go:5") || !strings.Contains(refusal, "small letter") {
		t.Fatalf("the error text's refusal names no line or edit: %s", refusal)
	}
	text.write("internal/p/p.go", emptyPackage+"\nimport \"errors\"\n\nfunc failed() error { return errors.New(\"failed\") }\n")
	testcheck.NoError(t, text.candidate().rules(t.Context()))

	cuda := newRowFixture(t, map[string]string{"internal/q/q.go": "package q\n"})
	cuda.write("internal/q/q_test.go", "package q\n\nfunc init() { cudatest"+".Require(nil) }\n")
	if refusal := refused(cuda, "no lane runs"); !strings.Contains(refusal, `add "./internal/q" to cudaPackages`) {
		t.Fatalf("the CUDA lane's refusal names no edit: %s", refusal)
	}

	grown := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	grown.write("internal/p/p.go", oneFunction)
	testcheck.NoError(t, grown.candidate().rules(t.Context()))
	editPlan(t, grown, func(document *plan.Plan) { document.Items[0].Class = plan.ClassConsolidation })
	grown.git("add", "-A")
	grown.git("commit", "-q", "-m", "a consolidation")
	grown.write("internal/p/p.go", oneFunction+"\nfunc b() string { return \"b\" }\n")
	refused(grown, "edit: remove")
}
