package landing

import (
	"strings"
	"testing"
)

// TestDeriveRefusesToWriteOverAnEditItDoesNotTake holds derive to the edits a checkout holds of a declared file the
// change derives anew (SQA-403): a change to a file that is no Go, whose candidate's claims generator rebinds
// compatibility.json unlike the base, beside a compatibility.json edited for another step, is refused naming the
// file, and the edit stands; derive had written the candidate's whole file over it, publishing the change's binding
// and losing the edit. A file nothing declares, the API manifest, is derived again as before.
func TestDeriveRefusesToWriteOverAnEditItDoesNotTake(t *testing.T) {
	t.Parallel()
	const generator = "package main\n\nimport \"os\"\n\nfunc main() { os.WriteFile(\"compatibility.json\", []byte(\"rebound\\n\"), 0o644) }\n"
	fixture := newRowFixture(t, map[string]string{"cmd/validate/main.go": generator, "compatibility.json": "base\n", "notes.txt": "base\n"})
	fixture.write("compatibility.json", "declared\n")
	fixture.write("notes.txt", "changed\n")
	l := fixture.candidate()
	if err := l.fix(t.Context()); err == nil || !strings.Contains(err.Error(), "compatibility.json holds an edit this change does not take") {
		t.Fatalf("a change rebinding the claims beside a claim edit it does not take: %v; want it refused naming the file", err)
	}
	if held := fixture.read("compatibility.json"); held != "declared\n" {
		t.Fatalf("the checkout's compatibility.json is %q after the refusal; want the edit as it stood", held)
	}
}
