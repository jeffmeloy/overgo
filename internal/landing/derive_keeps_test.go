package landing

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestDeriveLeavesAnEditItDoesNotMake holds derive to the files the change derives anew (cycle-lanes/derive-keeps): a
// change to a file that is no Go, whose candidate derives the manifest as the base holds it, leaves the checkout's own
// copy of the manifest, another step's edit, as it stands, and commits the base's. Derive had written every file it
// derived into the checkout, and on 2026-11-14 a claim edited for a step not yet landed was written over by the
// landing before it.
func TestDeriveLeavesAnEditItDoesNotMake(t *testing.T) {
	t.Parallel()
	const generator = "package main\n\nimport \"os\"\n\nfunc main() { os.WriteFile(\"docs/api_manifest.json\", []byte(\"regenerated\\n\"), 0o644) }\n"
	fixture := newRowFixture(t, map[string]string{"cmd/api-manifest/main.go": generator, "docs/api_manifest.json": "regenerated\n", "notes.txt": "base\n"})
	fixture.write("docs/api_manifest.json", "declared\n")
	fixture.write("notes.txt", "changed\n")
	l := fixture.candidate()
	testcheck.NoError(t, l.fix(t.Context()))
	if held := fixture.read("docs/api_manifest.json"); held != "declared\n" {
		t.Fatalf("the checkout's manifest is %q after a change that derives it as the base holds it; want the edit left as it stood", held)
	}
	if taken := fixture.git("cat-file", "blob", l.candidate+":docs/api_manifest.json"); taken != "regenerated" {
		t.Fatalf("the candidate holds the manifest %q; want the base's", taken)
	}
}
