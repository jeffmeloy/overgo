package landing

import (
	"bytes"
	"strings"
	"testing"
)

// laterEditLog writes edit into the checkout as the landing's format stage begins: after its first capture, before
// the stage that captures again.
type laterEditLog struct {
	edit func()
	said *strings.Builder
}

func (l laterEditLog) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte(": format\n")) {
		l.edit()
	}
	l.said.Write(p)
	return len(p), nil
}

// TestALandingPublishesTheTreeItFirstCaptured holds a landing to the tree it first captured (landing-snapshot/do,
// SQA-494). The change rewrites internal/p/p.go unformatted. After the first capture and before the format stage
// rewrites anything, another file is written in the same package and another edited beside it. The commit holds the
// change as first captured with the gate's own formatting applied, and neither later edit; both stay in the checkout as
// written, named as left for their own step. Every capture had taken the checkout afresh, so moss-page/restored
// published moss-page/own-run's edit, and own-run was retired.
func TestALandingPublishesTheTreeItFirstCaptured(t *testing.T) {
	t.Parallel()
	fixture := newCheckedFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n", "docs/notes.txt": "base\n"}, shippedCheck)
	fixture.git("config", "core.autocrlf", "false")
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string {return \"shipped\"}\n")
	const later = "package p\n\nfunc b() string { return \"the next step's\" }\n"
	var said strings.Builder
	log := laterEditLog{said: &said, edit: func() {
		fixture.write("internal/p/later.go", later)
		fixture.write("docs/notes.txt", "the next step's notes\n")
	}}
	if _, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: log,
		Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")}); err != nil {
		t.Fatalf("the landing refused: %v\n%s", err, said.String())
	}
	if committed := fixture.git("show", "HEAD:internal/p/p.go"); committed != "package p\n\nfunc a() string { return \"shipped\" }" {
		t.Errorf("the commit holds p.go as %q; want the change as first captured, formatted by the gate", committed)
	}
	if listed := fixture.git("ls-tree", "--name-only", "-r", "HEAD"); strings.Contains(listed, "later.go") {
		t.Errorf("the commit holds %s; want the file written after its first capture left out", listed)
	}
	if notes := fixture.git("show", "HEAD:docs/notes.txt"); notes != "base" {
		t.Errorf("the commit holds the notes as %q; want them as first captured", notes)
	}
	if fixture.read("internal/p/later.go") != later || fixture.read("docs/notes.txt") != "the next step's notes\n" {
		t.Error("the edits written after the first capture were not left in the checkout as written")
	}
	if !strings.Contains(said.String(), "left in the checkout for its own step: ") || !strings.Contains(said.String(), "internal/p/later.go") {
		t.Errorf("the landing said\n%s\nwant the later edits named as left for their own step", said.String())
	}
}
