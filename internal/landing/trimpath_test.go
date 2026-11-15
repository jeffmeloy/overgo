package landing

import (
	"io"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// trimmedTest fails unless its test binary was built with -trimpath, as the build records it.
const trimmedTest = "package p\n\nimport (\n\t\"runtime/debug\"\n\t\"testing\"\n)\n\nfunc TestTrimmed(t *testing.T) {\n" +
	"\tinfo, _ := debug.ReadBuildInfo()\n\tfor _, setting := range info.Settings {\n" +
	"\t\tif setting.Key == \"-trimpath\" && setting.Value == \"true\" {\n\t\t\treturn\n\t\t}\n\t}\n" +
	"\tt.Fatal(\"built without -trimpath\")\n}\n"

// TestALandingBuildsWithTrimmedPaths holds a landing's go commands to trimmed builds (gate-trimpath/do): go keys a
// build without -trimpath on the package's directory, so each tree a landing made fresh, its undo run's among them,
// built what it checked anew (2026-11-15: 56 packages for cmd/plan in each fresh checkout, none in the second with
// -trimpath) and the build cache grew past 235 GB. A landing whose check is a test failing unless its binary was built
// trimmed lands, with no GOFLAGS set around it.
func TestALandingBuildsWithTrimmedPaths(t *testing.T) {
	t.Setenv("GOFLAGS", "")
	fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue})
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Verify = "go test ./p -run '^TestTrimmed$' -count=1"
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "the row's check")
	fixture.write("p/p_test.go", trimmedTest)
	testcheck.Must(Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: []byte("A test of trimmed builds."), Log: io.Discard}))(t)
}
