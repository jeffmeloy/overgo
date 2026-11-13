package landing

import (
	"io"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestALandingLowersACeilingItsChangeFellUnder holds a landing to writing a ceiling its change's count fell under into
// its commit, where the ceiling's test had refused it advising that very edit. The change shortens internal/p, which a
// ceiling test of internal/repoanalysis counts, from five lines to three; the step lists internal/p alone. While the
// checkout holds a change of its own to the ceiling's file, which the commit would leave out, the landing writes nothing
// over it and refuses with the edit to make; once that change is gone the commit carries internal/p and the ceiling of
// three.
func TestALandingLowersACeilingItsChangeFellUnder(t *testing.T) {
	t.Parallel()
	const ceilingFile = "internal/repoanalysis/ceiling_test.go"
	// The fall said as testevidence.CeilingFell says it, written out so the check builds without the change.
	const fall = "ceiling fell: lineCeiling of " + ceilingFile + " from 5 to 3"
	ceiling := "package repoanalysis\n\nimport (\n\t\"os\"\n\t\"strings\"\n\t\"testing\"\n)\n\n" +
		"// lineCeiling is the most lines internal/p may hold.\nconst lineCeiling = 5\n\n" +
		"func TestPHoldsToItsCeiling(t *testing.T) {\n" +
		"\tsource, err := os.ReadFile(\"../p/p.go\")\n\tif err != nil {\n\t\tt.Fatal(err)\n\t}\n" +
		"\tcount := strings.Count(string(source), \"\\n\")\n" +
		"\tif count > lineCeiling {\n\t\tt.Errorf(\"internal/p holds %d lines, over its ceiling\", count)\n\t}\n" +
		"\tif count < lineCeiling {\n\t\tt.Log(" + strconv.Quote(fall) + ")\n\t}\n}\n"
	fixture := newRowFixture(t, map[string]string{
		"internal/p/p.go": "package p\n\nfunc a() string {\n\treturn \"base\"\n}\n",
		ceilingFile:       ceiling,
	})
	fixture.git("config", "core.autocrlf", "false")
	editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"internal/p"} })
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "a step that lists internal/p")
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")

	another := strings.Replace(ceiling, "most lines", "most lines, another session's,", 1)
	fixture.write(ceilingFile, another)
	_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: io.Discard})
	if err == nil || !strings.Contains(err.Error(), "lineCeiling fell to 3 under its ceiling of 5") {
		t.Fatalf("a landing whose fall's file holds a change it leaves out ended as %v, want it refused naming the fall", err)
	}
	if held := heldFile(fixture, ceilingFile); held != another {
		t.Fatalf("the refused landing wrote over the change the checkout held:\n%s", held)
	}

	fixture.write(ceilingFile, ceiling)
	landed := testcheck.Must(Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: io.Discard}))(t)
	if !strings.Contains(fixture.git("show", landed+":internal/p/p.go"), "shipped") {
		t.Fatal("the landing did not commit the change to internal/p")
	}
	if committed := fixture.git("show", landed+":"+ceilingFile); !strings.Contains(committed, "const lineCeiling = 3") {
		t.Fatalf("the commit holds the ceiling as\n%s\nwant it lowered to 3", committed)
	}
	if status := fixture.git("status", "--porcelain", "--", ceilingFile); status != "" {
		t.Fatalf("the checkout's ceiling file differs from the commit's: %s", status)
	}
}
