package landing

import (
	"io"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestALandingLowersEveryCeilingOfAFile holds a landing to every ceiling its change's counts fell under in one file
// (SQA-399): under a fixture whose ceiling test holds two ceilings in one grouped const block, aligned since their names
// differ in length, a change that shortens internal/p makes both fall, and the commit holds both lowered; where the
// test also says a fall of a constant the file does not hold, the landing refuses, naming it, and the file is as it was,
// with neither ceiling written. The lowering had replaced the text "const NAME = N", which a grouped block holds
// nowhere, so the block's falls refused; and a second fall in one file refused after the first had written.
func TestALandingLowersEveryCeilingOfAFile(t *testing.T) {
	t.Parallel()
	const ceilingFile = "internal/repoanalysis/ceiling_test.go"
	// The falls said as testevidence.CeilingFell says them, written out so the check builds without the change.
	fell := func(name string, ceiling, count int) string {
		return strconv.Quote("ceiling fell: " + name + " of " + ceilingFile + " from " + strconv.Itoa(ceiling) + " to " + strconv.Itoa(count))
	}
	ceilings := func(extra string) string {
		return "package repoanalysis\n\nimport (\n\t\"os\"\n\t\"strings\"\n\t\"testing\"\n)\n\n" +
			"// The most lines and tabs internal/p may hold.\nconst (\n\tlineCeiling = 5\n\ttabCeiling  = 1\n)\n\n" +
			"func TestPHoldsToItsCeilings(t *testing.T) {\n" +
			"\tsource, err := os.ReadFile(\"../p/p.go\")\n\tif err != nil {\n\t\tt.Fatal(err)\n\t}\n" +
			"\tif strings.Count(string(source), \"\\n\") < lineCeiling {\n\t\tt.Log(" + fell("lineCeiling", 5, 3) + ")\n\t}\n" +
			"\tif strings.Count(string(source), \"\\t\") < tabCeiling {\n\t\tt.Log(" + fell("tabCeiling", 1, 0) + ")\n\t}\n" + extra + "}\n"
	}
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")
	land := func(ceiling string) (*sliceFixture, string, error) {
		fixture := newCheckedFixture(t, map[string]string{
			"internal/p/p.go": "package p\n\nfunc a() string {\n\treturn \"base\"\n}\n",
			ceilingFile:       ceiling,
		}, shippedCheck)
		fixture.git("config", "core.autocrlf", "false")
		editPlan(t, fixture, func(document *plan.Plan) { document.Items[0].Steps[0].Files = []string{"internal/p"} })
		fixture.git("add", "-A")
		fixture.git("commit", "-q", "-m", "a step that lists internal/p")
		fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")
		landed, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Message: message, Log: io.Discard})
		return fixture, landed, err
	}

	fixture, landed, err := land(ceilings(""))
	testcheck.NoError(t, err)
	if committed := fixture.git("show", landed+":"+ceilingFile); !strings.Contains(committed, "lineCeiling = 3") || !strings.Contains(committed, "tabCeiling  = 0") {
		t.Fatalf("the commit holds the ceilings as\n%s\nwant both lowered, lineCeiling to 3 and tabCeiling to 0", committed)
	}

	unheld := ceilings("\tt.Log(" + fell("missingCeiling", 7, 6) + ")\n")
	fixture, _, err = land(unheld)
	if err == nil || !strings.Contains(err.Error(), "missingCeiling") {
		t.Fatalf("a landing whose ceiling test says a fall of a constant its file does not hold ended %v, want it refused naming it", err)
	}
	if held := heldFile(fixture, ceilingFile); held != unheld {
		t.Fatalf("the refused landing wrote the ceilings it could lower:\n%s", held)
	}
}
