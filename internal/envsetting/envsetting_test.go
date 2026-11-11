// Package envsetting_test holds envsetting to what a count of the tree reads of it, through the identity's exported
// linker flag, so it builds whether or not the identity reads settings through envsetting.
package envsetting_test

import (
	"strings"
	"testing"

	"overgo/internal/inferencesurface"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestASettingReadThroughItsFunctionIsDeclared counts a tree whose capabilities read their settings through
// envsetting (setting-reader): on 2026-11-10 a read of OVERGO_DEVICE_BUDGET_GIB that the identity's lists did not hold
// left every identity reaching it uncounted, found by a lane test. A name read through Result joins the count of the
// package whose code reads it, and a name read through Neutral is counted as changing nothing, neither written into a
// list; a read through Result of a name no count can read, or a read of the environment of the code's own, leaves the
// package that makes it uncounted. It reads the counts from the linker flag a build is given.
func TestASettingReadThroughItsFunctionIsDeclared(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) { testutil.WriteTextFile(t, root, name, content) }
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("internal/mediacapability/catalog.go", "package mediacapability\n\nimport (\n\t_ \"fixture/internal/capabilityruntime\"\n\t_ \"fixture/internal/picture\"\n\t_ \"fixture/internal/speech\"\n)\n")
	write("internal/capabilityruntime/runtime.go", "package capabilityruntime\n")
	write("internal/envsetting/envsetting.go", "package envsetting\n\nimport \"os\"\n\nfunc Result(name string) string { return os.Getenv(name) }\n\nfunc Neutral(name, because string) string { return os.Getenv(name) }\n")
	picture := func(read string) {
		write("internal/picture/picture.go", "package picture\n\nimport (\n\t\"os\"\n\n\t\"fixture/internal/envsetting\"\n)\n\nvar name = \"FIXTURE_NAMED\"\n\nvar _, _ = os.Args, "+read+"\n")
	}
	picture(`envsetting.Result("FIXTURE_PRECISION")`)
	write("internal/speech/speech.go", "package speech\n\nimport \"fixture/internal/envsetting\"\n\nvar trace = envsetting.Neutral(\"FIXTURE_TRACE\", \"prints a trace\")\n")
	counts := func() map[string]string {
		t.Helper()
		flag := testcheck.Must(inferencesurface.LinkFlag(t.Context(), root))(t)
		_, stamped, found := strings.Cut(flag, ".stamp=")
		if !found {
			t.Fatalf("the linker flag %q sets no stamp", flag)
		}
		counted := map[string]string{}
		for entry := range strings.SplitSeq(stamped, ",") {
			pkg, count, _ := strings.Cut(entry, "=")
			counted[pkg] = count
		}
		return counted
	}
	counted := counts()
	if count := counted["fixture/internal/picture"]; !strings.HasSuffix(count, ";FIXTURE_PRECISION") {
		t.Errorf("a package that reads FIXTURE_PRECISION through Result is counted %q: want the name among its count's settings", count)
	}
	if count, held := counted["fixture/internal/speech"]; !held || strings.Contains(count, ";") || strings.Contains(count, "FIXTURE_TRACE") {
		t.Errorf("a package that reads FIXTURE_TRACE through Neutral is counted %q (%t): want it counted with no setting", count, held)
	}
	for _, read := range []string{"envsetting.Result(name)", `os.Getenv("FIXTURE_PRECISION")`} {
		picture(read)
		if count, held := counts()["fixture/internal/picture"]; held {
			t.Errorf("a package that reads %s is counted %q: want it uncounted, as nothing says what it depends on", read, count)
		}
	}
}
