package landing

import (
	"fmt"
	"io"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestALanesRefusalNamesTheLaneThatFailedFirst lands a change that reaches two lanes, a device lane whose test waits
// and a web UI lane beside it whose test fails (lane-cause/do). The refusal names the web UI lane and its failed test,
// not the device lane its failure cancelled, and the record's worklist counts the test. On 2026-11-13 the record held
// no-clocks/own-budgets' refusal as "lane device: ... context canceled", the lanes' failures joined in their own order,
// and guidance-ops/image-readings' as the CUDA lane's command line, cut before the package that failed.
func TestALanesRefusalNamesTheLaneThatFailedFirst(t *testing.T) {
	// Serial: it names the lane turn and the lanes the tests take, not the host's own.
	turn, gated := laneTurn, lanes
	laneTurn = func() (string, error) { return "overgo-test-lane-cause-turn", nil }
	running := func(test string) func(string) []string {
		return func(string) []string {
			return []string{"test", "-json", "-count=1", "./internal/p", "-run", "^" + test + "$"}
		}
	}
	lanes = []lane{
		{name: "device", packages: []string{"./internal/p"}, tags: "fixturelane", command: running("TestWaits")},
		{name: "webui", packages: []string{"./internal/p"}, tags: "fixturelane", command: running("TestFails"), hostOnly: true},
	}
	t.Cleanup(func() { laneTurn, lanes = turn, gated })
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n\nfunc a() string { return \"base\" }\n"})
	fixture.write("internal/p/p.go", "package p\n\nfunc a() string { return \"shipped\" }\n")
	fixture.write("internal/p/lane_test.go", "//go:build fixturelane\n\npackage p\n\nimport (\n\t\"testing\"\n\t\"time\"\n)\n\n"+
		"func TestWaits(t *testing.T) { time.Sleep(time.Hour) }\n\nfunc TestFails(t *testing.T) { t.Fatal(\"the page showed nothing\") }\n")
	_, err := Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: io.Discard,
		Message: []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: first-output holds.\n")})
	const failed = "fixture/internal/p: TestFails"
	refusal := Classify("row/do", fmt.Sprint(err))
	if !strings.HasPrefix(refusal.Cause, "lanes: lane webui: failed tests/packages: "+failed) {
		t.Fatalf("the lanes refused with %q (%v), want the web UI lane and its failed test, not the device lane it cancelled", refusal.Cause, err)
	}
	root := t.TempDir()
	testcheck.Must(RecordRefusal(root, refusal))(t)
	if listed := testcheck.Must(Worklist(root))(t); len(listed) != 1 || listed[0].By != failed {
		t.Errorf("the worklist lists the refusal as %+v, want it by %s", listed, failed)
	}
}
