package landing

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
)

// TestCheckingAStepJudgesItsCheckAsTheAttemptDoes holds JudgeCheck, which plan -verify runs after the check passes on
// the change, to the attempt's own judgment of the check over the working tree (verify-as-gate): a check that passes
// with the change's production Go restored is refused as the attempt refuses it, and so is one naming a package that
// holds no test it runs; a check that fails without the change is taken, the undo run said; and a check the caller
// passed already is not run again, while the attempt's check the caller did not run is run on the change first.
func TestCheckingAStepJudgesItsCheckAsTheAttemptDoes(t *testing.T) {
	t.Parallel()
	judged := func(t *testing.T, check, source, test string, passed map[string]time.Duration) (string, error) {
		t.Helper()
		fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue, "q/q.go": "// Package q is a fixture.\npackage q\n"})
		editPlan(t, fixture, func(document *plan.Plan) {
			document.Items[0].Budget.Nodes = 100
			document.Items[0].Steps[0].Verify = check
		})
		fixture.git("add", "-A")
		fixture.git("commit", "-q", "-m", "the row's check")
		fixture.write("p/p.go", source)
		fixture.write("p/p_test.go", "package p\n\nimport \"testing\"\n\n"+test)
		var log bytes.Buffer
		err := JudgeCheck(t.Context(), fixture.repo, "row/do", passed, &log)
		return log.String(), err
	}
	changed := strings.Replace(undoneValue, "return 1", "return 2", 1)
	ran := map[string]time.Duration{undoneCheck: time.Second}
	t.Run("a check that passes without the change", func(t *testing.T) {
		t.Parallel()
		_, err := judged(t, undoneCheck, undoneValue+twice, "func TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n", ran)
		if err == nil || !strings.Contains(err.Error(), "proves nothing") {
			t.Fatalf("a check that passes with the change's production Go restored was judged %v; want it refused", err)
		}
	})
	t.Run("a package without the check's test", func(t *testing.T) {
		t.Parallel()
		check := "go test ./p ./q -run '^TestValue$' -count=1"
		_, err := judged(t, check, changed, caughtTest, map[string]time.Duration{check: time.Second})
		if err == nil || !strings.Contains(err.Error(), "./q holds no test") {
			t.Fatalf("a check naming ./q, which holds no TestValue, was judged %v; want it refused naming ./q", err)
		}
	})
	t.Run("a check that catches the change", func(t *testing.T) {
		t.Parallel()
		log, err := judged(t, undoneCheck, changed, caughtTest, ran)
		if err != nil || !strings.Contains(log, undoSection+"\n- with p/p.go restored to ") || strings.Contains(log, "verify: pass\tfixture/p") {
			t.Fatalf("a check that fails without the change was judged %v, saying:\n%s\nwant it taken, the undo run said, the check not run again on the change", err, log)
		}
	})
	t.Run("the attempt's check the caller did not run", func(t *testing.T) {
		t.Parallel()
		log, err := judged(t, undoneCheck, changed, caughtTest, nil)
		if err != nil || !strings.Contains(log, "verify: pass\tfixture/p") || !strings.Contains(log, "with p/p.go restored to ") {
			t.Fatalf("the attempt's check not run by the caller was judged %v, saying:\n%s\nwant it run on the change, then without it", err, log)
		}
	})
}
