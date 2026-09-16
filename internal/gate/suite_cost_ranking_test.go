package gate

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestSuiteCostRanking(t *testing.T) {
	t.Parallel()
	// Parent and child durations overlap. Failed assertions still consumed work.
	const events = `{"Action":"start","Package":"slow"}
{"Action":"run","Package":"slow","Test":"TestParent"}
{"Action":"run","Package":"slow","Test":"TestParent/child"}
{"Action":"pass","Package":"slow","Test":"TestParent/child","Elapsed":3}
{"Action":"pass","Package":"slow","Test":"TestParent","Elapsed":4}
{"Action":"run","Package":"slow","Test":"TestFailure"}
{"Action":"fail","Package":"slow","Test":"TestFailure","Elapsed":2}
{"Action":"run","Package":"slow","Test":"TestInterrupted"}
{"Action":"pass","Package":"slow","Test":"TestUnstarted","Elapsed":99}
{"Action":"run","Package":"slow","Test":"TestSkip"}
{"Action":"skip","Package":"slow","Test":"TestSkip","Elapsed":100}
{"Action":"fail","Package":"slow","Elapsed":10}
{"Action":"start","Package":"fast"}
{"Action":"run","Package":"fast","Test":"TestFast"}
{"Action":"pass","Package":"fast","Test":"TestFast","Elapsed":1}
{"Action":"pass","Package":"fast","Elapsed":2}
{"Action":"start","Package":"cheap"}
{"Action":"pass","Package":"cheap","Elapsed":0}
{"Action":"fail","ImportPath":"unstarted","Elapsed":500}
{"Action":"start","Package":"interrupted"}
`
	// Use ordered actual Go event timestamps; never synthesize elapsed from them.
	var stream strings.Builder
	stamp := time.Unix(0, 0).UTC()
	for line := range strings.SplitSeq(strings.TrimSpace(events), "\n") {
		fmt.Fprintf(&stream, "{\"Time\":%q,%s\n", stamp.Format(time.RFC3339Nano), strings.TrimPrefix(line, "{"))
		stamp = stamp.Add(time.Second)
	}
	report, err := testevidence.GoTestJSONReport(stream.String())
	if err != nil {
		t.Fatal(err)
	}
	if report.StreamDigest != fmt.Sprintf("%x", sha256.Sum256([]byte(stream.String()))) {
		t.Fatal("cost identity does not bind the consumed stream")
	}
	ranked := suiteCostRanking(report)
	if len(ranked) != 2 || ranked[0].Package != "slow" || ranked[1].Package != "fast" {
		t.Fatalf("ranking credited planned/interrupted packages: %+v", ranked)
	}
	slow := ranked[0]
	if slow.SummedTopLevelSeconds != 6 || slow.Elapsed == nil || *slow.Elapsed != 10 || slow.Unmeasured != 3 {
		t.Fatalf("overlap or unmeasured work counted as wall: %+v", slow)
	}
	if slow.ChildExecutionSeconds != nil || slow.AssertionSeconds != nil || slow.Tests[0].Name != "TestParent" || slow.Tests[1].Name != "TestParent/child" {
		t.Fatalf("guessed subprocess attribution or wrong rank: %+v", slow)
	}
	for _, test := range slow.Tests {
		if test.Elapsed != nil && (test.Start == "" || test.End == "" || test.Unknown != "") {
			t.Fatalf("ranked test lacks observed lifecycle: %+v", test)
		}
	}
	if !report.PackagePassed("fast") || report.PackagePassed("slow") {
		t.Fatal("costs changed verdict authority")
	}
	t.Run("durable identities and separate attempts", func(t *testing.T) {
		root := t.TempDir()
		store := mustGateValue(overgodb.Open(root))
		result := testutil.ArtifactID(t, artifact.KindEvidence, "retained gate result")
		batch := artifact.Batch{Key: "suite-cost/fixture", Artifacts: []artifact.Descriptor{{ID: result}}}
		g := &gateContext{}
		g.recordPackageExecution([]string{"slow", "fast", "unstarted"}, true, time.Minute, report, fmt.Errorf("observed failure"))
		g.recordPackageExecution([]string{"slow", "fast"}, false, time.Minute, report, nil)
		g.recordPackageExecution([]string{"unstarted-only"}, false, time.Second, testevidence.GoTestReport{}, fmt.Errorf("interrupted before execution"))
		g.testExecutions[2].Unobserved = []string{"unstarted-only"}
		if err := g.appendSuiteCost(&batch, result); err != nil {
			t.Fatal(err)
		}
		if len(batch.Contents) != 1 || len(batch.Lineage) != 1 || batch.Lineage[0].Parent != result {
			t.Fatal("unbound or duplicated cost record")
		}
		content := batch.Contents[0]
		mustGateValue(store.Commit(t.Context(), batch))
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		store = mustGateValue(overgodb.OpenReadOnly(root))
		defer store.Close()
		retained, found, err := artifact.ReadContent(t.Context(), store, content.Descriptor.ID)
		if err != nil || !found || retained.Validate() != nil {
			t.Fatalf("lost retained cost: %v", err)
		}
		var decoded struct {
			Result      artifact.ID `json:"result"`
			Invocations []struct {
				packageExecutionBatch
				Suites []suiteTestCost `json:"suites"`
			} `json:"invocations"`
		}
		if err := json.Unmarshal(retained.Data, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Result != result || len(decoded.Invocations) != 3 || !decoded.Invocations[0].Short || decoded.Invocations[1].Short || !decoded.Invocations[0].Failed || decoded.Invocations[1].Failed {
			t.Fatalf("result/profile/retry identity lost: %+v", decoded)
		}
		if !slices.Equal(decoded.Invocations[0].Requested, []string{"slow", "fast", "unstarted"}) || len(decoded.Invocations[0].Executions) != len(report.Executions) || decoded.Invocations[0].WallNS != uint64(time.Minute) {
			t.Fatal("observed invocation lost its scope, executions or wall")
		}
		interrupted := decoded.Invocations[2]
		if !interrupted.Failed || interrupted.Short || !slices.Equal(interrupted.Requested, []string{"unstarted-only"}) || !slices.Equal(interrupted.Unobserved, interrupted.Requested) || len(interrupted.Executions) != 0 || len(interrupted.Suites) != 0 || interrupted.WallNS != uint64(time.Second) {
			t.Fatalf("unstarted invocation lost or assigned measured work: %+v", interrupted)
		}
		if decoded.Invocations[0].StreamDigest != report.StreamDigest || decoded.Invocations[0].Suites[0].SummedTopLevelSeconds != 6 {
			t.Fatal("retained ranking changed")
		}
		retained.Data = slices.Clone(retained.Data)
		retained.Data[0] = '['
		if retained.Validate() == nil {
			t.Fatal("changed record kept original identity")
		}
		if len(compactAudit(g.audit)) != 1 || !strings.Contains(compactAudit(g.audit)[0], content.Descriptor.ID.String()) {
			t.Fatal("cost record is not discoverable")
		}
	})
	t.Run("invalid timing never supplies a measurement", func(t *testing.T) {
		const start = `{"Time":"2026-09-13T00:00:00Z","Action":"run","Package":"p","Test":"TestCost"}`
		const pass = `{"Time":"2026-09-13T00:00:01Z","Action":"pass","Package":"p","Test":"TestCost","Elapsed":1}`
		for _, invalid := range []string{
			pass,
			start,
			start + "\n" + start + "\n" + pass,
			start + "\n" + pass + "\n" + pass,
			start + "\n" + strings.Replace(pass, "pass", "skip", 1),
			start + "\n" + strings.Replace(pass, "00:00:01Z", "bad-time", 1),
			start + "\n" + strings.Replace(pass, "2026", "2025", 1),
			start + "\n" + strings.Replace(pass, `,"Elapsed":1`, "", 1),
			start + "\n" + strings.Replace(pass, `"Elapsed":1`, `"Elapsed":-1`, 1),
		} {
			observed, err := testevidence.GoTestJSONReport(invalid)
			if err != nil {
				t.Fatal(err)
			}
			costs := observed.TestCosts()
			if len(costs) != 1 || costs[0].Elapsed != nil || costs[0].Unknown == "" {
				t.Fatalf("invalid lifecycle measured: %+v", costs)
			}
		}
		zero, err := testevidence.GoTestJSONReport(start + "\n" + strings.Replace(pass, `"Elapsed":1`, `"Elapsed":0`, 1))
		if err != nil || zero.TestCosts()[0].Elapsed == nil || *zero.TestCosts()[0].Elapsed != 0 {
			t.Fatal("observed zero became unknown")
		}
	})
}
