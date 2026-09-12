package testevidence

import (
	"strings"
	"testing"
)

func TestPackageExecutionCostAcceptance(t *testing.T) {
	t.Parallel()
	const events = `{"Action":"start","Package":"good"}
{"Action":"run","Package":"good","Test":"TestGood"}
{"Action":"pass","Package":"good","Test":"TestGood","Elapsed":99}
{"Action":"pass","Package":"good","Elapsed":2.5}
{"Action":"start","Package":"bad"}
{"Action":"run","Package":"bad","Test":"TestBad"}
{"Action":"fail","Package":"bad","Test":"TestBad","Elapsed":1}
{"Action":"fail","Package":"bad","Elapsed":3}
{"Action":"start","Package":"interrupted"}
{"Action":"run","Package":"interrupted","Test":"TestPending"}
{"Action":"output","Package":"unstarted","Output":"compiler diagnostic"}
{"Action":"fail","ImportPath":"build-failed"}
{"Action":"skip","Package":"empty","Elapsed":0}
{"Action":"fail","Package":"invalid-time","Elapsed":-1}
`
	report, err := GoTestJSONReport(events)
	if err != nil {
		t.Fatal(err)
	}
	if !report.PackagePassed("good") || report.PackagePassed("bad") || report.PackagePassed("empty") || RequireComplete(report) == nil {
		t.Fatal("cost reporting changed evidence eligibility")
	}
	if len(report.Executions) != 6 {
		t.Fatalf("executions = %+v", report.Executions)
	}
	want := map[string]struct {
		action  string
		started bool
		elapsed *float64
	}{
		"good":         {"pass", true, new(2.5)},
		"bad":          {"fail", true, new(3.0)},
		"interrupted":  {"", true, nil},
		"build-failed": {"fail", false, nil},
		"empty":        {"skip", false, new(0.0)},
		"invalid-time": {"fail", false, nil},
	}
	for _, got := range report.Executions {
		expected, found := want[got.Package]
		if !found || got.Action != expected.action || got.Started != expected.started || (got.Elapsed == nil) != (expected.elapsed == nil) {
			t.Fatalf("lifecycle = %+v, expected %+v", got, expected)
		}
		if got.Elapsed != nil && *got.Elapsed != *expected.elapsed {
			t.Fatalf("package time includes test time: %+v", got)
		}
	}
	partial, err := GoTestJSONReader(strings.NewReader(events+"not JSON\n"), false, 1024, nil)
	if err == nil || len(partial.Executions) != len(report.Executions) || partial.PackagePassed("good") {
		t.Fatalf("malformed stream lost observations or retained invalid credit: %v, %+v", err, partial)
	}
}
