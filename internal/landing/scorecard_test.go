package landing

import (
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/scorecard/metric"
)

// sliceFixture is a repository whose cmd/benchmark is a stub the slice
// builds and runs.
type sliceFixture struct {
	t    *testing.T
	repo string
}

// newSliceFixture commits a module holding the stub and a document outside
// the slice.
func newSliceFixture(t *testing.T, stub map[string]string) *sliceFixture {
	t.Helper()
	fixture := &sliceFixture{t: t, repo: t.TempDir()}
	fixture.write("go.mod", "module fixture\n\ngo 1.26\n")
	fixture.write("docs/readme.md", "fixture\n")
	for name, content := range stub {
		fixture.write(name, content)
	}
	fixture.git("init", "-q")
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "fixture")
	return fixture
}

func (f *sliceFixture) write(name, content string) {
	f.t.Helper()
	path := filepath.Join(f.repo, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		f.t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		f.t.Fatal(err)
	}
}

func (f *sliceFixture) git(arguments ...string) string {
	f.t.Helper()
	command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...)
	command.Dir = f.repo
	output, err := command.CombinedOutput()
	if err != nil {
		f.t.Fatalf("git %v: %v: %s", arguments, err, output)
	}
	return strings.TrimSpace(string(output))
}

// land runs the slice stage for a change to one path.
func (f *sliceFixture) land(changed string) *landing {
	f.t.Helper()
	l := &landing{Options: Options{Repo: f.repo, Ref: "row/step", Message: []byte("change"), Log: io.Discard},
		paths: []string{changed}, details: map[string]string{}}
	if err := l.scorecard(f.t.Context()); err != nil {
		f.t.Fatalf("the slice refused a landing: %v", err)
	}
	return l
}

// commit lands the tree as the gate would and returns the short commit.
func (f *sliceFixture) commit(l *landing) string {
	f.t.Helper()
	f.git("add", "-A")
	f.git("commit", "-q", "--allow-empty", "-m", l.message())
	return f.git("rev-parse", "--short", "HEAD")
}

// sliceStub stands in for the benchmark's slice mode. It reads what to do
// from cmd/benchmark/result.json: a record to measure, with "then" panicking
// or failing after it selects; or busy.
const sliceStub = `package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func main() {
	data, _ := os.ReadFile("cmd/benchmark/result.json")
	behaviour := strings.TrimSpace(string(data))
	write := func(record map[string]any) {
		record["slice"] = "inference"
		data, _ := json.Marshal(record)
		fmt.Println(string(data))
	}
	if behaviour == "busy" {
		write(map[string]any{"unmeasured": "unavailable: device busy"})
		return
	}
	var record map[string]any
	json.Unmarshal([]byte(behaviour), &record)
	write(map[string]any{"identity": record["identity"], "model": record["model"]})
	then, _ := record["then"].(string)
	delete(record, "then")
	switch {
	case then == "panic":
		panic(fmt.Sprint(record["model"], " broke"))
	case strings.HasPrefix(then, "fail: "):
		write(map[string]any{"identity": record["identity"], "model": record["model"],
			"failure": strings.TrimPrefix(then, "fail: "), "metrics": map[string]float64{"failure-rate": 1}})
	default:
		write(record)
	}
}
`

// sliceResult is a stub result: a record the stub measures, and what it
// does after selecting.
func sliceResult(identity, outputs, then string) string {
	data, _ := json.Marshal(map[string]any{"identity": identity, "model": "small", "outputs": outputs, "then": then,
		"metrics": map[string]float64{"first-output": 150, "throughput": 450, "load-time": 4000, "peak-device-memory": 100 * bytesPerMiB, "failure-rate": 0}})
	return string(data)
}

// TestLandingMeasuresTheScorecardSlice runs the inference slice through a
// stub benchmark in a fixture repository. A measurement writes one trailer
// and is compared by identity with the newest first-parent landing that
// measured it, including one from before slices were named; its timed cells
// are reported as measured. A failure or a crash is reported with its cause
// and writes a trailer, a claimed device is reported unavailable, and none
// of them refuses the landing. A change outside the slice runs nothing.
func TestLandingMeasuresTheScorecardSlice(t *testing.T) {
	t.Parallel()
	const result = "cmd/benchmark/result.json"
	fixture := newSliceFixture(t, map[string]string{
		"cmd/benchmark/main.go": sliceStub,
		sliceDocument:           `{"task":"inference"}`,
		result:                  sliceResult("inf-1", "a", ""),
	})
	expect := func(name string, l *landing, wants ...string) {
		t.Helper()
		for _, want := range wants {
			if detail := l.details["scorecard"]; !strings.Contains(detail, want) {
				t.Fatalf("%s: slice = %q, want it to hold %q", name, detail, want)
			}
		}
	}

	first := fixture.land(result)
	expect("first", first, "inference: small, no earlier slice of this identity: outputs recorded; peak-device-memory 100.0 MiB; first-output 150 ms, throughput 450 tok/s, load-time 4000 ms")
	if len(first.trailers) != 1 || !strings.Contains(first.trailers[0], `"identity":"inf-1"`) {
		t.Fatalf("first: trailers = %q, want the measured slice's", first.trailers)
	}
	firstCommit := fixture.commit(first)
	expect("unchanged", fixture.land(result), "inference: small, against "+firstCommit+": outputs identical; peak-device-memory 100.0 MiB (+0.0)")
	fixture.write(result, sliceResult("inf-1", "b", ""))
	changed := fixture.land(result)
	expect("changed", changed, "against "+firstCommit+": outputs differ")
	changedCommit := fixture.commit(changed)

	// A trailer from before slices were named is found by its identity; a
	// merged branch's landing is not the baseline.
	fixture.git("commit", "-q", "--allow-empty", "-m", "legacy\n\nScorecard-Slice: "+strings.ReplaceAll(sliceResult("inf-2", "z", ""), `,"then":""`, "")+"\nLanded-Row: old/step")
	legacyCommit := fixture.git("rev-parse", "--short", "HEAD")
	fixture.write(result, sliceResult("inf-2", "z", ""))
	expect("legacy", fixture.land(result), "against "+legacyCommit+": outputs identical")
	fixture.write(result, sliceResult("inf-1", "b", ""))
	fixture.git("checkout", "-q", "-b", "side")
	fixture.write(result, sliceResult("inf-1", "side", ""))
	fixture.commit(fixture.land(result))
	fixture.git("checkout", "-q", "-")
	fixture.git("merge", "-q", "--no-ff", "-m", "merge side", "side")
	fixture.write(result, sliceResult("inf-1", "b", ""))
	expect("merged", fixture.land(result), "against "+changedCommit+": outputs identical")
	fixture.write(result, sliceResult("inf-1", metric.VariedOutputs, ""))
	varied := fixture.land(result)
	expect("varied", varied, "outputs vary between runs")
	variedCommit := fixture.commit(varied)
	fixture.write(result, sliceResult("inf-1", "b", ""))
	expect("after varied", fixture.land(result), "against "+variedCommit+": outputs not compared: the baseline's varied")

	// Failures and a claimed device are reported, never refused.
	fixture.write(result, sliceResult("inf-1", "b", "fail: bad input"))
	failed := fixture.land(result)
	expect("failed", failed, "inference: small fails (measured at "+variedCommit+"): bad input")
	if len(failed.trailers) != 1 || !strings.Contains(failed.trailers[0], `"failure":"bad input"`) {
		t.Fatalf("failed: trailers = %q, want the failed slice's", failed.trailers)
	}
	fixture.write(result, sliceResult("inf-1", "b", "panic"))
	expect("crash", fixture.land(result), "inference: small fails (measured at "+variedCommit+"): exit 2: panic: small broke")
	fixture.write(result, "busy")
	if busy := fixture.land(result); !strings.Contains(busy.details["scorecard"], "inference: unavailable: device busy") || len(busy.trailers) != 0 {
		t.Fatalf("busy: slice = %q with trailers %q", busy.details["scorecard"], busy.trailers)
	}

	if l := fixture.land("docs/readme.md"); l.details["scorecard"] != "not reached" || len(l.trailers) != 0 {
		t.Fatalf("outside: slice = %q with trailers %q", l.details["scorecard"], l.trailers)
	}
}
