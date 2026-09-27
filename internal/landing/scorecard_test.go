package landing

import (
	"fmt"
	"io"
	"math"
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

// land runs the slice stage for a change to one path under the message.
func (f *sliceFixture) land(changed, message string) (*landing, error) {
	l := &landing{Options: Options{Repo: f.repo, Ref: "row/step", Message: []byte(message), Log: io.Discard},
		paths: []string{changed}, details: map[string]string{}}
	return l, l.scorecard(f.t.Context())
}

// commit lands the tree as the gate would and returns the short commit.
func (f *sliceFixture) commit(l *landing) string {
	f.t.Helper()
	f.git("add", "-A")
	f.git("commit", "-q", "-m", l.message())
	return f.git("rev-parse", "--short", "HEAD")
}

// sliceStub stands in for the benchmark: it prints the measurement a case
// wrote, panics on "panic", and with none exits as a claimed device does,
// its error in the file the landing names and progress on standard error.
const sliceStub = `package main

import "os"

func main() {
	data, err := os.ReadFile("cmd/benchmark/result.json")
	if err != nil {
		os.WriteFile(os.Getenv("OVERGO_ERROR_FILE"), []byte("benchmark: reserve device 0: device busy"), 0o600)
		os.Stderr.WriteString("loading model\n")
		os.Exit(75)
	}
	if string(data) == "panic" {
		var model map[string]int
		model["weights"]++
	}
	os.Stdout.Write(data)
}
`

// TestLandingMeasuresTheScorecardSlice runs the slice through a stub
// benchmark in a fixture repository. A change that reaches the benchmark
// measures it and compares its exact cells with the newest first-parent
// landing of the same identity, past newer landings of another identity and
// a merged branch's; a change that does not reach it, a claimed device or a
// crash on HEAD and the tree alike measures nothing, refuses nothing and
// says which it was.
func TestLandingMeasuresTheScorecardSlice(t *testing.T) {
	t.Parallel()
	const result = "cmd/benchmark/result.json"
	fixture := newSliceFixture(t, map[string]string{"cmd/benchmark/main.go": sliceStub})
	measurement := func(model string, peakMiB int, outputs ...string) string {
		var runs []string
		for _, output := range outputs {
			runs = append(runs, fmt.Sprintf(`{"output_digests":[%q]}`, output))
		}
		return fmt.Sprintf(`{"model_id":%q,"recipe_id":"recipe","workload_digest":"workload","options_digest":"options",`+
			`"environment_id":"environment","model_name":%q,"load_ms":4000,"device_peak_bytes":%d,`+
			`"summary":{"ttft_ms_p50":150,"decode_tokens_per_second_p50":450},"runs":[%s]}`,
			model, model, peakMiB*bytesPerMiB, strings.Join(runs, ","))
	}
	measure := func(name, measured string) *landing {
		t.Helper()
		fixture.write(result, measured)
		l, err := fixture.land(result, "change")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !strings.HasPrefix(l.trailer, sliceTrailer+": {") {
			t.Fatalf("%s: trailer = %q, want the slice", name, l.trailer)
		}
		return l
	}
	expect := func(name string, l *landing, want string) {
		t.Helper()
		if detail := l.details["scorecard"]; !strings.Contains(detail, want) {
			t.Fatalf("%s: slice = %q, want it to hold %q", name, detail, want)
		}
	}

	first := measure("first", measurement("small", 100, "a", "a"))
	expect("first", first, "small, no earlier slice of this identity: outputs recorded; peak-device-memory 100.0 MiB; "+
		"unpaired first-output 150 ms, throughput 450 tok/s, load-time 4000 ms")
	firstCommit := fixture.commit(first)
	changed := measure("changed", measurement("small", 102, "b", "b"))
	expect("changed", changed, "against "+firstCommit+": outputs differ; peak-device-memory 102.0 MiB (+2.0)")
	changedCommit := fixture.commit(changed)
	other := measure("other model", measurement("other", 100, "a", "a"))
	expect("other model", other, "other, no earlier slice of this identity")
	fixture.commit(other)
	// A merged branch's newer slice of the same identity is not the
	// mainline's baseline.
	fixture.git("checkout", "-q", "-b", "side")
	fixture.commit(measure("side", measurement("small", 200, "z", "z")))
	fixture.git("checkout", "-q", "-")
	fixture.git("merge", "-q", "--no-ff", "-m", "merge side", "side")
	unchanged := measure("unchanged", measurement("small", 102, "b", "b"))
	expect("unchanged", unchanged, "against "+changedCommit+": outputs identical; peak-device-memory 102.0 MiB (+0.0)")
	varied := measure("varied", measurement("small", 102, "b", "c"))
	expect("varied", varied, "outputs vary between runs")
	variedCommit := fixture.commit(varied)
	stable := measure("stable", measurement("small", 102, "b", "b"))
	expect("stable", stable, "against "+variedCommit+": outputs not compared: the baseline's varied")
	fixture.git("checkout", "-q", "--", result)

	for _, missing := range []struct {
		name, changed, content, want string
	}{
		{"outside", "docs/readme.md", "", "not reached"},
		{"busy", result, "", "unavailable: benchmark: reserve device 0: device busy"},
		{"crash on both", result, "panic", "failed: exit 2: panic: assignment to entry in nil map"},
	} {
		path := filepath.Join(fixture.repo, filepath.FromSlash(result))
		if missing.content == "" {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				t.Fatal(err)
			}
		} else {
			fixture.write(result, missing.content)
		}
		l, err := fixture.land(missing.changed, "change")
		if err != nil || l.details["scorecard"] != missing.want || l.trailer != "" {
			t.Fatalf("%s: slice = %q with trailer %q and error %v, want %q, none and none",
				missing.name, l.details["scorecard"], l.trailer, err, missing.want)
		}
	}
}

// pairedStub is a benchmark whose timed cells, crash and claim are compiled
// in, so HEAD's build and the tree's measure differently. Each cell
// alternates between two values from one run to the next, so same-binary
// runs have noise. It counts its runs, and with crashLater it crashes from
// its second run on.
const pairedStub = `package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if busy {
		os.WriteFile(os.Getenv("OVERGO_ERROR_FILE"), []byte("benchmark: reserve device 0: device busy"), 0o600)
		os.Exit(75)
	}
	if crash {
		var model map[string]int
		model["weights"]++
	}
	if crashLater {
		if _, err := os.Stat(".git/slice-crash"); err == nil {
			var model map[string]int
			model["weights"]++
		}
		os.WriteFile(".git/slice-crash", nil, 0o600)
	}
	count, _ := os.ReadFile(".git/slice-runs")
	runs, _ := strconv.Atoi(string(count))
	os.WriteFile(".git/slice-runs", []byte(strconv.Itoa(runs+1)), 0o600)
	jitter := runs % 2
	fmt.Printf(` + "`" + `{"model_id":"small","recipe_id":"recipe","workload_digest":"workload","options_digest":"options",` +
	`"environment_id":"environment","model_name":"small","load_ms":%d,"device_peak_bytes":1048576,` +
	`"summary":{"ttft_ms_p50":%d,"decode_tokens_per_second_p50":%d},"runs":[{"output_digests":["a"]}]}` + "`" + `,
		load+jitter*80, firstOutput+jitter*3, throughput+jitter*10)
}
`

// pairedBuild is what the stub compiles in.
type pairedBuild struct {
	throughput, firstOutput, load int
	crash, crashLater, busy       bool
}

func (b pairedBuild) source() string {
	return fmt.Sprintf("package main\n\nvar throughput, firstOutput, load = %d, %d, %d\n\nconst crash, crashLater, busy = %t, %t, %t\n",
		b.throughput, b.firstOutput, b.load, b.crash, b.crashLater, b.busy)
}

// TestLandingPairsPredictedTimedCells holds the paired slice to its
// contract. A predicted effect that names a timed metric runs HEAD's
// benchmark beside the tree's for every A B B A round: a named metric that
// worsens beyond its noise refuses the landing, in either direction a metric
// counts as better; an improvement, an unchanged build or a worsening within
// the noise lands with the paired change. A message that names a timed
// metric only outside its predicted effect runs nothing paired. A tree whose
// benchmark crashes where HEAD's measures is refused, whether the first run
// or a paired one crashes; a tree whose device is claimed is not. HEAD's
// build leaves out a file the change adds.
func TestLandingPairsPredictedTimedCells(t *testing.T) {
	t.Parallel()
	const speed, boost = "cmd/benchmark/speed.go", "cmd/benchmark/boost.go"
	base := pairedBuild{throughput: 450, firstOutput: 150, load: 4000}
	with := func(change func(*pairedBuild)) pairedBuild {
		build := base
		change(&build)
		return build
	}
	fixture := newSliceFixture(t, map[string]string{"cmd/benchmark/main.go": pairedStub, speed: base.source()})
	runs := func() int {
		count, err := os.ReadFile(filepath.Join(fixture.repo, ".git", "slice-runs"))
		if err != nil {
			return 0
		}
		value := 0
		fmt.Sscan(string(count), &value)
		return value
	}
	predicts := func(prediction string) string {
		return "change\n\nCause: a faster decode.\nPredicted effect: " + prediction + "\n"
	}
	const unpredicted = "change\n\nCause: a message's Predicted effect: section relabels the throughput cell.\nPredicted effect: none.\n"
	paired := 1 + len(abba)*pairedRounds
	for _, change := range []struct {
		name, message, source, changed string
		runs                           int
		refused, want                  string
	}{
		{name: "slower", message: predicts("throughput rises."), source: with(func(b *pairedBuild) { b.throughput = 400 }).source(),
			refused: "throughput -11.0% (noise 2.5%)"},
		{name: "faster", message: predicts("throughput rises."), source: with(func(b *pairedBuild) { b.throughput = 500 }).source(),
			runs: paired, want: "paired over 4 rounds: throughput +11.0% (noise 2.2%)"},
		{name: "unchanged", message: predicts("throughput holds."), source: base.source() + "\n// A comment.\n",
			runs: paired, want: "paired over 4 rounds: throughput +0.0% (noise 2.2%)"},
		{name: "within noise", message: predicts("throughput holds."), source: with(func(b *pairedBuild) { b.throughput = 448 }).source(),
			runs: paired, want: "throughput -0.4% (noise 2.2%)"},
		{name: "named outside the prediction", message: unpredicted, source: with(func(b *pairedBuild) { b.throughput = 400 }).source(),
			runs: 1, want: "unpaired first-output"},
		{name: "first output slower", message: predicts("first-output falls."), source: with(func(b *pairedBuild) { b.firstOutput = 180 }).source(),
			refused: "first-output +19.8% (noise 2.0%)"},
		{name: "load faster", message: predicts("load-time falls."), source: with(func(b *pairedBuild) { b.load = 3000 }).source(),
			runs: paired, want: "load-time -24.8% (noise 2.6%)"},
		{name: "second named metric slower", message: predicts("throughput holds and load-time falls."),
			source: with(func(b *pairedBuild) { b.load = 5000 }).source(), refused: "names load-time +24.8% (noise 2.0%), which"},
		{name: "crash", message: unpredicted, source: with(func(b *pairedBuild) { b.crash = true }).source(),
			refused: "the slice measures on HEAD and fails on the tree (failed: exit 2: panic: assignment to entry in nil map)"},
		{name: "crash while paired", message: predicts("throughput holds."), source: with(func(b *pairedBuild) { b.crashLater = true }).source(),
			refused: "the slice measures on HEAD and fails on the tree (failed: exit 2: panic: assignment to entry in nil map)"},
		{name: "claimed device", message: predicts("throughput holds."), source: with(func(b *pairedBuild) { b.busy = true }).source(),
			want: "unavailable: benchmark: reserve device 0: device busy"},
		{name: "added file", message: predicts("throughput holds."), source: base.source(), changed: boost,
			refused: "throughput -11.0% (noise 2.5%)"},
	} {
		fixture.write(speed, change.source)
		changed := speed
		if change.changed != "" {
			fixture.write(change.changed, "package main\n\nfunc init() { throughput = 400 }\n")
			changed = change.changed
		}
		if err := os.Remove(filepath.Join(fixture.repo, ".git", "slice-crash")); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		before := runs()
		l, err := fixture.land(changed, change.message)
		switch {
		case change.refused != "":
			if err == nil || !strings.Contains(err.Error(), change.refused) || strings.Contains(err.Error(), "throughput") && !strings.Contains(change.refused, "throughput") {
				t.Fatalf("%s: refusal = %v, want one naming only %q", change.name, err, change.refused)
			}
		case err != nil:
			t.Fatalf("%s: refused: %v", change.name, err)
		case !strings.Contains(l.details["scorecard"], change.want):
			t.Fatalf("%s: slice = %q, want it to hold %q", change.name, l.details["scorecard"], change.want)
		case change.runs != 0 && runs()-before != change.runs:
			t.Fatalf("%s: the benchmark ran %d times, want %d", change.name, runs()-before, change.runs)
		}
		if change.changed != "" {
			if err := os.Remove(filepath.Join(fixture.repo, filepath.FromSlash(change.changed))); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// TestPairedChangeIsMeanAgainstRange pins the paired statistic: the change
// between the sides' means, so one slow run moves it, and the noise as the
// wider side's range relative to its mean, so a slowdown confined to one
// side's runs widens the noise with it.
func TestPairedChangeIsMeanAgainstRange(t *testing.T) {
	t.Parallel()
	records := func(values ...float64) []sliceRecord {
		var result []sliceRecord
		for _, value := range values {
			result = append(result, sliceRecord{Metrics: map[metric.Name]float64{metric.Throughput: value}})
		}
		return result
	}
	change, noise := pairedChange([sides][]sliceRecord{
		headSide: records(100, 100, 100, 100),
		treeSide: records(100, 100, 100, 140),
	}, metric.Throughput)
	if math.Abs(change-0.1) > 1e-9 || math.Abs(noise-40.0/110) > 1e-9 {
		t.Fatalf("change, noise = %v, %v; want the means' 10%% and the tree's range over its mean", change, noise)
	}
}
