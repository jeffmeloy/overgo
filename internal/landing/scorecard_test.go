package landing

import (
	"encoding/json"
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

// sliceFixture is a repository whose cmd/benchmark is a stub the slices
// build and run.
type sliceFixture struct {
	t    *testing.T
	repo string
}

// newSliceFixture commits a module holding the stub and a document outside
// the slices.
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
	f.git("commit", "-q", "--allow-empty", "-m", l.message())
	return f.git("rev-parse", "--short", "HEAD")
}

// sliceStub stands in for the benchmark's slice mode. It logs each
// invocation's slices, then for each slice reads what to do from its
// compiled-in override, which differs between HEAD's build and the tree's,
// else from cmd/benchmark/results/<slice>.json: a record to measure, with
// "then" panicking or failing after it selects; or panic-early, busy or
// absent.
const sliceStub = `package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	var names []string
	for _, argument := range os.Args[2:] {
		names = append(names, strings.TrimSuffix(filepath.Base(argument), ".json"))
	}
	log, _ := os.OpenFile(".git/slice-invocations", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	fmt.Fprintln(log, strings.Join(names, " "))
	log.Close()
	for _, name := range names {
		behaviour, found := override[name]
		if !found {
			data, _ := os.ReadFile("cmd/benchmark/results/" + name + ".json")
			behaviour = strings.TrimSpace(string(data))
		}
		write := func(record map[string]any) {
			record["slice"] = name
			data, _ := json.Marshal(record)
			fmt.Println(string(data))
		}
		switch behaviour {
		case "panic-early":
			panic("before selecting " + name)
		case "busy":
			write(map[string]any{"unmeasured": "unavailable: device busy"})
			continue
		case "absent":
			write(map[string]any{"unmeasured": "absent: no model"})
			continue
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
}
`

// sliceResult is a stub result: a record the stub measures, and what it
// does after selecting.
func sliceResult(identity, model, outputs, then string, metrics map[string]float64) string {
	data, _ := json.Marshal(map[string]any{"identity": identity, "model": model, "outputs": outputs, "metrics": metrics, "then": then})
	return string(data)
}

// TestLandingMeasuresEachModalitySlice runs every slice document through
// one stub benchmark process in a fixture repository. The inference slice
// runs first, then the rest by name; each slice that measures or fails
// writes one trailer, compared by identity with the newest first-parent
// landing of that identity, including a trailer from before slices were
// named. A crash is the slice that selected and printed no outcome, and the
// slices after it still run; it is refused only when HEAD's benchmark
// measures the same identity, or selects and measures where the tree failed
// before selecting. HEAD runs only the failed slices, and a prediction
// pairs only the slice it names.
func TestLandingMeasuresEachModalitySlice(t *testing.T) {
	t.Parallel()
	inference := func(identity, outputs string) string {
		return sliceResult(identity, "small", outputs, "", map[string]float64{
			"first-output": 150, "throughput": 450, "load-time": 4000, "peak-device-memory": 100 * bytesPerMiB, "failure-rate": 0})
	}
	forecast := func(identity, outputs, then string) string {
		return sliceResult(identity, "timesfm", outputs, then, map[string]float64{"first-output": 20, "failure-rate": 0})
	}
	speech := func(outputs string) string {
		return sliceResult("sp-1", "pocket", outputs, "", map[string]float64{"first-output": 30, "failure-rate": 0})
	}
	const behaviour = "cmd/benchmark/behaviour.go"
	override := func(slices map[string]string) string {
		var source strings.Builder
		source.WriteString("package main\n\nvar override = map[string]string{\n")
		for name, value := range slices {
			source.WriteString(fmt.Sprintf("\t%q: %q,\n", name, value))
		}
		return source.String() + "}\n"
	}
	fixture := newSliceFixture(t, map[string]string{
		"cmd/benchmark/main.go": sliceStub,
		behaviour:               override(nil),
		"fixtures/scorecard_slice/inference.json":    `{"task":"inference"}`,
		"fixtures/scorecard_slice/forecast.json":     `{"task":"forecast"}`,
		"fixtures/scorecard_slice/speech.json":       `{"task":"speech"}`,
		"fixtures/scorecard_slice/image-gen.json":    `{"task":"image-gen"}`,
		"cmd/benchmark/results/inference.json":       inference("inf-1", "a"),
		"cmd/benchmark/results/forecast.json":        forecast("fc-1", "f", ""),
		"cmd/benchmark/results/speech.json":          speech("s1"),
		"cmd/benchmark/results/image-gen.json":       "absent",
		"fixtures/scorecard_slice/scratch/README.md": "not a document\n",
	})
	invocations := func() []string {
		t.Helper()
		path := filepath.Join(fixture.repo, ".git", "slice-invocations")
		data, _ := os.ReadFile(path)
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if strings.TrimSpace(string(data)) == "" {
			return nil
		}
		return strings.Split(strings.TrimSpace(string(data)), "\n")
	}
	land := func(name, changed, message string) *landing {
		t.Helper()
		invocations()
		l, err := fixture.land(changed, message)
		if err != nil {
			t.Fatalf("%s: refused: %v", name, err)
		}
		return l
	}
	expect := func(name string, l *landing, wants ...string) {
		t.Helper()
		for _, want := range wants {
			if detail := l.details["scorecard"]; !strings.Contains(detail, want) {
				t.Fatalf("%s: slices = %q, want them to hold %q", name, detail, want)
			}
		}
	}
	const document = "fixtures/scorecard_slice/speech.json"

	// Every slice in one process, inference first; an absent model writes
	// no trailer and is not counted.
	first := land("first", document, "change")
	if got := invocations(); strings.Join(got, "|") != "inference forecast image-gen speech" {
		t.Fatalf("first: invocations = %q, want one of every slice, inference first", got)
	}
	expect("first", first, "inference: small, no earlier slice of this identity: outputs recorded; peak-device-memory 100.0 MiB; unpaired first-output 150 ms, throughput 450 tok/s, load-time 4000 ms | forecast: timesfm, ",
		" | image-gen: absent: no model | speech: pocket, no earlier slice", "| failure-rate 0 of 3 slices")
	if len(first.trailers) != 3 || !strings.Contains(first.trailers[2], `"slice":"speech"`) {
		t.Fatalf("first: trailers = %q, want one per measured slice", first.trailers)
	}
	firstCommit := fixture.commit(first)

	// Each slice against its own identity's newest landing.
	fixture.write("cmd/benchmark/results/speech.json", speech("s2"))
	changed := land("changed", document, "change")
	expect("changed", changed, "inference: small, against "+firstCommit+": outputs identical; peak-device-memory 100.0 MiB (+0.0)",
		"speech: pocket, against "+firstCommit+": outputs differ")
	changedCommit := fixture.commit(changed)

	// A trailer from before slices were named is found by its identity; a
	// newer landing of another identity and a merged branch's are not the
	// baseline.
	fixture.git("commit", "-q", "--allow-empty", "-m", "legacy\n\nScorecard-Slice: "+strings.ReplaceAll(inference("inf-2", "z"), `,"then":""`, "")+"\nLanded-Row: old/step")
	legacyCommit := fixture.git("rev-parse", "--short", "HEAD")
	fixture.write("cmd/benchmark/results/inference.json", inference("inf-2", "z"))
	legacy := land("legacy", document, "change")
	expect("legacy", legacy, "inference: small, against "+legacyCommit+": outputs identical")
	fixture.git("checkout", "-q", "--", "cmd/benchmark/results/inference.json")
	fixture.git("checkout", "-q", "-b", "side")
	fixture.write("cmd/benchmark/results/speech.json", speech("side"))
	fixture.commit(land("side", document, "change"))
	fixture.git("checkout", "-q", "-")
	fixture.git("merge", "-q", "--no-ff", "-m", "merge side", "side")
	fixture.write("cmd/benchmark/results/speech.json", speech("s2"))
	merged := land("merged", document, "change")
	expect("merged", merged, "inference: small, against "+changedCommit+": outputs identical", "speech: pocket, against "+changedCommit+": outputs identical")
	fixture.write("cmd/benchmark/results/speech.json", speech(metric.VariedOutputs))
	varied := land("varied", document, "change")
	expect("varied", varied, "speech: pocket, against "+changedCommit+": outputs vary between runs")
	variedCommit := fixture.commit(varied)
	fixture.write("cmd/benchmark/results/speech.json", speech("s2"))
	expect("after varied", land("after varied", document, "change"), "speech: pocket, against "+variedCommit+": outputs not compared: the baseline's varied")

	// A failure on both sides lands as the failures cell; a later measure
	// reads the transition.
	fixture.write("cmd/benchmark/results/forecast.json", forecast("fc-1", "f", "fail: bad input"))
	both := land("both fail", document, "change")
	expect("both fail", both, "forecast: timesfm fails (measured at ", "): bad input", "| failure-rate 1 of 3 slices")
	if got := invocations(); strings.Join(got, "|") != "inference forecast image-gen speech|forecast" {
		t.Fatalf("both fail: invocations = %q, want the tree's and HEAD's failed slice", got)
	}
	if !strings.Contains(strings.Join(both.trailers, "\n"), `"failure":"bad input"`) {
		t.Fatalf("both fail: trailers = %q, want the failed slice's", both.trailers)
	}
	failedCommit := fixture.commit(both)
	expect("still failing", land("still failing", document, "change"), "forecast: timesfm still fails (as at "+failedCommit+"): bad input")
	fixture.write("cmd/benchmark/results/forecast.json", forecast("fc-1", "f", ""))
	// The exact cells compare with the newest measurement, past the failure.
	expect("recovered", land("recovered", document, "change"),
		"forecast: timesfm, measures (failed at "+failedCommit+"); against "+variedCommit+": outputs identical")

	// A crash the tree alone has, against HEAD's measure of the same
	// identity, is refused; the slices after it still run, and HEAD runs
	// only the crashed slice.
	for _, crash := range []struct {
		name, override, refused, note string
	}{
		{name: "crash", override: forecast("fc-1", "f", "panic"),
			refused: "the forecast slice measures on HEAD and fails on the tree (exit 2: panic: timesfm broke"},
		{name: "crash before selecting", override: "panic-early",
			refused: "the forecast slice selects timesfm and measures it on HEAD but fails on the tree before selecting a model (exit 2: panic: before selecting forecast"},
		{name: "crash under another identity", override: forecast("fc-tree", "f", "panic"),
			note: "forecast: timesfm fails: exit 2: panic: timesfm broke; not compared: HEAD measures timesfm under another identity"},
	} {
		fixture.write(behaviour, override(map[string]string{"forecast": crash.override}))
		invocations()
		l, err := fixture.land(behaviour, "change")
		got := invocations()
		if want := "inference forecast image-gen speech|image-gen speech|forecast"; strings.Join(got, "|") != want {
			t.Fatalf("%s: invocations = %q, want the tree's, the slices after the crash, and HEAD's crashed slice: %s", crash.name, got, want)
		}
		switch {
		case crash.refused != "" && (err == nil || !strings.Contains(err.Error(), crash.refused)):
			t.Fatalf("%s: refusal = %v, want %q", crash.name, err, crash.refused)
		case crash.note != "" && err != nil:
			t.Fatalf("%s: refused: %v", crash.name, err)
		case crash.note != "":
			expect(crash.name, l, crash.note, "speech: pocket, against ")
		}
	}
	fixture.write(behaviour, override(nil))

	// A claimed device is unavailable: no refusal, no HEAD build.
	fixture.write("cmd/benchmark/results/inference.json", "busy")
	busy := land("busy", document, "change")
	expect("busy", busy, "inference: unavailable: device busy | forecast: timesfm, ")
	if got := invocations(); len(got) != 1 {
		t.Fatalf("busy: invocations = %q, want the tree's alone", got)
	}
	fixture.git("checkout", "-q", "--", "cmd/benchmark/results/inference.json")

	// A prediction pairs only the slice it names; a bare metric names the
	// inference slice; a metric the slice does not measure is not paired.
	for _, prediction := range []struct {
		name, message, want, paired string
		pairings                    int
	}{
		{"named slice", "speech first-output holds.", "speech: pocket, against ", "speech", len(abba) * pairedRounds},
		{"bare metric", "first-output holds.", "inference: small, against ", "inference", len(abba) * pairedRounds},
		{"unmeasured metric", "speech throughput holds.", "not paired: speech measures no throughput", "", 0},
	} {
		l := land(prediction.name, document, "change\n\nCause: a change.\nPredicted effect: "+prediction.message+"\n")
		expect(prediction.name, l, prediction.want)
		got := invocations()
		if len(got) != 1+prediction.pairings {
			t.Fatalf("%s: %d invocations, want the slices once and %d paired runs: %q", prediction.name, len(got), prediction.pairings, got)
		}
		for _, invocation := range got[1:] {
			if invocation != prediction.paired {
				t.Fatalf("%s: a paired run measured %q, want only %s", prediction.name, invocation, prediction.paired)
			}
		}
	}

	// A change outside the slices runs none.
	if l := land("outside", "docs/readme.md", "change"); l.details["scorecard"] != "not reached" || len(l.trailers) != 0 {
		t.Fatalf("outside: slices = %q with trailers %q", l.details["scorecard"], l.trailers)
	}
	if got := invocations(); len(got) != 0 {
		t.Fatalf("outside: invocations = %q, want none", got)
	}
}

// pairedStub is a benchmark whose timed cells, crash and claim are compiled
// in, so HEAD's build and the tree's measure differently. Each cell
// alternates between two values from one run to the next, so same-binary
// runs have noise. It counts its measuring runs, and with crashLater it
// crashes from its second run on, selecting laterIdentity when one is set.
const pairedStub = `package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if busy {
		fmt.Println(` + "`" + `{"slice":"inference","unmeasured":"unavailable: benchmark: reserve device 0: device busy"}` + "`" + `)
		return
	}
	_, marked := os.Stat(".git/slice-crash")
	crashes := crash || crashLater && marked == nil
	identity := "small"
	if crashes && laterIdentity != "" {
		identity = laterIdentity
	}
	fmt.Printf(` + "`" + `{"slice":"inference","identity":%q,"model":"small"}` + "`" + `+"\n", identity)
	if crashes {
		var model map[string]int
		model["weights"]++
	}
	if crashLater {
		os.WriteFile(".git/slice-crash", nil, 0o600)
	}
	count, _ := os.ReadFile(".git/slice-runs")
	runs, _ := strconv.Atoi(string(count))
	os.WriteFile(".git/slice-runs", []byte(strconv.Itoa(runs+1)), 0o600)
	jitter := runs % 2
	fmt.Printf(` + "`" + `{"slice":"inference","identity":"small","model":"small","outputs":"a","metrics":{"load-time":%d,"peak-device-memory":1048576,` +
	`"first-output":%d,"throughput":%d,"failure-rate":0}}` + "`" + `+"\n",
		load+jitter*80, firstOutput+jitter*3, throughput+jitter*10)
}
`

// pairedBuild is what the stub compiles in.
type pairedBuild struct {
	throughput, firstOutput, load int
	crash, crashLater, busy       bool
	laterIdentity                 string
}

func (b pairedBuild) source() string {
	return fmt.Sprintf("package main\n\nvar throughput, firstOutput, load = %d, %d, %d\n\nconst crash, crashLater, busy = %t, %t, %t\n\nconst laterIdentity = %q\n",
		b.throughput, b.firstOutput, b.load, b.crash, b.crashLater, b.busy, b.laterIdentity)
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
	fixture := newSliceFixture(t, map[string]string{
		"cmd/benchmark/main.go": pairedStub, speed: base.source(),
		"fixtures/scorecard_slice/inference.json": `{"task":"inference"}`,
	})
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
	const crashed = "the inference slice measures on HEAD and fails on the tree (exit 2: panic: assignment to entry in nil map)"
	paired := 1 + len(abba)*pairedRounds
	for _, change := range []struct {
		name, message, source, changed string
		runs                           int
		refused, want                  string
	}{
		{name: "slower", message: predicts("throughput rises."), source: with(func(b *pairedBuild) { b.throughput = 400 }).source(),
			refused: "inference throughput -11.0% (noise 2.5%)"},
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
			source: with(func(b *pairedBuild) { b.load = 5000 }).source(), refused: "names inference load-time +24.8% (noise 2.0%), which"},
		{name: "crash", message: unpredicted, source: with(func(b *pairedBuild) { b.crash = true }).source(), refused: crashed},
		{name: "crash while paired", message: predicts("throughput holds."), source: with(func(b *pairedBuild) { b.crashLater = true }).source(), refused: crashed},
		{name: "crash while paired under another identity", message: predicts("throughput holds."),
			source: with(func(b *pairedBuild) { b.crashLater, b.laterIdentity = true, "small-other" }).source(),
			want:   "not paired: the tree failed: exit 2: panic: assignment to entry in nil map"},
		{name: "a slice with no document", message: predicts("speech first-output falls."), source: with(func(b *pairedBuild) { b.firstOutput = 180 }).source(),
			runs: 1, want: "not paired: no speech slice | inference: small"},
		{name: "claimed device", message: predicts("throughput holds."), source: with(func(b *pairedBuild) { b.busy = true }).source(),
			want: "inference: unavailable: benchmark: reserve device 0: device busy"},
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
	records := func(values ...float64) []metric.Record {
		var result []metric.Record
		for _, value := range values {
			result = append(result, metric.Record{Metrics: map[metric.Name]float64{metric.Throughput: value}})
		}
		return result
	}
	change, noise := pairedChange([sides][]metric.Record{
		headSide: records(100, 100, 100, 100),
		treeSide: records(100, 100, 100, 140),
	}, metric.Throughput)
	if math.Abs(change-0.1) > 1e-9 || math.Abs(noise-40.0/110) > 1e-9 {
		t.Fatalf("change, noise = %v, %v; want the means' 10%% and the tree's range over its mean", change, noise)
	}
}
