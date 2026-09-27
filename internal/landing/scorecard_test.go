package landing

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

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
// crash measures nothing, refuses nothing and says which it was.
func TestLandingMeasuresTheScorecardSlice(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git := func(arguments ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...)
		command.Dir = repo
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	const result = "cmd/benchmark/result.json"
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
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("docs/readme.md", "fixture\n")
	write("cmd/benchmark/main.go", sliceStub)
	git("init", "-q")
	git("add", "-A")
	git("commit", "-q", "-m", "fixture")
	land := func(changed string) *landing {
		t.Helper()
		l := &landing{Options: Options{Repo: repo, Ref: "row/step", Message: []byte("change"), Log: io.Discard},
			paths: []string{changed}, details: map[string]string{}}
		if err := l.scorecard(t.Context()); err != nil {
			t.Fatal(err)
		}
		return l
	}
	measure := func(name, measured string) *landing {
		t.Helper()
		write(result, measured)
		l := land(result)
		if !strings.HasPrefix(l.trailer, sliceTrailer+": {") {
			t.Fatalf("%s: trailer = %q, want the slice", name, l.trailer)
		}
		return l
	}
	commit := func(l *landing) string {
		t.Helper()
		git("add", "-A")
		git("commit", "-q", "-m", l.message())
		return git("rev-parse", "--short", "HEAD")
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
	firstCommit := commit(first)
	changed := measure("changed", measurement("small", 102, "b", "b"))
	expect("changed", changed, "against "+firstCommit+": outputs differ; peak-device-memory 102.0 MiB (+2.0)")
	changedCommit := commit(changed)
	other := measure("other model", measurement("other", 100, "a", "a"))
	expect("other model", other, "other, no earlier slice of this identity")
	commit(other)
	// A merged branch's newer slice of the same identity is not the
	// mainline's baseline.
	git("checkout", "-q", "-b", "side")
	commit(measure("side", measurement("small", 200, "z", "z")))
	git("checkout", "-q", "-")
	git("merge", "-q", "--no-ff", "-m", "merge side", "side")
	unchanged := measure("unchanged", measurement("small", 102, "b", "b"))
	expect("unchanged", unchanged, "against "+changedCommit+": outputs identical; peak-device-memory 102.0 MiB (+0.0)")
	varied := measure("varied", measurement("small", 102, "b", "c"))
	expect("varied", varied, "outputs vary between runs")
	variedCommit := commit(varied)
	stable := measure("stable", measurement("small", 102, "b", "b"))
	expect("stable", stable, "against "+variedCommit+": outputs not compared: the baseline's varied")
	git("checkout", "-q", "--", result)

	for _, missing := range []struct {
		name, changed, content, want string
	}{
		{"outside", "docs/readme.md", "", "not reached"},
		{"busy", result, "", "unavailable: benchmark: reserve device 0: device busy"},
		{"crash", result, "panic", "failed: exit 2: panic: assignment to entry in nil map"},
	} {
		path := filepath.Join(repo, filepath.FromSlash(result))
		if missing.content == "" {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				t.Fatal(err)
			}
		} else {
			write(result, missing.content)
		}
		if l := land(missing.changed); l.details["scorecard"] != missing.want || l.trailer != "" {
			t.Fatalf("%s: slice = %q with trailer %q, want %q and none", missing.name, l.details["scorecard"], l.trailer, missing.want)
		}
	}
}
