package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// readingsFake stands in for the build's processes: a render writes its suite unless its arm is told to fail, a judged
// run prints another suite's record before its own, and an admission prints a verdict over several lines.
type readingsFake struct {
	t        *testing.T
	failDraw map[string]bool
	calls    []string
}

func readingsRecord(t *testing.T, suite string) string {
	t.Helper()
	return testcheck.Must(artifact.JSONID(artifact.KindEvaluation, suite))(t).String()
}

func (f *readingsFake) run(_ context.Context, args []string, stdout, stderr io.Writer) error {
	named := map[string]string{}
	for index := 0; index+1 < len(args); index += 2 {
		named[args[index]] = args[index+1]
	}
	switch {
	case named["-render-tifa"] != "" || named["-render-vidgen"] != "":
		f.calls = append(f.calls, "render "+filepath.Base(named["-media-out"]))
		if f.failDraw[filepath.Base(named["-media-out"])] {
			fmt.Fprintln(stderr, "workflow runtime: step integrate: artifact: invalid content size")
			return errors.New("exit status 1")
		}
		return os.WriteFile(named["-suite-out"], []byte("{}"), 0o644)
	case named["-manifest"] != "":
		data, err := os.ReadFile(named["-manifest"])
		if err != nil {
			return err
		}
		var judged manifest
		if err := strictjson.DecodeBytes(data, &judged); err != nil {
			return err
		}
		suite := judged.Models[0].Suites[0]
		f.calls = append(f.calls, "judge "+filepath.Base(suite))
		if judged.CodeCommit != "commit-under-test" || judged.Models[0].MediaRoot+"-suite.json" != suite {
			f.t.Errorf("judge manifest = %s", data)
		}
		other := filepath.Join(filepath.Dir(suite), "another-suite.json")
		fmt.Fprintf(stdout, "  progress image-questions: 1/1 score=1.000\n%s%s%s%s\n", evaluatedLine, readingsRecord(f.t, other), evaluatedOf, other)
		fmt.Fprintf(stdout, "%s%s%s%s\n", evaluatedLine, readingsRecord(f.t, suite), evaluatedOf, suite)
		return nil
	case named["-admit-preset"] != "":
		f.calls = append(f.calls, "admit "+filepath.Base(named["-admit-preset"]))
		fmt.Fprintf(stdout, "{\n  \"Preset\": %q,\n  \"Admitted\": true\n}\n", named["-preset-evaluation"])
		return nil
	}
	return fmt.Errorf("unexpected arguments %q", args)
}

// TestAReadingsCampaignReadsEachArmsOwnRecord holds the preset-readings campaign (guidance-ops/readings-campaign),
// which readings.sh ran by hand: each arm is drawn, judged in a manifest of its own and recorded with the record its
// judged run names for its suite, not the store's newest; a preset whose render fails is passed over for the next, and
// the error names it; a campaign run again draws, judges and reads only what it has not.
func TestAReadingsCampaignReadsEachArmsOwnRecord(t *testing.T) {
	root := t.TempDir()
	out := filepath.Join(root, "out")
	campaign := readingsCampaign{Out: out, Repository: filepath.Join(root, "store"), Catalog: filepath.Join(root, "catalog.json"),
		Judge: "judge.gguf", Projector: "projector.gguf", Models: []readingsModel{
			{Name: "pictures", Path: "picture-model", Task: "image-gen", Source: "tifa.json", Cases: 4,
				Reference: filepath.Join(root, "pictures-person"), Arms: filepath.Join(root, "pictures"), Presets: []string{"apg", "interval"}},
			{Name: "clips", Path: "clip-model", Task: "video-gen", Source: "vidgen.json", Cases: 2,
				Reference: filepath.Join(root, "clips-person"), Arms: filepath.Join(root, "clips"), Presets: []string{"shift"}},
		}}
	encoded := testcheck.Must(json.Marshal(campaign))(t)
	path := filepath.Join(root, "campaign.json")
	testcheck.NoError(t, os.WriteFile(path, encoded, 0o644))
	now := func() time.Time { return time.Date(2026, 11, 12, 9, 0, 0, 0, time.UTC) }

	first := &readingsFake{t: t, failDraw: map[string]bool{"pictures-apg": true}}
	err := runReadings(t.Context(), path, first.run, "commit-under-test", now)
	if err == nil || !strings.Contains(err.Error(), "pictures-apg: render failed") || strings.Contains(err.Error(), "interval") {
		t.Fatalf("first run's error = %v, want the failed render of pictures-apg alone", err)
	}
	records := testcheck.Must(readArmLines(filepath.Join(out, readingsRecords)))(t)
	for name, media := range map[string]string{"pictures-reference": "pictures-person", "pictures-interval": "pictures-interval", "clips-reference": "clips-person", "clips-shift": "clips-shift"} {
		want := readingsRecord(t, filepath.Join(root, media+"-suite.json")) + " " + filepath.Join(root, media)
		if records[name] != want {
			t.Errorf("record of %s = %q, want its own suite's %q", name, records[name], want)
		}
	}
	if _, recorded := records["pictures-apg"]; recorded || len(records) != 4 {
		t.Errorf("records = %v, want none for the arm whose render failed", records)
	}
	verdicts := testcheck.Must(readArmLines(filepath.Join(out, readingsVerdicts)))(t)
	interval, _, _ := strings.Cut(records["pictures-interval"], " ")
	if want := fmt.Sprintf(`{"Preset":%q,"Admitted":true}`, interval); verdicts["pictures-interval"] != want || len(verdicts) != 2 {
		t.Errorf("verdicts = %v, want pictures-interval's on one line, %s, and clips-shift's", verdicts, want)
	}
	if !slices.Contains(first.calls, "render clips-person") || slices.Contains(first.calls, "admit pictures-apg") {
		t.Errorf("first run's calls = %q, want the clips drawn after the failed preset and no reading of it", first.calls)
	}

	again := &readingsFake{t: t}
	testcheck.NoError(t, runReadings(t.Context(), path, again.run, "commit-under-test", now))
	if want := []string{"render pictures-apg", "judge pictures-apg-suite.json", "admit pictures-apg"}; !slices.Equal(again.calls, want) {
		t.Errorf("second run's calls = %q, want only the failed arm's %q", again.calls, want)
	}
}

// TestAJudgedRunThatNamesNoRecordOfItsSuiteFailsItsArm holds the lookup: output naming no record of the arm's suite,
// or naming one that is no evaluation record, fails the arm rather than recording another's.
func TestAJudgedRunThatNamesNoRecordOfItsSuiteFailsItsArm(t *testing.T) {
	suite := filepath.Join(t.TempDir(), "arm-suite.json")
	other := filepath.Join(filepath.Dir(suite), "other-suite.json")
	if _, err := evaluatedRecord(strings.NewReader(evaluatedLine+readingsRecord(t, other)+evaluatedOf+other+"\n"), suite); err == nil ||
		!strings.Contains(err.Error(), "named no evaluation record") {
		t.Errorf("another suite's record gave %v", err)
	}
	if _, err := evaluatedRecord(strings.NewReader(evaluatedLine+"run:sha256:00"+evaluatedOf+suite+"\n"), suite); err == nil {
		t.Error("a record that is no evaluation was taken")
	}
}
