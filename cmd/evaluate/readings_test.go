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
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// readingsFake stands in for the build's processes: a render writes its suite unless its arm is told to fail, a judged
// run prints another suite's record before its own, an admission prints a verdict over several lines, and a
// reference's room prints a gain possible unless the reference's record is roomless.
type readingsFake struct {
	t        *testing.T
	failDraw map[string]bool
	calls    []string
	// negatives are the negative prompts each render asked with, by its media directory.
	negatives map[string]string
	// roomless are the reference records whose room leaves no gain able to show, and rooms the records whose room
	// was read.
	roomless map[string]bool
	rooms    []string
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
	case named["-reading-room"] != "":
		f.rooms = append(f.rooms, named["-reading-room"])
		room := evaluation.ReadingRoom{Media: 80, Room: 30, BestTail: 1.0 / (1 << 30), PassMark: 0.05, Needed: 5}
		if f.roomless[named["-reading-room"]] {
			room = evaluation.ReadingRoom{Media: 80, Room: 4, BestTail: 1.0 / 16, PassMark: 0.05}
		}
		return json.NewEncoder(stdout).Encode(room)
	case named["-pairwise-evaluation"] != "":
		f.calls = append(f.calls, "admit-pairwise "+filepath.Base(named["-pairwise-preset"]))
		fmt.Fprintf(stdout, "{\"Preset\": %q, \"Admitted\": true}\n", named["-pairwise-evaluation"])
		return nil
	case named["-pairwise-preset"] != "":
		f.calls = append(f.calls, "compose "+filepath.Base(named["-media-out"])+" from "+filepath.Base(named["-reference-render"])+" and "+filepath.Base(named["-pairwise-preset"]))
		return os.WriteFile(named["-suite-out"], []byte("{}"), 0o644)
	case named["-render-tifa"] != "" || named["-render-vidgen"] != "":
		f.calls = append(f.calls, "render "+filepath.Base(named["-media-out"]))
		if f.negatives != nil {
			f.negatives[filepath.Base(named["-media-out"])] = named["-render-negative"]
		}
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
		asked := judged.Models[0]
		if judged.CodeCommit != "commit-under-test" || asked.MediaRoot+"-suite.json" != suite || asked.Projector != "projector.gguf" {
			f.t.Errorf("judge manifest = %s", data)
		}
		// A pairwise suite's composites no model drew, so the judge answers as itself; a question suite's judge reads
		// the generator's drawings.
		if paired := strings.HasSuffix(suite, "-pairwise-suite.json"); paired && (asked.Path != "judge.gguf" || asked.Judge != "" || asked.Task != "") ||
			!paired && (asked.Path == "judge.gguf" || asked.Judge != "judge.gguf") {
			f.t.Errorf("the manifest judging %s names model %q with judge %q and task %q", filepath.Base(suite), asked.Path, asked.Judge, asked.Task)
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
		want := readingsRecord(t, filepath.Join(root, media+"-suite.json")) + " " + filepath.Join(root, media) + " " + wallsField + "0"
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

// TestANAGArmAsksWithItsNegativePromptAndIsReadPairwise holds a NAG preset's reading (guidance-ops/nag-negative): the
// campaign draws the preset's arm with the negative prompt it names for that preset and the reference with none, and
// reads a picture model's arm pairwise beside its questions, the two renders composed, judged and read once each; a
// negative prompt for no preset, or a clip model read pairwise, is refused. A reading pairs the arm's draws with the
// reference's alike but the preset and the negative prompt the arm's manifest names, so a reference that drew with a
// negative prompt pairs with none.
func TestANAGArmAsksWithItsNegativePromptAndIsReadPairwise(t *testing.T) {
	root := t.TempDir()
	model := readingsModel{Name: "pictures", Path: "picture-model", Task: "image-gen", Source: "tifa.json", Cases: 2,
		Reference: filepath.Join(root, "pictures-person"), Arms: filepath.Join(root, "pictures"), Presets: []string{"nag"},
		Negative: map[string]string{"nag": "low resolution, blurry"}, Pairwise: true}
	campaign := readingsCampaign{Out: filepath.Join(root, "out"), Repository: filepath.Join(root, "store"), Catalog: filepath.Join(root, "catalog.json"),
		Judge: "judge.gguf", Projector: "projector.gguf", Models: []readingsModel{model}}
	path := filepath.Join(root, "campaign.json")
	testcheck.NoError(t, os.WriteFile(path, testcheck.Must(json.Marshal(campaign))(t), 0o644))
	now := func() time.Time { return time.Date(2026, 11, 16, 21, 0, 0, 0, time.UTC) }

	first := &readingsFake{t: t, negatives: map[string]string{}}
	testcheck.NoError(t, runReadings(t.Context(), path, first.run, "commit-under-test", now))
	want := []string{"render pictures-person", "judge pictures-person-suite.json", "render pictures-nag", "judge pictures-nag-suite.json", "admit pictures-nag",
		"compose pictures-nag-pairwise from pictures-person and pictures-nag", "judge pictures-nag-pairwise-suite.json", "admit-pairwise pictures-nag"}
	if !slices.Equal(first.calls, want) {
		t.Errorf("calls = %q, want %q", first.calls, want)
	}
	if first.negatives["pictures-nag"] != "low resolution, blurry" || first.negatives["pictures-person"] != "" {
		t.Errorf("the renders asked with negative prompts %q; want the preset's arm alone asked with its own", first.negatives)
	}
	verdicts := testcheck.Must(readArmLines(filepath.Join(campaign.Out, readingsVerdicts)))(t)
	if _, read := verdicts["pictures-nag-pairwise"]; !read || len(verdicts) != 2 {
		t.Errorf("verdicts = %v, want the arm's questions' and its pairwise reading's", verdicts)
	}
	again := &readingsFake{t: t}
	testcheck.NoError(t, runReadings(t.Context(), path, again.run, "commit-under-test", now))
	if len(again.calls) != 0 {
		t.Errorf("a campaign run again made %q, want nothing", again.calls)
	}
	for name, refused := range map[string]readingsModel{
		"a negative prompt for no preset": {Negative: map[string]string{"apg": "blurry"}},
		"clips read pairwise":             {Task: "video-gen"},
	} {
		changed := model
		if refused.Negative != nil {
			changed.Negative = refused.Negative
		}
		if refused.Task != "" {
			changed.Task = refused.Task
		}
		campaign.Models = []readingsModel{changed}
		testcheck.NoError(t, os.WriteFile(path, testcheck.Must(json.Marshal(campaign))(t), 0o644))
		if _, err := readReadingsCampaign(path); err == nil {
			t.Errorf("a campaign with %s was read", name)
		}
	}

	painter, painting := testutil.ArtifactID(t, artifact.KindModel, "painter"), testutil.ArtifactID(t, artifact.KindRecipe, "painting")
	drawn := func(preset, negative string, request string) renderManifest {
		return renderManifest{Model: painter, Recipe: painting, Task: recipe.TaskImageGen, Preset: preset, Negative: negative,
			Draws: map[string]renderedMedia{"river": {File: "river.png", Request: json.RawMessage(request)}}}
	}
	reference := drawn("", "", `{"prompt":"a river","seed":7}`)
	guided := drawn("nag", "blurry", `{"negative_prompt":"blurry","preset":"nag","prompt":"a river","seed":7}`)
	if pairs, err := pairedDraws(reference, guided); err != nil || len(pairs) != 1 {
		t.Errorf("the arm asked with its negative prompt paired %d draws (%v); want its reference's one", len(pairs), err)
	}
	if _, err := pairedDraws(drawn("", "", `{"negative_prompt":"blurry","prompt":"a river","seed":7}`), guided); err == nil {
		t.Error("a reference drawn with a negative prompt was paired with the arm that asked with it")
	}
	if _, err := pairedDraws(reference, drawn("nag", "", `{"negative_prompt":"blurry","preset":"nag","prompt":"a river","seed":7}`)); err == nil {
		t.Error("an arm whose manifest names no negative prompt was paired aside the one its requests asked with")
	}
}

// TestACampaignDrawsNoArmThatCannotShowAGain holds the campaign to its reference's own scores (reading-power/do, the
// owner's direction of 2026-11-16): before a model's arms are drawn, the room a reading against its reference has is
// read; a reference that leaves no gain able to reach the pass mark draws none of the model's arms and says so with the
// card hours they would take, read from the reference's own timed draw and judging, and one with room draws its arms
// and says how many media a gain must turn right.
func TestACampaignDrawsNoArmThatCannotShowAGain(t *testing.T) {
	root := t.TempDir()
	out := filepath.Join(root, "out")
	campaign := readingsCampaign{Out: out, Repository: filepath.Join(root, "store"), Catalog: filepath.Join(root, "catalog.json"),
		Judge: "judge.gguf", Projector: "projector.gguf", Models: []readingsModel{
			{Name: "full", Path: "full-model", Task: "image-gen", Source: "tifa.json", Cases: 4,
				Reference: filepath.Join(root, "full-person"), Arms: filepath.Join(root, "full"), Presets: []string{"apg", "interval"}},
			{Name: "open", Path: "open-model", Task: "image-gen", Source: "tifa.json", Cases: 4,
				Reference: filepath.Join(root, "open-person"), Arms: filepath.Join(root, "open"), Presets: []string{"apg"}},
		}}
	path := filepath.Join(root, "campaign.json")
	testcheck.NoError(t, os.WriteFile(path, testcheck.Must(json.Marshal(campaign))(t), 0o644))
	testcheck.NoError(t, os.MkdirAll(out, 0o755))
	full := readingsRecord(t, filepath.Join(root, "full-person-suite.json"))
	timed := strings.Join([]string{"full-reference", full, filepath.Join(root, "full-person"), wallsField + "5400"}, " ")
	testcheck.NoError(t, os.WriteFile(filepath.Join(out, readingsRecords), []byte(timed+"\n"), 0o644))
	now := func() time.Time { return time.Date(2026, 11, 16, 23, 0, 0, 0, time.UTC) }

	fake := &readingsFake{t: t, roomless: map[string]bool{full: true}}
	testcheck.NoError(t, runReadings(t.Context(), path, fake.run, "commit-under-test", now))
	if slices.ContainsFunc(fake.calls, func(call string) bool {
		return strings.Contains(call, "full-") && !strings.Contains(call, "full-person")
	}) {
		t.Errorf("calls = %q; a model whose reference leaves no gain able to show drew an arm", fake.calls)
	}
	if !slices.Contains(fake.calls, "render open-apg") || !slices.Contains(fake.calls, "admit open-apg") {
		t.Errorf("calls = %q; want the model with room drawn, judged and admitted", fake.calls)
	}
	if len(fake.rooms) != 2 || fake.rooms[0] != full {
		t.Errorf("rooms read = %q; want each model's reference read before its arms", fake.rooms)
	}
	said := string(testcheck.Must(os.ReadFile(filepath.Join(out, "campaign.log")))(t))
	for _, want := range []string{"full: no gain can show", "its 2 arms are not drawn, about 3.0 card hours", "open: a gain shows only where a preset turns a wrong answer right in at least 5 of the 30 media"} {
		if !strings.Contains(said, want) {
			t.Errorf("the campaign's log says %q; want %q", said, want)
		}
	}
}
