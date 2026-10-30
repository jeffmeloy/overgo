package scorecard

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAReconvertedModelKeepsItsDeclines replays Qwen3.5-4B's throughput: six readings near 86 tok/s on its converted
// file, then one of 30, a decline whose row is open. The model is then served from other files, a new identity whose
// recorded architecture file is the converted model's, and which the catalog lists in its place. Its row takes the
// converted model's place: it is paired with that model's row, judged on that model's readings, and its decline is the
// open row's, found by that row's key and held to its target, so it is not filed again; a cell back at the target closes
// it. A succession is decided task by task, so a conversion retired for inference and still listed for training is
// succeeded for inference alone. A model listed beside the one it would succeed, or two listed models built from the
// same sources, take no place,
// and a model with no recorded sources, or sources that do not declare its architecture, takes none. Without a recorded succession the new identity is a new model, and
// the decline is not carried.
func TestAReconvertedModelKeepsItsDeclines(t *testing.T) {
	t.Parallel()
	converted, native := testutil.ArtifactID(t, artifact.KindModel, "converted"), testutil.ArtifactID(t, artifact.KindModel, "native")
	twin, other := testutil.ArtifactID(t, artifact.KindModel, "twin"), testutil.ArtifactID(t, artifact.KindModel, "other")
	read := []modelartifact.ConfigSource{{Name: "config.json", SHA256: "c0"}, {Name: "generation_config.json", SHA256: "g0"}}
	sources := map[artifact.ID][]modelartifact.ConfigSource{converted: read, native: {read[0]}, other: {{Name: "config.json", SHA256: "c1"}}}
	served := func(models ...artifact.ID) map[recipe.Task][]artifact.ID {
		return map[recipe.Task][]artifact.ID{recipe.TaskInference: models}
	}
	if got := successions(sources, served(native, other))[recipe.TaskInference]; !slices.Equal(got[native], []artifact.ID{converted}) || len(got) != 1 {
		t.Fatalf("successions with the converted model no longer listed = %v, want the native model in its place alone", got)
	}
	if got := successions(sources, served(converted, native))[recipe.TaskInference]; len(got) != 0 {
		t.Errorf("successions with both listed = %v, want none", got)
	}
	trained := map[recipe.Task][]artifact.ID{recipe.TaskInference: {native}, recipe.TaskTraining: {converted}}
	if got := successions(sources, trained); !slices.Equal(got[recipe.TaskInference][native], []artifact.ID{converted}) || len(got[recipe.TaskTraining][native]) != 0 {
		t.Errorf("successions with the conversion listed for training alone = %v, want the native model in its place for inference only", got)
	}
	sources[twin] = read
	if got := successions(sources, served(native, twin))[recipe.TaskInference]; len(got) != 0 {
		t.Errorf("successions with two listed models read from the same sources = %v, want none", got)
	}
	delete(sources, native)
	if got := successions(sources, served(native))[recipe.TaskInference]; len(got) != 0 {
		t.Errorf("successions of a model with no recorded sources = %v, want none", got)
	}
	card := []modelartifact.ConfigSource{{Name: "model_card_sampling.json", SHA256: "s0"}}
	sampled := map[artifact.ID][]modelartifact.ConfigSource{converted: card, native: card}
	if got := successions(sampled, served(native))[recipe.TaskInference]; len(got) != 0 {
		t.Errorf("successions by a sampling card alone, which Qwen3.5-4B and 9B share = %v, want none", got)
	}

	contract := Contract{Metric: "output tokens over decode time", Origin: originServing, Scope: testutil.ArtifactID(t, artifact.KindRecipe, "served")}
	row := func(model artifact.ID, value float64, succeeds ...artifact.ID) Row {
		served := measuredFixture(model, map[metric.Name]Cell{metric.Throughput: {Value: value, Unit: "tok/s", Contract: contract}})
		served.Succeeds = succeeds
		return served
	}
	var records []Record
	for _, value := range []float64{86, 87, 85, 86, 87, 86} {
		records = append(records, Record{Commit: "converted", Rows: []Row{row(converted, value)}})
	}
	declines := func(records []Record) []Gap {
		return slices.DeleteFunc(records[len(records)-1].Gaps(Reference(records)), func(gap Gap) bool { return gap.kind != string(regressed) })
	}
	open := declines(append(slices.Clone(records), Record{Commit: "declined", Rows: []Row{row(converted, 30)}}))
	if len(open) != 1 || open[0].Target <= 30 {
		t.Fatalf("30 tok/s after six readings near 86 = %+v, want one decline", open)
	}
	records = append(records, Record{Commit: "declined", Rows: []Row{row(converted, 30)}})
	heir := Record{Commit: "native", Rows: []Row{row(native, 30, converted)}}
	carried := declines(append(slices.Clone(records), heir))
	if len(carried) != 1 || carried[0].Key == open[0].Key || !slices.Contains(carried[0].Aliases, open[0].Key) {
		t.Fatalf("the model in the converted model's place at 30 tok/s = %+v, want its decline found by the open row's key %s", carried, open[0].Key)
	}
	if pairs := heir.Pairs(records[len(records)-1]); len(pairs) != 1 || pairs[0].Before == nil || pairs[0].Before.Model != converted || pairs[0].After.Model != native {
		t.Errorf("the pairs of the model in the converted model's place = %+v, want it beside the converted model's row", pairs)
	}
	if restored, found := Restored(heir.Rows, open[0].Key, open[0].Target); !found || restored {
		t.Errorf("the open row's key on the model in its place at 30 = (restored %t, found %t), want found and open", restored, found)
	}
	if restored, found := Restored([]Row{row(native, 90, converted)}, open[0].Key, open[0].Target); !found || !restored {
		t.Errorf("the open row's key on the model in its place at 90 = (restored %t, found %t), want found and closed", restored, found)
	}
	// A further reading of the model in its place is judged on the converted model's readings and its own, and the
	// decline stays the open row's.
	again := declines(append(append(slices.Clone(records), heir), Record{Commit: "again", Rows: []Row{row(native, 31, converted)}}))
	if len(again) != 1 || !slices.Contains(again[0].Aliases, open[0].Key) {
		t.Errorf("the next reading of the model in its place = %+v, want the decline still the open row's", again)
	}
	stranger := Record{Commit: "native", Rows: []Row{row(native, 30)}}
	if unrecorded := declines(append(slices.Clone(records), stranger)); len(unrecorded) != 0 {
		t.Errorf("a new identity with no recorded succession = %+v, want no decline carried to it", unrecorded)
	}
	if restored, found := Restored(stranger.Rows, open[0].Key, open[0].Target); found || restored {
		t.Errorf("the open row's key on a new identity with no recorded succession = (restored %t, found %t), want not found", restored, found)
	}
}
