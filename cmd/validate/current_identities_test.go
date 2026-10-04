package main

import (
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestSupersededIdentitiesKeepNoTrainingTheCurrentOneHolds holds the cells of a
// weights file to its current identity for training alone: where two identities
// answer to one location and both carry a training capability, the identity the
// weights no longer resolve to loses it, whose proof could only fail its
// handoff every cycle, unless the current identity's own is stale. What else a
// superseded identity carries, tasks other than training, locations where one
// identity alone trains, and an unresolvable location stay, and only locations
// two identities train at are resolved.
func TestSupersededIdentitiesKeepNoTrainingTheCurrentOneHolds(t *testing.T) {
	t.Parallel()
	model := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, name) }
	entry := func(name, location, stale string, tasks ...recipe.Task) discovery.CatalogEntry {
		var capabilities []discovery.Capability
		for _, task := range tasks {
			capabilities = append(capabilities, discovery.Capability{Task: task, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name+string(task)), Stale: stale})
		}
		return discovery.CatalogEntry{Model: model(name), Location: location, Present: true, Capabilities: capabilities}
	}
	entries := []discovery.CatalogEntry{
		entry("current", "C:/models/4b.gguf", "", recipe.TaskInference, recipe.TaskTraining),
		entry("superseded", "C:/models/4b.gguf", "", recipe.TaskTraining, recipe.TaskVQA),
		entry("lone", "C:/models/lone.gguf", "", recipe.TaskInference),
		entry("trains", "C:/models/two-tasks.gguf", "", recipe.TaskTraining),
		entry("serves", "C:/models/two-tasks.gguf", "", recipe.TaskInference),
		entry("twin-a", "C:/models/twin.gguf", "", recipe.TaskInference),
		entry("twin-b", "C:/models/twin.gguf", "", recipe.TaskInference),
		entry("unresolved-a", "C:/models/unresolved.gguf", "", recipe.TaskTraining),
		entry("unresolved-b", "C:/models/unresolved.gguf", "", recipe.TaskTraining),
		entry("stale-current", "C:/models/stale.gguf", "its recipe was replaced", recipe.TaskTraining),
		entry("stale-other", "C:/models/stale.gguf", "", recipe.TaskTraining),
	}
	var resolved []string
	resolve := func(location string) (artifact.ID, error) {
		resolved = append(resolved, location)
		switch location {
		case "C:/models/unresolved.gguf":
			return artifact.ID{}, errors.New("unreadable")
		case "C:/models/stale.gguf":
			return model("stale-current"), nil
		}
		return model("current"), nil
	}
	kept := currentIdentities(entries, resolve)

	tasksOf := func(name string) []recipe.Task {
		for _, held := range kept {
			if held.Model == model(name) {
				var tasks []recipe.Task
				for _, capability := range held.Capabilities {
					tasks = append(tasks, capability.Task)
				}
				return tasks
			}
		}
		return nil
	}
	for name, want := range map[string][]recipe.Task{
		"current": {recipe.TaskInference, recipe.TaskTraining}, "superseded": {recipe.TaskVQA}, "lone": {recipe.TaskInference},
		"trains": {recipe.TaskTraining}, "serves": {recipe.TaskInference},
		"twin-a": {recipe.TaskInference}, "twin-b": {recipe.TaskInference},
		"unresolved-a": {recipe.TaskTraining}, "unresolved-b": {recipe.TaskTraining},
		"stale-current": {recipe.TaskTraining}, "stale-other": {recipe.TaskTraining},
	} {
		if got := tasksOf(name); !slices.Equal(got, want) {
			t.Errorf("%s carries %v, want %v", name, got, want)
		}
	}
	slices.Sort(resolved)
	if want := []string{"C:/models/4b.gguf", "C:/models/stale.gguf", "C:/models/unresolved.gguf"}; !slices.Equal(resolved, want) {
		t.Errorf("resolved %v, want only the locations two identities train at: %v", resolved, want)
	}
	if got := len(entries[1].Capabilities); got != 2 {
		t.Errorf("the catalog's own entry holds %d capabilities after the drop, want it left as listed (2)", got)
	}
}
