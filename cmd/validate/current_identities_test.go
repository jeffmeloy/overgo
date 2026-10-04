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

// TestSupersededIdentitiesKeepNoCapabilityTheCurrentOneHolds holds the cells
// of a weights file to its current identity: where two identities answer to
// one location and share a task, the identity the weights no longer resolve to
// loses that task, whose proof could only fail its handoff every cycle; what
// else it carries, the other locations and an unresolvable location stay.
func TestSupersededIdentitiesKeepNoCapabilityTheCurrentOneHolds(t *testing.T) {
	t.Parallel()
	model := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, name) }
	entry := func(name, location string, tasks ...recipe.Task) discovery.CatalogEntry {
		var capabilities []discovery.Capability
		for _, task := range tasks {
			capabilities = append(capabilities, discovery.Capability{Task: task, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name+string(task))})
		}
		return discovery.CatalogEntry{Model: model(name), Location: location, Present: true, Capabilities: capabilities}
	}
	entries := []discovery.CatalogEntry{
		entry("current", "C:/models/4b.gguf", recipe.TaskInference, recipe.TaskTraining),
		entry("superseded", "C:/models/4b.gguf", recipe.TaskTraining, recipe.TaskVQA),
		entry("lone", "C:/models/lone.gguf", recipe.TaskInference),
		entry("first-of-two", "C:/models/two-tasks.gguf", recipe.TaskTraining),
		entry("second-of-two", "C:/models/two-tasks.gguf", recipe.TaskVQA),
		entry("unresolved-a", "C:/models/unresolved.gguf", recipe.TaskTraining),
		entry("unresolved-b", "C:/models/unresolved.gguf", recipe.TaskTraining),
	}
	var resolved []string
	resolve := func(location string) (artifact.ID, error) {
		resolved = append(resolved, location)
		if location == "C:/models/unresolved.gguf" {
			return artifact.ID{}, errors.New("unreadable")
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
		"first-of-two": {recipe.TaskTraining}, "second-of-two": {recipe.TaskVQA},
		"unresolved-a": {recipe.TaskTraining}, "unresolved-b": {recipe.TaskTraining},
	} {
		if got := tasksOf(name); !slices.Equal(got, want) {
			t.Errorf("%s carries %v, want %v", name, got, want)
		}
	}
	slices.Sort(resolved)
	if want := []string{"C:/models/4b.gguf", "C:/models/unresolved.gguf"}; !slices.Equal(resolved, want) {
		t.Errorf("resolved %v, want only the locations two identities share a task at: %v", resolved, want)
	}
	if got := len(entries[1].Capabilities); got != 2 {
		t.Errorf("the catalog's own entry holds %d capabilities after the drop, want it left as listed (2)", got)
	}
}
