package workflowrecipe

import (
	"testing"

	"overgo/internal/recipe"
)

func TestDPORecipeStageOrder(t *testing.T) {
	want := []recipe.ModuleID{
		ModuleBatchPreference, ModuleScorePolicy, ModuleScoreReference,
		ModuleDPOObjective,
	}
	modules := preferenceModules()
	if len(modules) != len(want) {
		t.Fatalf("modules=%d want=%d", len(modules), len(want))
	}
	for index, module := range want {
		if modules[index].ID != module {
			t.Fatalf("module %d=%q want=%q", index, modules[index].ID, module)
		}
		if _, ok := Catalog().Module(module); !ok {
			t.Fatalf("catalog lacks %q", module)
		}
	}
}
