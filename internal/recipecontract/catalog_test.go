package recipecontract

import (
	"slices"
	"testing"

	"overgo/internal/recipe"
)

func TestCanonicalModuleCatalogContract(t *testing.T) {
	module := recipe.Module{
		ID:         "contract.source",
		Tasks:      []recipe.Task{recipe.TaskTraining, recipe.TaskInference, recipe.TaskInference},
		Placements: []recipe.Placement{recipe.PlacementDevice, recipe.PlacementHost},
		Outputs: []recipe.Port{
			{Name: "tokens", Data: recipe.DataTokens, Cardinality: recipe.CardinalityOne},
			{Name: "embeddings", Data: recipe.DataEmbeddings, Cardinality: recipe.CardinalityOne},
		},
	}
	catalog, err := recipe.NewCatalog(module)
	if err != nil {
		t.Fatal(err)
	}

	module.Tasks[0] = recipe.TaskForecast
	module.Outputs[0].Data = recipe.DataImage
	stored, ok := catalog.Module(module.ID)
	if !ok {
		t.Fatal("canonical module is absent")
	}
	if !slices.Equal(stored.Tasks, []recipe.Task{recipe.TaskInference, recipe.TaskTraining}) {
		t.Fatalf("canonical tasks = %v", stored.Tasks)
	}
	if stored.Outputs[0].Name != "embeddings" || stored.Outputs[0].Data != recipe.DataEmbeddings {
		t.Fatalf("catalog retained caller mutation or noncanonical ports: %+v", stored.Outputs)
	}

	stored.Outputs[0].Data = recipe.DataImage
	reloaded, _ := catalog.Module(module.ID)
	if reloaded.Outputs[0].Data != recipe.DataEmbeddings {
		t.Fatal("catalog accessor exposed mutable module state")
	}
}
