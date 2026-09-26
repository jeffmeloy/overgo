package composition_test

import (
	"overgo/internal/composition"
	"overgo/internal/composition/compositiontest"
	"testing"

	"overgo/internal/representation"
)

func TestRepresentationContractRepository(t *testing.T) {
	store, authority := compositiontest.Authority(t)
	loaded, err := representation.LoadContract(t.Context(), store, authority.SourceContract.ID)
	if err != nil || loaded.ID != authority.SourceContract.ID || loaded.Producer.Model != authority.Recipe.SourceModel {
		t.Fatalf("loaded source contract = %+v, %v", loaded, err)
	}
	if _, err := representation.LoadContract(t.Context(), store, authority.Bridge.ID); err == nil {
		t.Fatal("bridge definition admitted as a representation contract")
	}
}

func TestBridgeDefinitionRepository(t *testing.T) {
	store, authority := compositiontest.Authority(t)
	loaded, err := composition.LoadBridgeDefinition(t.Context(), store, authority.Bridge.ID)
	if err != nil || loaded != authority.Bridge || loaded.Weights != authority.Recipe.BridgeWeights {
		t.Fatalf("loaded bridge definition = %+v, %v", loaded, err)
	}
	weights, err := composition.LoadBridgeWeights(t.Context(), store, authority.Bridge.ID)
	if err != nil || weights.Weights.ID != authority.Bridge.Weights ||
		weights.Inventory.ID != authority.Bridge.WeightInventory {
		t.Fatalf("loaded bridge weights = %+v, %v", weights, err)
	}
}

func TestCompositionRecipeRepository(t *testing.T) {
	store, authority := compositiontest.Authority(t)
	loaded, err := composition.LoadCompositionRecipe(t.Context(), store, authority.Recipe.ID)
	if err != nil || loaded != authority.Recipe || loaded.Promotion != authority.Promotion.ID {
		t.Fatalf("loaded composition recipe = %+v, %v", loaded, err)
	}
}
