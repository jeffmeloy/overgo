package recipe

import (
	"slices"
	"strings"
	"testing"

	"llamacpp2go/internal/artifact"
)

const (
	fixtureSourceModule ModuleID = "fixture.source"
	fixtureSinkModule   ModuleID = "fixture.sink"
)

func fixtureRecipe(t *testing.T) (Definition, *Catalog) {
	t.Helper()
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	catalog, err := NewCatalog(
		Module{
			ID: fixtureSourceModule, Tasks: []Task{TaskInference}, Placements: []Placement{PlacementHost},
			Outputs: []Port{{Name: "tokens", Data: DataTokens, Cardinality: CardinalityOne}},
		},
		Module{
			ID: fixtureSinkModule, Tasks: []Task{TaskInference}, Placements: []Placement{PlacementHost},
			Inputs:  []Port{{Name: "tokens", Data: DataTokens, Cardinality: CardinalityOne}},
			Outputs: []Port{{Name: "logits", Data: DataLogits, Cardinality: CardinalityOne}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	source := Node{ID: "source", Module: fixtureSourceModule, Placement: PlacementHost}
	sink := Node{ID: "sink", Module: fixtureSinkModule, Placement: PlacementHost}
	definition, err := NewDefinition(
		TaskInference, modelID, []Node{sink, source},
		[]Edge{{From: Endpoint{Node: source.ID, Port: "tokens"}, To: Endpoint{Node: sink.ID, Port: "tokens"}}},
		nil,
		[]Output{{Name: "logits", Data: DataLogits, Source: Endpoint{Node: sink.ID, Port: "logits"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	return definition, catalog
}

func TestDefinitionCanonicalIdentityAndValidation(t *testing.T) {
	definition, catalog := fixtureRecipe(t)
	if err := definition.Validate(catalog); err != nil {
		t.Fatal(err)
	}
	nodes := slices.Clone(definition.Nodes)
	slices.Reverse(nodes)
	rebuilt, err := NewDefinition(definition.Task, definition.Model, nodes, definition.Edges, definition.Inputs, definition.Outputs)
	if err != nil {
		t.Fatal(err)
	}
	if rebuilt.ID != definition.ID {
		t.Fatalf("canonical IDs differ: %s != %s", rebuilt.ID, definition.ID)
	}
	descriptor, err := definition.Descriptor()
	if err != nil || descriptor.ID != definition.ID || descriptor.Schema != Schema {
		t.Fatalf("descriptor = (%+v, %v)", descriptor, err)
	}
}

func TestDefinitionRejectsSchemaMismatchCycleAndDeadNode(t *testing.T) {
	definition, catalog := fixtureRecipe(t)
	for _, mutate := range []func(*Definition){
		func(value *Definition) { value.Outputs[0].Data = DataTensor },
		func(value *Definition) {
			value.Edges = append(value.Edges, Edge{
				From: Endpoint{Node: "sink", Port: "logits"}, To: Endpoint{Node: "source", Port: "missing"},
			})
		},
		func(value *Definition) {
			value.Nodes = append(value.Nodes, Node{ID: "dead", Module: fixtureSourceModule, Placement: PlacementHost})
		},
	} {
		changed := definition
		changed.Nodes = slices.Clone(definition.Nodes)
		changed.Edges = slices.Clone(definition.Edges)
		changed.Outputs = slices.Clone(definition.Outputs)
		mutate(&changed)
		changed, err := NewDefinition(changed.Task, changed.Model, changed.Nodes, changed.Edges, changed.Inputs, changed.Outputs)
		if err == nil {
			err = changed.Validate(catalog)
		}
		if err == nil {
			t.Fatal("accepted invalid recipe")
		}
		if strings.TrimSpace(err.Error()) == "" {
			t.Fatal("empty validation error")
		}
	}
}
