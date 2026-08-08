package recipe

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const (
	fixtureSourceModule ModuleID = "fixture.source"
	fixtureSinkModule   ModuleID = "fixture.sink"
)

func fixtureRecipe(t *testing.T) (Definition, *Catalog) {
	t.Helper()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "model")
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

func TestCatalogCloneOwnsModuleMap(t *testing.T) {
	_, catalog := fixtureRecipe(t)
	cloned := catalog.Clone()
	extra := Module{
		ID: "fixture.extra", Tasks: []Task{TaskInference}, Placements: []Placement{PlacementHost},
		Outputs: []Port{{Name: "value", Data: DataTensor, Cardinality: CardinalityOne}},
	}
	if err := cloned.Register(extra); err != nil {
		t.Fatal(err)
	}
	if _, ok := catalog.Module(extra.ID); ok {
		t.Fatal("catalog clone mutated source")
	}
	if _, ok := cloned.Module(extra.ID); !ok {
		t.Fatal("catalog clone lost registered module")
	}
}

func TestDefinitionDependenciesDriveIdentity(t *testing.T) {
	definition, _ := fixtureRecipe(t)
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "profile")
	tokenizerID := testutil.ArtifactID(t, artifact.KindTokenizer, "tokenizer")
	dependencies := []Dependency{
		{Role: DependencyTokenizer, Artifact: tokenizerID},
		{Role: DependencyModel, Artifact: definition.Model},
		{Role: DependencyProfile, Artifact: profileID},
	}
	bound, err := NewDefinitionWithDependencies(
		definition.Task, dependencies, definition.Nodes, definition.Edges,
		definition.Inputs, definition.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	slices.Reverse(dependencies)
	reordered, err := NewDefinitionWithDependencies(
		definition.Task, dependencies, definition.Nodes, definition.Edges,
		definition.Inputs, definition.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if reordered.ID != bound.ID {
		t.Fatal("dependency order changed recipe identity")
	}
	if found, ok := bound.Dependency(DependencyProfile, 0); !ok || found != profileID {
		t.Fatalf("profile dependency = (%s, %v)", found, ok)
	}
	otherProfile := testutil.ArtifactID(t, artifact.KindProfile, "other-profile")
	dependencies[0].Artifact = otherProfile
	changed, err := NewDefinitionWithDependencies(
		definition.Task, dependencies, definition.Nodes, definition.Edges,
		definition.Inputs, definition.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if changed.ID == bound.ID {
		t.Fatal("profile change preserved recipe identity")
	}
}

func TestDefinitionReadsCanonicalLegacyVersion(t *testing.T) {
	current, _ := fixtureRecipe(t)
	legacy, err := newDefinition(
		LegacyVersion, current.Task, current.Model, nil,
		current.Nodes, current.Edges, current.Inputs, current.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := legacy.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseDefinition(content)
	if err != nil {
		t.Fatal(err)
	}
	descriptor, err := parsed.Descriptor()
	if err != nil || parsed.ID != legacy.ID || parsed.Version != LegacyVersion || descriptor.Schema != LegacySchema {
		t.Fatalf("legacy definition = (%+v, %+v, %v)", parsed, descriptor, err)
	}
}

func TestDefinitionRejectsInvalidDependencies(t *testing.T) {
	definition, _ := fixtureRecipe(t)
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "profile")
	for _, dependencies := range [][]Dependency{
		{{Role: DependencyProfile, Artifact: profileID}},
		{{Role: DependencyModel, Artifact: definition.Model}, {Role: DependencyModel, Artifact: definition.Model}},
		{{Role: DependencyProfile, Artifact: definition.Model}, {Role: DependencyModel, Artifact: definition.Model}},
		{{Role: DependencyProfile, Artifact: profileID}, {Role: DependencyModel, Slot: 1, Artifact: definition.Model}},
	} {
		if _, err := NewDefinitionWithDependencies(
			definition.Task, dependencies, definition.Nodes, definition.Edges,
			definition.Inputs, definition.Outputs,
		); err == nil {
			t.Fatalf("accepted dependencies %+v", dependencies)
		}
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
