package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestParallelReadySet(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "parallel-model")
	branch := Module{
		ID: "branch", Tasks: []Task{TaskInference}, Placements: []Placement{PlacementHost},
		Inputs:  []Port{{Name: "input", Data: DataText, Cardinality: CardinalityOne}},
		Outputs: []Port{{Name: "output", Data: DataText, Cardinality: CardinalityOne}},
	}
	join := Module{
		ID: "join", Tasks: []Task{TaskInference}, Placements: []Placement{PlacementHost},
		Inputs:  []Port{{Name: "inputs", Data: DataText, Cardinality: CardinalityMany}},
		Outputs: []Port{{Name: "output", Data: DataText, Cardinality: CardinalityOne}},
	}
	left := Node{ID: "left", Module: branch.ID, Placement: PlacementHost}
	right := Node{ID: "right", Module: branch.ID, Placement: PlacementHost}
	sink := Node{ID: "sink", Module: join.ID, Placement: PlacementHost}
	definition, err := NewDefinitionWithDependencies(
		TaskInference, []Dependency{{Role: DependencyModel, Artifact: model}},
		[]Node{left, right, sink},
		[]Edge{
			{From: Endpoint{Node: left.ID, Port: "output"}, To: Endpoint{Node: sink.ID, Port: "inputs"}},
			{From: Endpoint{Node: right.ID, Port: "output"}, To: Endpoint{Node: sink.ID, Port: "inputs"}},
		},
		[]Input{
			{Name: "left", Data: DataText, Target: Endpoint{Node: left.ID, Port: "input"}},
			{Name: "right", Data: DataText, Target: Endpoint{Node: right.ID, Port: "input"}},
		},
		[]Output{{Name: "output", Data: DataText, Source: Endpoint{Node: sink.ID, Port: "output"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := NewCatalog(branch, join)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	sets := program.ReadySets()
	if len(sets) != 2 || len(sets[0]) != 2 || sets[0][0].Node.ID != left.ID ||
		sets[0][1].Node.ID != right.ID || len(sets[1]) != 1 || sets[1][0].Node.ID != sink.ID {
		t.Fatalf("ready sets = %+v", sets)
	}
}
