package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const (
	visibilityRoot  NodeID = "root"
	visibilityLeft  NodeID = "left"
	visibilityRight NodeID = "right"
	visibilityJoin  NodeID = "join"
)

func visibilityProgram(t *testing.T) Program {
	t.Helper()
	const (
		sourceModule ModuleID = "visibility.source"
		branchModule ModuleID = "visibility.branch"
		joinModule   ModuleID = "visibility.join"
		valuePort    PortName = "value"
		valuesPort   PortName = "values"
	)
	placements := []Placement{PlacementHost}
	catalog, err := NewCatalog(
		Module{ID: sourceModule, Tasks: []Task{TaskInference}, Placements: placements,
			Outputs: []Port{{Name: valuePort, Data: DataTensor, Cardinality: CardinalityOne}}},
		Module{ID: branchModule, Tasks: []Task{TaskInference}, Placements: placements,
			Inputs:  []Port{{Name: valuePort, Data: DataTensor, Cardinality: CardinalityOne}},
			Outputs: []Port{{Name: valuePort, Data: DataTensor, Cardinality: CardinalityOne}}},
		Module{ID: joinModule, Tasks: []Task{TaskInference}, Placements: placements,
			Inputs:  []Port{{Name: valuesPort, Data: DataTensor, Cardinality: CardinalityMany}},
			Outputs: []Port{{Name: valuePort, Data: DataTensor, Cardinality: CardinalityOne}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "visibility-model")
	nodes := []Node{
		{ID: visibilityRoot, Module: sourceModule, Placement: PlacementHost},
		{ID: visibilityLeft, Module: branchModule, Placement: PlacementHost},
		{ID: visibilityRight, Module: branchModule, Placement: PlacementHost},
		{ID: visibilityJoin, Module: joinModule, Placement: PlacementHost},
	}
	edge := func(from NodeID, fromPort PortName, to NodeID, toPort PortName) Edge {
		return Edge{From: Endpoint{Node: from, Port: fromPort}, To: Endpoint{Node: to, Port: toPort}}
	}
	definition, err := NewDefinitionWithDependencies(
		TaskInference,
		[]Dependency{{Role: DependencyModel, Artifact: modelID}},
		nodes,
		[]Edge{
			edge(visibilityRoot, valuePort, visibilityLeft, valuePort),
			edge(visibilityRoot, valuePort, visibilityRight, valuePort),
			edge(visibilityLeft, valuePort, visibilityJoin, valuesPort),
			edge(visibilityRight, valuePort, visibilityJoin, valuesPort),
		}, nil,
		[]Output{{Name: valuePort, Data: DataTensor, Source: Endpoint{Node: visibilityJoin, Port: valuePort}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	return program
}

func visibilityActivation(t *testing.T, program Program, node NodeID) Activation {
	t.Helper()
	return Activation{
		Recipe: program.Definition().ID, Node: node,
		Event: testutil.ArtifactID(t, artifact.KindEvidence, string(node)),
	}
}

func TestInteractionVisibility(t *testing.T) {
	program := visibilityProgram(t)
	scope, err := program.InteractionScope(visibilityJoin)
	if err != nil || !scope.Valid() {
		t.Fatalf("interaction scope = (%+v, %v)", scope, err)
	}
	if !scope.Visibility.Allows(
		visibilityActivation(t, program, visibilityJoin),
		visibilityActivation(t, program, visibilityRoot),
	) {
		t.Fatal("join cannot see root")
	}
}

func TestSiblingHistoryIsolation(t *testing.T) {
	program := visibilityProgram(t)
	scope, err := program.InteractionScope(visibilityLeft)
	if err != nil {
		t.Fatal(err)
	}
	if scope.Visibility.Allows(
		visibilityActivation(t, program, visibilityLeft),
		visibilityActivation(t, program, visibilityRight),
	) {
		t.Fatal("left branch sees right branch")
	}
}

func TestAncestorHistoryVisibility(t *testing.T) {
	program := visibilityProgram(t)
	scope, err := program.InteractionScope(visibilityLeft)
	if err != nil {
		t.Fatal(err)
	}
	current := visibilityActivation(t, program, visibilityLeft)
	if !scope.Visibility.Allows(current, current) ||
		!scope.Visibility.Allows(current, visibilityActivation(t, program, visibilityRoot)) {
		t.Fatal("reflexive ancestor visibility is incomplete")
	}
}
