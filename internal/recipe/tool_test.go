package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const testToolModule ModuleID = "lookup"

func TestToolModuleContract(t *testing.T) {
	module := ToolModule(testToolModule, PlacementHost)
	if _, err := NewCatalog(module); err != nil {
		t.Fatal(err)
	}
	module.Outputs[0].Data = DataText
	if err := validateToolModule(module); err == nil {
		t.Fatal("incompatible tool module accepted")
	}
}

func TestToolRecipeAdmission(t *testing.T) {
	program := testToolProgram(t)
	admission, err := AdmitTool(program)
	if err != nil {
		t.Fatal(err)
	}
	if admission.Module != testToolModule || admission.Recipe != program.Definition().ID {
		t.Fatalf("admission = %+v", admission)
	}
}

func testToolProgram(t *testing.T) Program {
	t.Helper()
	model := testutil.ArtifactID(t, artifact.KindModel, "tool-model")
	module := ToolModule(testToolModule, PlacementHost)
	node := Node{ID: "execute", Module: module.ID, Placement: PlacementHost}
	definition, err := NewDefinitionWithDependencies(
		TaskInference,
		[]Dependency{{Role: DependencyModel, Artifact: model}},
		[]Node{node}, nil,
		[]Input{{Name: ToolCallPort, Data: DataToolCall, Target: Endpoint{Node: node.ID, Port: ToolCallPort}}},
		[]Output{{Name: ToolResultPort, Data: DataToolResult, Source: Endpoint{Node: node.ID, Port: ToolResultPort}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := NewCatalog(module)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	return program
}
