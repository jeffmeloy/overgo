package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
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
	admission := testcheck.Must(AdmitTool(program))(t)
	if admission.Module != testToolModule || admission.Recipe != program.Definition().ID {
		t.Fatalf("admission = %+v", admission)
	}
}

func testToolProgram(t *testing.T) Program {
	t.Helper()
	model := testutil.ArtifactID(t, artifact.KindModel, "tool-model")
	module := ToolModule(testToolModule, PlacementHost)
	node := Node{ID: "execute", Module: module.ID, Placement: PlacementHost}
	definition := testcheck.Must(NewDefinitionWithDependencies(
		TaskInference,
		[]Dependency{{Role: DependencyModel, Artifact: model}},
		[]Node{node}, nil,
		[]Input{{Name: ToolCallPort, Data: DataToolCall, Target: Endpoint{Node: node.ID, Port: ToolCallPort}}},
		[]Output{{Name: ToolResultPort, Data: DataToolResult, Source: Endpoint{Node: node.ID, Port: ToolResultPort}}},
	))(t)
	catalog := testcheck.Must(NewCatalog(module))(t)
	program := testcheck.Must(CompileProgram(definition, catalog))(t)
	return program
}
