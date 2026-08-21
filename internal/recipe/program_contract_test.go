package recipe

import (
	"slices"
	"testing"
)

func TestCanonicalRecipeCompilationContract(t *testing.T) {
	definition, catalog := fixtureRecipe(t)
	program, err := CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	if !program.UsesCatalog(catalog) {
		t.Fatal("compiled program lost its immutable catalog authority")
	}
	stages := program.Stages()
	if len(stages) != len(definition.Nodes) || stages[0].Node.ID != "source" || stages[1].Node.ID != "sink" {
		t.Fatalf("execution order = %+v", stages)
	}

	definition.Nodes[0].Module = "uncatalogued.module"
	stages[0].Node.Module = "uncatalogued.module"
	compiled := program.Definition()
	if compiled.Nodes[0].Module == "uncatalogued.module" || program.Stages()[0].Node.Module == "uncatalogued.module" {
		t.Fatal("compiled program retained mutable caller storage")
	}

	base := program.Definition()
	reidentify := func(value Definition) Definition {
		t.Helper()
		rebuilt, err := NewDefinitionWithDependencies(
			value.Task, value.Dependencies, value.Nodes, value.Edges, value.Inputs, value.Outputs,
		)
		if err != nil {
			t.Fatal(err)
		}
		return rebuilt
	}
	tests := map[string]func(Definition, *Catalog) (Definition, *Catalog){
		"identity mismatch": func(value Definition, authority *Catalog) (Definition, *Catalog) {
			value.Task = TaskTraining
			return value, authority
		},
		"uncatalogued module": func(value Definition, authority *Catalog) (Definition, *Catalog) {
			value.Nodes[0].Module = "uncatalogued.module"
			return reidentify(value), authority
		},
		"ill-typed edge": func(value Definition, _ *Catalog) (Definition, *Catalog) {
			modules := catalog.Modules()
			for index := range modules {
				if modules[index].ID == fixtureSinkModule {
					modules[index].Inputs[0].Data = DataImage
				}
			}
			authority, err := NewCatalog(modules...)
			if err != nil {
				t.Fatal(err)
			}
			return value, authority
		},
		"unreachable node": func(value Definition, authority *Catalog) (Definition, *Catalog) {
			value.Nodes = append(value.Nodes, Node{ID: "dead", Module: fixtureSourceModule, Placement: PlacementHost})
			return reidentify(value), authority
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			candidate := cloneProgramDefinition(base)
			candidate, authority := mutate(candidate, catalog)
			if _, err := CompileProgram(candidate, authority); err == nil {
				t.Fatal("invalid definition compiled")
			}
		})
	}

	nodes := slices.Clone(base.Nodes)
	slices.Reverse(nodes)
	rebuilt, err := NewDefinitionWithDependencies(base.Task, base.Dependencies, nodes, base.Edges, base.Inputs, base.Outputs)
	if err != nil {
		t.Fatal(err)
	}
	if rebuilt.ID != base.ID {
		t.Fatalf("normalized definitions have different identities: %s != %s", rebuilt.ID, base.ID)
	}
	if _, err := CompileProgram(rebuilt, catalog); err != nil {
		t.Fatalf("normalized definition did not compile: %v", err)
	}
}
