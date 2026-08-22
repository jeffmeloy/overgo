package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestComponentSessionPlanRepresentationBridge(t *testing.T) {
	sourceModel := testutil.ArtifactID(t, artifact.KindModel, "source-model")
	targetModel := testutil.ArtifactID(t, artifact.KindModel, "target-model")
	sourceContract := testutil.ArtifactID(t, artifact.KindProfile, "source-contract")
	targetContract := testutil.ArtifactID(t, artifact.KindProfile, "target-contract")
	bridge := testutil.ArtifactID(t, artifact.KindAdapter, "representation-bridge")
	compiler := RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementHost, TargetPlacement: recipe.PlacementDevice,
		SourceResidency: recipe.ResidencyHostCache, TargetResidency: recipe.ResidencyDeviceNative,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}
	definition, err := compiler.Definition(sourceModel, targetModel, sourceContract, targetContract, bridge)
	if err != nil {
		t.Fatal(err)
	}
	bindings := []struct {
		role recipe.DependencyRole
		slot uint32
		want artifact.ID
	}{
		{recipe.DependencyModel, sourceModelSlot, sourceModel},
		{recipe.DependencyModel, targetModelSlot, targetModel},
		{recipe.DependencyProfile, sourceModelSlot, sourceContract},
		{recipe.DependencyProfile, targetModelSlot, targetContract},
		{recipe.DependencyAdapter, sourceModelSlot, bridge},
	}
	for _, binding := range bindings {
		got, ok := definition.Dependency(binding.role, binding.slot)
		if !ok || got != binding.want {
			t.Fatalf("dependency %s/%d = %s, %t; want %s", binding.role, binding.slot, got, ok, binding.want)
		}
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	extents := map[artifact.ID]uint64{sourceModel: 13, targetModel: 17}
	resources, err := CompileComponentSessionPlanWithExtents(
		context.Background(), program,
		func(_ context.Context, id artifact.ID) (uint64, error) { return extents[id], nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(resources.Components) != len(extents) || resources.ArtifactBytes != 30 {
		t.Fatalf("component resources = %+v", resources)
	}
	source, target := resources.Components[0], resources.Components[1]
	if source.Node != "capture" || source.Model != sourceModel ||
		source.Session != recipe.SessionCapacity || source.Residency != recipe.ResidencyHostCache {
		t.Fatalf("source component = %+v", source)
	}
	if target.Node != "inject" || target.Model != targetModel ||
		target.Session != recipe.SessionRequest || target.Residency != recipe.ResidencyDeviceNative {
		t.Fatalf("target component = %+v", target)
	}
	if !resources.RequestScoped() {
		t.Fatal("target request lifetime was not preserved")
	}
	content, err := definition.Content()
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := recipe.ParseDefinition(content)
	if err != nil {
		t.Fatal(err)
	}
	if roundTrip.ID != definition.ID {
		t.Fatalf("recipe identity changed: got %s want %s", roundTrip.ID, definition.ID)
	}
}
