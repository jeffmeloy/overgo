package agenttool

import (
	"strings"
	"testing"
)

func propensityManual(t *testing.T, name string, effect Effect) Manual {
	t.Helper()
	manual, err := NewManual(Manual{
		Name: name, Description: "Propensity gate fixture.", Effect: effect,
		Transport: Transport{Kind: TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	return manual
}

// TestPositiveToolSteeringRequiresInspectionOnlyCatalog pins the phase gate:
// a positive alpha exists only over an action space of inspections, and the
// masked proposal space drops every non-inspection manual.
func TestPositiveToolSteeringRequiresInspectionOnlyCatalog(t *testing.T) {
	read := propensityManual(t, "store.read", EffectInspection)
	head := propensityManual(t, "store.head", EffectInspection)
	write := propensityManual(t, "store.write", EffectMutation)
	if err := InspectionOnlySteering([]Manual{read, head}, nil); err != nil {
		t.Fatalf("inspection-only space refused: %v", err)
	}
	if err := InspectionOnlySteering([]Manual{read, write}, nil); err == nil ||
		!strings.Contains(err.Error(), write.Name) {
		t.Fatalf("mutation-exposing space admitted: %v", err)
	}
	if alpha := SteeringAlpha(0.4, []Manual{read, head}, nil); alpha != 0.4 {
		t.Fatalf("inspection-only alpha = %v, want 0.4", alpha)
	}
	masked := InspectionOnlyManuals([]Manual{read, write, head})
	if len(masked) != 2 || masked[0].Name != read.Name || masked[1].Name != head.Name {
		t.Fatalf("masked space = %+v", masked)
	}
}

// TestMutationProposalUsesZeroAlpha pins the identity rule: any decode
// exposing a mutation or an opaque effect runs at alpha zero, whatever the
// selector derived.
func TestMutationProposalUsesZeroAlpha(t *testing.T) {
	read := propensityManual(t, "store.read", EffectInspection)
	write := propensityManual(t, "store.write", EffectMutation)
	if alpha := SteeringAlpha(0.7, []Manual{read, write}, nil); alpha != 0 {
		t.Fatalf("mutation decode alpha = %v, want 0", alpha)
	}
	if alpha := SteeringAlpha(-0.2, []Manual{read}, nil); alpha != 0 {
		t.Fatalf("negative selected alpha = %v, want 0", alpha)
	}
	if alpha := SteeringAlpha(0, []Manual{read}, nil); alpha != 0 {
		t.Fatalf("zero selected alpha = %v, want 0", alpha)
	}
}

// TestCapabilityProxyCannotHideMutationFromPropensityGate pins the dynamic
// hole: the proxy manual is itself an inspection, but a proxy call
// dispatches whatever the active catalog resolves, so a visible proxy makes
// the reachable set part of the gated action space.
func TestCapabilityProxyCannotHideMutationFromPropensityGate(t *testing.T) {
	proxy, err := CapabilityProxyManual()
	if err != nil {
		t.Fatal(err)
	}
	read := propensityManual(t, "store.read", EffectInspection)
	write := propensityManual(t, "store.write", EffectMutation)
	if err := InspectionOnlySteering([]Manual{proxy, read}, []Manual{read}); err != nil {
		t.Fatalf("inspection-only proxy space refused: %v", err)
	}
	if err := InspectionOnlySteering([]Manual{proxy, read}, []Manual{read, write}); err == nil ||
		!strings.Contains(err.Error(), "proxy") {
		t.Fatalf("proxy-reachable mutation admitted: %v", err)
	}
	if alpha := SteeringAlpha(0.5, []Manual{proxy}, []Manual{write}); alpha != 0 {
		t.Fatalf("proxy-hidden mutation alpha = %v, want 0", alpha)
	}
	// Masking is stricter than gating: the steered proposal space drops
	// the proxy entirely, because its dispatch set is dynamic.
	masked := InspectionOnlyManuals([]Manual{proxy, read, write})
	if len(masked) != 1 || masked[0].Name != read.Name {
		t.Fatalf("masked proxy space = %+v", masked)
	}
}
