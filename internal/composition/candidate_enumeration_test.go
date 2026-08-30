package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestCompositionCandidateEnumerationRanksByResidualAndFitness pins the
// enumeration contract: donors shortlist from the exact catalog with the
// target's own model excluded, each candidate joins its measured residual
// and predicted fitness, the seam class derives from the typed contract,
// the adapter rung derives from the residual at mathematical boundaries,
// ranking is residual first then fitness then canonical identity,
// shortlisted seams without measurements are reported rather than
// silently dropped, and a target with no measured candidate refuses.
func TestCompositionCandidateEnumerationRanksByResidualAndFitness(t *testing.T) {
	target := descriptorComponent(t, "enumeration-target", "blk.10.ffn_gate.weight",
		descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2))
	graftDonor := descriptorComponent(t, "enumeration-graft", "blk.10.ffn_gate.weight",
		descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2))
	linearDonor := descriptorComponent(t, "enumeration-linear", "blk.10.ffn_gate.weight",
		descriptorStatistics(0.0012, 0.019, 0.011, 0.1, 0.2))
	residualDonor := descriptorComponent(t, "enumeration-residual", "blk.10.ffn_down.weight",
		descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2))
	expertDonor := descriptorComponent(t, "enumeration-expert", "blk.10.ffn_gate_exps.weight",
		descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2))
	bridgeDonor := descriptorComponent(t, "enumeration-bridge", "blk.10.ffn_gate.weight",
		descriptorStatistics(0.5, 3.0, 0.9, -0.4, 0.6))
	silentDonor := descriptorComponent(t, "enumeration-unmeasured", "blk.10.ffn_gate.weight",
		descriptorStatistics(0.0009, 0.021, 0.0095, 0.1, 0.2))
	index, err := NewExactComponentIndex([]CatalogComponent{
		target, graftDonor, linearDonor, residualDonor, expertDonor, bridgeDonor, silentDonor,
	})
	if err != nil {
		t.Fatal(err)
	}
	measurements := []DonorSeamMeasurement{
		{Donor: graftDonor.Model, Component: graftDonor.Name, Residual: 0, Fitness: 0.4},
		{Donor: linearDonor.Model, Component: linearDonor.Name, Residual: 0.3, Fitness: 0.9},
		{Donor: residualDonor.Model, Component: residualDonor.Name, Residual: 0.3, Fitness: 0.5},
		{Donor: expertDonor.Model, Component: expertDonor.Name, Residual: 0.6, Fitness: 0.8},
		{Donor: bridgeDonor.Model, Component: bridgeDonor.Name, Residual: 0.95, Fitness: 0.99},
	}
	candidates, unmeasured, err := EnumerateCompositionCandidates(index, target, measurements, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 5 {
		t.Fatalf("candidate census = %d: %+v", len(candidates), candidates)
	}
	for _, candidate := range candidates {
		if candidate.Donor == target.Model {
			t.Fatalf("the target composed with itself: %+v", candidate)
		}
	}
	order := make([]artifact.ID, 0, len(candidates))
	for _, candidate := range candidates {
		order = append(order, candidate.Donor)
	}
	expected := []artifact.ID{
		graftDonor.Model, linearDonor.Model, residualDonor.Model, expertDonor.Model, bridgeDonor.Model,
	}
	for index := range expected {
		if order[index] != expected[index] {
			t.Fatalf("ranking order[%d] = %s, want %s", index, order[index], expected[index])
		}
	}
	rungs := map[artifact.ID]AdapterRung{}
	seams := map[artifact.ID]SeamBoundary{}
	for _, candidate := range candidates {
		rungs[candidate.Donor] = candidate.Adapter
		seams[candidate.Donor] = candidate.Seam
	}
	if rungs[graftDonor.Model] != AdapterIdentity || rungs[linearDonor.Model] != AdapterLinear ||
		rungs[expertDonor.Model] != AdapterLinear || rungs[bridgeDonor.Model] != AdapterBridge {
		t.Fatalf("adapter rungs = %+v", rungs)
	}
	if seams[linearDonor.Model] != SeamLayerBoundary || seams[residualDonor.Model] != SeamResidualStream ||
		seams[expertDonor.Model] != SeamExpertBoundary {
		t.Fatalf("seam boundaries = %+v", seams)
	}
	if len(unmeasured) != 1 || unmeasured[0].Model != silentDonor.Model {
		t.Fatalf("unmeasured report = %+v", unmeasured)
	}

	if _, _, err := EnumerateCompositionCandidates(index, target, nil, 10); err == nil ||
		!strings.Contains(err.Error(), "measured residual and fitness") {
		t.Fatalf("unmeasured shortlist enumerated: %v", err)
	}
	broken := []DonorSeamMeasurement{{
		Donor: graftDonor.Model, Component: graftDonor.Name, Residual: -0.1, Fitness: 0.4,
	}}
	if _, _, err := EnumerateCompositionCandidates(index, target, broken, 10); err == nil ||
		!strings.Contains(err.Error(), "nonnegative residual") {
		t.Fatalf("negative residual enumerated: %v", err)
	}
	if _, _, err := EnumerateCompositionCandidates(nil, target, measurements, 10); err == nil {
		t.Fatal("nil index enumerated")
	}
}
