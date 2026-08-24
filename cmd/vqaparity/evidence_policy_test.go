package main

import "testing"

func TestCampaignEvidenceCoversTypedContracts(t *testing.T) {
	if len(campaignEvidence.Acceptance) != len(acceptanceClasses) || len(campaignEvidence.Measurement) != len(measurementStages) {
		t.Fatalf("evidence coverage = %d acceptance, %d measurement", len(campaignEvidence.Acceptance), len(campaignEvidence.Measurement))
	}
	for _, class := range acceptanceClasses {
		_ = acceptance(class)
	}
	for _, stage := range measurementStages {
		_ = measurement(stage)
	}
}

func TestProbeValuesCheckUsesEvidenceContract(t *testing.T) {
	contract := acceptance(acceptNorm)
	reference := []float64{1}
	limit := contract.Absolute + contract.Relative*reference[0]
	if _, err := probeValuesCheck("accepted", []float32{float32(reference[0] + limit/2)}, reference, acceptNorm); err != nil {
		t.Fatal(err)
	}
	if _, err := probeValuesCheck("refused", []float32{float32(reference[0] + 2*limit)}, reference, acceptNorm); err == nil {
		t.Fatal("probe outside evidence contract accepted")
	}
}

func TestMeasureUsesEvidenceBudget(t *testing.T) {
	budget := measurement(measureTerminal)
	runs := 0
	if _, err := measure(budget, func() error { runs++; return nil }); err != nil {
		t.Fatal(err)
	}
	if want := budget.Warmup + budget.Samples; runs != want {
		t.Fatalf("runs = %d, want %d", runs, want)
	}
}
