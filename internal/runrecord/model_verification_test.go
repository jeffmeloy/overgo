package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestModelVerificationLedger pins the verification-ledger contract: claims
// are typed and evidenced or they do not exist, every claim names the
// repository commit that verified it, training claims carry their dataset
// and span, identity is order-canonical, and the comparison matrix derives
// from committed records -- strongest evidenced tier per capability, union
// evidence at the winning tier, deterministic regardless of scan order.
// No markdown, no drift.
func TestModelVerificationLedger(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "verified-model")
	other := testutil.ArtifactID(t, artifact.KindModel, "other-model")
	golden := testutil.ArtifactID(t, artifact.KindEvidence, "serving-golden")
	smoke := testutil.ArtifactID(t, artifact.KindEvidence, "training-smoke")
	floor := testutil.ArtifactID(t, artifact.KindEvidence, "capability-floor")
	dataset := testutil.ArtifactID(t, artifact.KindDatasetShard, "training-dataset")
	commit := strings.Repeat("ab", 20)

	record, err := NewModelVerification(model, "Carbon-500M", []CapabilityClaim{
		{Capability: "training", Tier: TierRealArtifactSmoke, Commit: commit,
			Dataset: dataset, SpanSteps: 60, SpanTokens: 480_000, Evidence: []artifact.ID{smoke}},
		{Capability: "inference", Tier: TierExactGolden, Commit: commit, Evidence: []artifact.ID{golden}},
	})
	if err != nil {
		t.Fatal(err)
	}
	replay, err := NewModelVerification(model, "Carbon-500M", []CapabilityClaim{
		{Capability: "inference", Tier: TierExactGolden, Commit: commit, Evidence: []artifact.ID{golden}},
		{Capability: "training", Tier: TierRealArtifactSmoke, Commit: commit,
			Dataset: dataset, SpanSteps: 60, SpanTokens: 480_000, Evidence: []artifact.ID{smoke}},
	})
	if err != nil || replay.ID != record.ID {
		t.Fatalf("identity not order-canonical: (%v, %v)", replay.ID, err)
	}
	if record.Claims[0].Capability != "inference" {
		t.Fatalf("claims not sorted: %+v", record.Claims)
	}
	content, err := record.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseModelVerification(content.Data)
	if err != nil || parsed.ID != record.ID || len(parsed.Claims) != 2 {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}
	if parsed.Claims[1].Dataset != dataset || parsed.Claims[1].SpanSteps != 60 || parsed.Claims[1].Commit != commit {
		t.Fatalf("training provenance lost: %+v", parsed.Claims[1])
	}
	batch, err := record.Batch("verification/" + record.ID.String())
	if err != nil || len(batch.Lineage) != 4 {
		t.Fatalf("batch lineage = %d edges (%v), want model, dataset and two evidence parents", len(batch.Lineage), err)
	}

	valid := CapabilityClaim{Capability: "inference", Tier: TierExactGolden, Commit: commit, Evidence: []artifact.ID{golden}}
	for name, claims := range map[string][]CapabilityClaim{
		"no claims":             {},
		"tier without evidence": {{Capability: "inference", Tier: TierExactGolden, Commit: commit}},
		"invalid tier":          {{Capability: "inference", Tier: "vibes", Commit: commit, Evidence: []artifact.ID{golden}}},
		"uppercase capability":  {{Capability: "Inference", Tier: TierExactGolden, Commit: commit, Evidence: []artifact.ID{golden}}},
		"model as evidence":     {{Capability: "inference", Tier: TierExactGolden, Commit: commit, Evidence: []artifact.ID{model}}},
		"no verifier commit":    {{Capability: "inference", Tier: TierExactGolden, Evidence: []artifact.ID{golden}}},
		"uppercase commit":      {{Capability: "inference", Tier: TierExactGolden, Commit: strings.Repeat("AB", 20), Evidence: []artifact.ID{golden}}},
		"training without dataset": {{Capability: "training", Tier: TierRealArtifactSmoke, Commit: commit,
			SpanSteps: 60, Evidence: []artifact.ID{smoke}}},
		"training without span": {{Capability: "training", Tier: TierRealArtifactSmoke, Commit: commit,
			Dataset: dataset, Evidence: []artifact.ID{smoke}}},
		"span without dataset": {{Capability: "inference", Tier: TierExactGolden, Commit: commit,
			SpanTokens: 100, Evidence: []artifact.ID{golden}}},
		"peak memory without wall": {{Capability: "inference", Tier: TierExactGolden, Commit: commit,
			PeakDeviceBytes: 1 << 30, Evidence: []artifact.ID{golden}}},
		"context without wall": {{Capability: "inference", Tier: TierExactGolden, Commit: commit,
			ContextTokens: 8192, Evidence: []artifact.ID{golden}}},
		"duplicate capability": {valid, {Capability: "inference", Tier: TierRealArtifactSmoke, Commit: commit, Evidence: []artifact.ID{smoke}}},
	} {
		if _, err := NewModelVerification(model, "Carbon-500M", claims); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if _, err := NewModelVerification(golden, "not-a-model", []CapabilityClaim{valid}); err == nil {
		t.Fatal("evidence identity accepted as a model")
	}

	// Matrix: a later record raises inference to capability-measured and adds
	// a second model; the derived matrix takes the strongest evidenced tier
	// per capability and unions evidence only at the winning tier, invariant
	// to record order.
	upgrade, err := NewModelVerification(model, "Carbon-500M", []CapabilityClaim{
		{Capability: "inference", Tier: TierCapabilityMeasured, Commit: commit,
			ContextTokens: 8192, WallNS: 4_750_000_000, PeakDeviceBytes: 2 << 30,
			Evidence: []artifact.ID{floor}},
	})
	if err != nil {
		t.Fatal(err)
	}
	sibling, err := NewModelVerification(other, "Qwen2.5-0.5B", []CapabilityClaim{valid})
	if err != nil {
		t.Fatal(err)
	}
	matrix := VerificationMatrix([]ModelVerification{record, upgrade, sibling})
	if len(matrix) != 2 || matrix[0].Name != "Carbon-500M" || matrix[1].Name != "Qwen2.5-0.5B" {
		t.Fatalf("matrix rows = %+v", matrix)
	}
	carbon := matrix[0]
	if len(carbon.Capabilities) != 2 || carbon.Capabilities[0].Capability != "inference" ||
		carbon.Capabilities[0].Tier != TierCapabilityMeasured ||
		len(carbon.Capabilities[0].Evidence) != 1 || carbon.Capabilities[0].Evidence[0] != floor {
		t.Fatalf("inference row = %+v, want capability-measured grounded by the floor only", carbon.Capabilities)
	}
	if carbon.Capabilities[0].ContextTokens != 8192 || carbon.Capabilities[0].WallNS != 4_750_000_000 ||
		carbon.Capabilities[0].PeakDeviceBytes != 2<<30 {
		t.Fatalf("measurement lost through the matrix: %+v", carbon.Capabilities[0])
	}
	training := carbon.Capabilities[1]
	if training.Capability != "training" || training.Tier != TierRealArtifactSmoke ||
		training.Dataset != dataset || training.SpanSteps != 60 || training.Commit != commit {
		t.Fatalf("training row lost provenance: %+v", training)
	}
	measuredSibling, err := NewModelVerification(other, "Qwen2.5-0.5B", []CapabilityClaim{
		{Capability: "inference", Tier: TierExactGolden, Commit: commit,
			ContextTokens: 2625, WallNS: 2_983_801_000, PeakDeviceBytes: 1_368_364_544,
			Evidence: []artifact.ID{smoke}},
	})
	if err != nil {
		t.Fatal(err)
	}
	displaced := VerificationMatrix([]ModelVerification{sibling, measuredSibling})
	cell := displaced[0].Capabilities[0]
	if cell.Tier != TierExactGolden || cell.WallNS != 2_983_801_000 || len(cell.Evidence) != 2 {
		t.Fatalf("measured claim did not displace unmeasured provenance at equal tier: %+v", cell)
	}
	reordered := VerificationMatrix([]ModelVerification{sibling, upgrade, record})
	if len(reordered) != 2 || reordered[0].Capabilities[0].Tier != carbon.Capabilities[0].Tier ||
		len(reordered[0].Capabilities[0].Evidence) != len(carbon.Capabilities[0].Evidence) {
		t.Fatal("matrix depends on record order")
	}
	if TierCapabilityMeasured.Rank() <= TierExactGolden.Rank() || VerificationTier("vibes").Rank() != 0 {
		t.Fatal("tier ranking broken")
	}
}
