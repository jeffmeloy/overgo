package composition

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/tensorstats"
	"overgo/internal/testutil"
)

// TestHypervectorRetrievalRanksKnownBridges pins the ported HDC contract:
// the index signs the lexical organ contract together with the quantized
// distributional signal; a query component ranks its known cross-model
// counterpart (same role, similar distribution) above same-model different
// -role components and above different-role different-distribution noise;
// width derives from catalog size at the measured resolvable margin; and
// hits convert to blocked-proposal candidates excluding within-model rows.
func TestHypervectorRetrievalRanksKnownBridges(t *testing.T) {
	modelA := testutil.ArtifactID(t, artifact.KindModel, "hdc-model-a")
	modelB := testutil.ArtifactID(t, artifact.KindModel, "hdc-model-b")
	statistics := func(median, iqr, l2, tau3, tau4 float64) *tensorstats.Characterization {
		return &tensorstats.Characterization{
			Median: median, InterquartileRange: iqr,
			LMoments: tensorstats.LMoments{L2: l2, Tau3: tau3, Tau4: tau4},
		}
	}
	contract := func(name string) organ.Contract {
		return organ.Classify(name, "bf16", "qwen2", "text", "")
	}
	gateA := CatalogComponent{
		Model: modelA, Name: "blk.10.ffn_gate.weight",
		Contract: contract("blk.10.ffn_gate.weight"), Statistics: statistics(0.001, 0.02, 0.01, 0.1, 0.2),
	}
	gateB := CatalogComponent{
		Model: modelB, Name: "blk.12.ffn_gate.weight",
		Contract: contract("blk.12.ffn_gate.weight"), Statistics: statistics(0.0012, 0.025, 0.012, 0.1, 0.2),
	}
	attentionA := CatalogComponent{
		Model: modelA, Name: "blk.10.attn_q.weight",
		Contract: contract("blk.10.attn_q.weight"), Statistics: statistics(0.001, 0.02, 0.01, 0.1, 0.2),
	}
	noise := CatalogComponent{
		Model: modelB, Name: "output_norm.weight",
		Contract: contract("output_norm.weight"), Statistics: statistics(500, 900, 700, -0.9, -0.8),
	}
	index, err := NewHypervectorIndex([]CatalogComponent{gateB, attentionA, noise})
	if err != nil {
		t.Fatal(err)
	}
	if index.Len() != 3 {
		t.Fatalf("indexed %d components", index.Len())
	}
	hits, err := index.Search(gateA, 3)
	if err != nil {
		t.Fatal(err)
	}
	if hits[0].Component.Name != gateB.Name {
		t.Fatalf("top hit = %s (%.4f), want the cross-model gate counterpart", hits[0].Component.Name, hits[0].Relevance)
	}
	if hits[0].Relevance <= hits[1].Relevance || hits[0].Relevance <= 0 {
		t.Fatalf("counterpart relevance %.4f does not dominate %.4f", hits[0].Relevance, hits[1].Relevance)
	}
	if last := hits[len(hits)-1]; last.Component.Name != noise.Name {
		t.Fatalf("weakest hit = %s, want the distributional noise row", last.Component.Name)
	}

	repeat, err := index.Search(gateA, 3)
	if err != nil || repeat[0].Relevance != hits[0].Relevance {
		t.Fatalf("retrieval not deterministic: (%v, %v)", repeat, err)
	}

	if width := HypervectorDimensionsFor(3); width != hdcDefaultDimensions {
		t.Fatalf("small catalog width = %d, want the floor %d", width, hdcDefaultDimensions)
	}
	if width := HypervectorDimensionsFor(100_000); width <= hdcDefaultDimensions || width%hdcWordBits != 0 {
		t.Fatalf("large catalog width = %d, want derived word-aligned growth", width)
	}

	candidates := Candidates(hits, modelA)
	for _, candidate := range candidates {
		if candidate.Donor == modelA {
			t.Fatal("within-model hit survived candidate conversion")
		}
		if candidate.Distance < 0 {
			t.Fatalf("negative distance %f", candidate.Distance)
		}
	}
	if len(candidates) != 2 || candidates[0].Component != gateB.Name {
		t.Fatalf("candidates = %+v, want the cross-model counterpart first", candidates)
	}

	if _, err := NewHypervectorIndex(nil); err == nil {
		t.Fatal("empty catalog accepted")
	}
	if _, err := index.Search(gateA, 0); err == nil {
		t.Fatal("zero limit accepted")
	}
}
