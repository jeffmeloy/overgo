package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/tensorstats"
	"overgo/internal/testutil"
)

func descriptorStatistics(median, iqr, l2, tau3, tau4 float64) *tensorstats.Characterization {
	return &tensorstats.Characterization{
		Median: median, InterquartileRange: iqr,
		LMoments: tensorstats.LMoments{L2: l2, Tau3: tau3, Tau4: tau4},
	}
}

func descriptorComponent(
	t *testing.T, model, name string, statistics *tensorstats.Characterization,
) CatalogComponent {
	t.Helper()
	return CatalogComponent{
		Model: testutil.ArtifactID(t, artifact.KindModel, model), Name: name,
		Contract:   organ.Classify(name, "bf16", "qwen2", "text", ""),
		Statistics: statistics,
	}
}

// TestCanonicalComponentDescriptorComparable pins the fixed-schema
// contract: the descriptor carries the typed contract verbatim and the
// characterization as unit-normalized buckets — location and spread are
// divided by the component's own L-scale, so two components that differ
// only by weight magnitude produce identical normalized fields while the
// scale bucket records the magnitude; an unmeasured component holds the
// sentinel in every bucket, and a component without identity or with a
// degenerate scale refuses.
func TestCanonicalComponentDescriptorComparable(t *testing.T) {
	base := descriptorComponent(
		t, "descriptor-model-a", "blk.10.ffn_gate.weight",
		descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2),
	)
	scaled := descriptorComponent(
		t, "descriptor-model-b", "blk.10.ffn_gate.weight",
		descriptorStatistics(1.0, 20.0, 10.0, 0.1, 0.2),
	)
	baseDescriptor, err := CanonicalDescriptor(base)
	if err != nil {
		t.Fatal(err)
	}
	scaledDescriptor, err := CanonicalDescriptor(scaled)
	if err != nil {
		t.Fatal(err)
	}
	if !baseDescriptor.Measured || !scaledDescriptor.Measured {
		t.Fatalf("measured components lost their measurements: %+v %+v", baseDescriptor, scaledDescriptor)
	}
	if baseDescriptor.Location != scaledDescriptor.Location ||
		baseDescriptor.Spread != scaledDescriptor.Spread ||
		baseDescriptor.Tau3 != scaledDescriptor.Tau3 || baseDescriptor.Tau4 != scaledDescriptor.Tau4 {
		t.Fatalf(
			"unit-normalized fields differ across a pure scale change:\n%+v\n%+v",
			baseDescriptor, scaledDescriptor,
		)
	}
	if baseDescriptor.Scale == scaledDescriptor.Scale {
		t.Fatalf("scale bucket lost the thousandfold magnitude difference: %d", baseDescriptor.Scale)
	}
	if baseDescriptor.Role != scaledDescriptor.Role || baseDescriptor.Modality != scaledDescriptor.Modality {
		t.Fatalf("typed contract drifted: %+v vs %+v", baseDescriptor, scaledDescriptor)
	}

	unmeasured, err := CanonicalDescriptor(descriptorComponent(t, "descriptor-model-a", "blk.11.attn_q.weight", nil))
	if err != nil {
		t.Fatal(err)
	}
	if unmeasured.Measured || unmeasured.Scale != descriptorUnmeasuredBucket ||
		unmeasured.Location != descriptorUnmeasuredBucket || unmeasured.Tau4 != descriptorUnmeasuredBucket {
		t.Fatalf("unmeasured descriptor = %+v", unmeasured)
	}

	nameless := base
	nameless.Name = ""
	if _, err := CanonicalDescriptor(nameless); err == nil ||
		!strings.Contains(err.Error(), "model and a name") {
		t.Fatalf("nameless component described: %v", err)
	}
	degenerate := descriptorComponent(
		t, "descriptor-model-a", "blk.12.ffn_up.weight",
		descriptorStatistics(0.1, 0.2, 0, 0.1, 0.2),
	)
	if _, err := CanonicalDescriptor(degenerate); err == nil ||
		!strings.Contains(err.Error(), "positive L-scale") {
		t.Fatalf("degenerate scale described: %v", err)
	}
}

// TestExactComponentRetrievalDeterministic pins the retrieval contract:
// the inverted index over canonical descriptors is a pure function of
// catalog content — reordering the input changes nothing — repeated
// queries return identical shortlists, an exact self-match ranks first
// with complete overlap, ranking touches only the query's own postings,
// and the limit bounds the shortlist.
func TestExactComponentRetrievalDeterministic(t *testing.T) {
	components := []CatalogComponent{
		descriptorComponent(t, "retrieval-model-a", "blk.10.ffn_gate.weight",
			descriptorStatistics(0.001, 0.02, 0.01, 0.1, 0.2)),
		descriptorComponent(t, "retrieval-model-b", "blk.10.ffn_gate.weight",
			descriptorStatistics(0.011, 0.21, 0.105, 0.1, 0.2)),
		descriptorComponent(t, "retrieval-model-c", "blk.10.ffn_gate.weight",
			descriptorStatistics(0.5, 3.0, 0.9, -0.4, 0.6)),
		descriptorComponent(t, "retrieval-model-a", "blk.10.attn_q.weight",
			descriptorStatistics(0.002, 0.03, 0.02, 0.05, 0.15)),
		descriptorComponent(t, "retrieval-model-b", "output_norm.weight", nil),
	}
	forward, err := NewExactComponentIndex(components)
	if err != nil {
		t.Fatal(err)
	}
	reversed := make([]CatalogComponent, len(components))
	for index, component := range components {
		reversed[len(components)-1-index] = component
	}
	backward, err := NewExactComponentIndex(reversed)
	if err != nil {
		t.Fatal(err)
	}
	query := components[0]
	first, err := forward.Search(query, 4)
	if err != nil {
		t.Fatal(err)
	}
	second, err := backward.Search(query, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) == 0 || len(first) != len(second) {
		t.Fatalf("shortlists differ in size: %d vs %d", len(first), len(second))
	}
	for index := range first {
		if first[index] != second[index] {
			t.Fatalf("input order changed the shortlist at %d:\n%+v\n%+v", index, first[index], second[index])
		}
	}
	again, err := forward.Search(query, 4)
	if err != nil {
		t.Fatal(err)
	}
	for index := range first {
		if first[index] != again[index] {
			t.Fatalf("repeated query drifted at %d", index)
		}
	}
	if first[0].Descriptor.Model != query.Model || first[0].Descriptor.Name != query.Name ||
		first[0].Relevance != 1 {
		t.Fatalf("self-match did not rank first with complete overlap: %+v", first[0])
	}
	if first[1].Descriptor.Model != components[1].Model {
		t.Fatalf("the twin component did not rank second: %+v", first[1])
	}
	limited, err := forward.Search(query, 2)
	if err != nil || len(limited) != 2 {
		t.Fatalf("limit was not respected: (%d, %v)", len(limited), err)
	}
	if _, err := forward.Search(query, 0); err == nil {
		t.Fatal("nonpositive limit searched")
	}
	if _, err := NewExactComponentIndex(nil); err == nil {
		t.Fatal("empty catalog indexed")
	}
}
