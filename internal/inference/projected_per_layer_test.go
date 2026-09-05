package inference

import (
	"slices"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tokenizer"
)

func TestProjectedPerLayerTokenIdentity(t *testing.T) {
	rows := []uint32{2, 19, 20, 7}
	plan := projectedRequestPlan{overridePolicy: model.EmbeddingOverrideRawScaled,
		overrides: []EmbeddingOverride{{TokenIndex: 1}, {TokenIndex: 2}}}
	selected, err := plan.perLayerEmbeddingRows(rows, 3)
	if err != nil || !slices.Equal(selected, []uint32{2, 3, 3, 7}) {
		t.Fatalf("media identity=%v error=%v", selected, err)
	}
	if !slices.Equal(rows, []uint32{2, 19, 20, 7}) {
		t.Fatalf("caller prompt changed: %v", rows)
	}
	if _, err := plan.perLayerEmbeddingRows(rows, tokenizer.NullToken); err == nil {
		t.Fatal("missing padding declaration accepted")
	}
	plan.overrides = []EmbeddingOverride{{TokenIndex: uint32(len(rows))}}
	if _, err := plan.perLayerEmbeddingRows(rows, 3); err == nil {
		t.Fatal("out-of-range media position accepted")
	}
	plan.overrides = nil
	selected, err = plan.perLayerEmbeddingRows(rows, tokenizer.NullToken)
	if err != nil || !slices.Equal(selected, rows) {
		t.Fatalf("text-only identities changed: %v %v", selected, err)
	}
	plan.overridePolicy = model.EmbeddingOverrideMappedBase
	plan.overrides = []EmbeddingOverride{{TokenIndex: 1}}
	selected, err = plan.perLayerEmbeddingRows(rows, tokenizer.NullToken)
	if err != nil || !slices.Equal(selected, rows) {
		t.Fatalf("other override policy changed: %v %v", selected, err)
	}
}
