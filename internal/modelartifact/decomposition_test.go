package modelartifact

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/testutil"
)

// TestComponentDecompositionContract pins the component-decomposition row:
// typed components derive from tensor name, shape, dtype, family and modality
// across the organ axes; the document is deterministic, name-ordered,
// lineage-bound to the source model, and classification-only.
func TestComponentDecompositionContract(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "decomposition-source")
	tensors := []TensorFact{
		{Name: "blk.1.ffn_gate.weight", Shape: []uint64{896, 4864}, Storage: "bf16"},
		{Name: "token_embd.weight", Shape: []uint64{151936, 896}, Storage: "f32"},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{896, 896}, Storage: "bf16"},
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{896}, Storage: "f32"},
	}
	document, err := NewComponentDecomposition(model, "qwen2", "text", tensors)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := NewComponentDecomposition(model, "qwen2", "text", tensors)
	if err != nil || replay.ID != document.ID {
		t.Fatalf("replayed decomposition = (%v, %v), want identical identity %v", replay.ID, err, document.ID)
	}
	if len(document.Components) != len(tensors) {
		t.Fatalf("components = %d, want %d", len(document.Components), len(tensors))
	}
	byName := map[string]ComponentContract{}
	for index, component := range document.Components {
		if index > 0 && document.Components[index-1].Name >= component.Name {
			t.Fatalf("components unordered at %q", component.Name)
		}
		byName[component.Name] = component
	}
	expected := map[string]organ.Role{
		"token_embd.weight":      organ.RoleEmbedding,
		"blk.0.attn_q.weight":    organ.RoleQKV,
		"blk.1.ffn_gate.weight":  organ.RoleMLPGate,
		"blk.0.attn_norm.weight": organ.RoleNorm,
	}
	for name, role := range expected {
		component, ok := byName[name]
		if !ok || component.Contract.Role != role {
			t.Fatalf("component %s = %+v, want role %s", name, component.Contract, role)
		}
		if component.Contract.Modality != organ.ModalityText {
			t.Fatalf("component %s modality = %s, want text", name, component.Contract.Modality)
		}
		if component.Contract.DType == "" || component.Contract.DType == organ.DTypeUnknown {
			t.Fatalf("component %s dtype unclassified from storage %q", name, component.Storage)
		}
		if issues := organ.Validate(component.Contract); len(issues) != 0 {
			t.Fatalf("component %s contract issues: %+v", name, issues)
		}
	}
	if byName["blk.1.ffn_gate.weight"].Contract.DType != organ.DTypeBF16 ||
		byName["token_embd.weight"].Contract.DType != organ.DTypeFP32 {
		t.Fatal("storage dtype did not reach the contract axis")
	}

	lineage := document.Lineage()
	if lineage.Child != document.ID || lineage.Parent != model || lineage.Relation != artifact.RelationDerivedFrom {
		t.Fatalf("lineage = %+v, want derived-from source model", lineage)
	}
	batch, err := document.Batch("decompose:" + document.ID.String())
	if err != nil || len(batch.Contents) != 1 || len(batch.Lineage) != 1 {
		t.Fatalf("batch = (%+v, %v)", batch, err)
	}
	parsed, err := ParseComponentDecomposition(batch.Contents[0].Data)
	if err != nil || parsed.ID != document.ID || len(parsed.Components) != len(tensors) {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}

	if _, err := NewComponentDecomposition(model, "qwen2", "text", nil); err == nil {
		t.Fatal("empty decomposition accepted")
	}
	if _, err := NewComponentDecomposition(model, "qwen2", "text", []TensorFact{{Name: "  "}}); err == nil {
		t.Fatal("unnamed tensor accepted")
	}
	dataset := testutil.ArtifactID(t, artifact.KindDataset, "not-a-model")
	if _, err := NewComponentDecomposition(dataset, "qwen2", "text", tensors); err == nil {
		t.Fatal("non-model source accepted")
	}
}
