package server

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestSimilarComponentsSpanStoreCatalog pins the component-catalog row: the
// similarity endpoint retrieves from the store's committed cross-model
// measurement catalog -- neighbors span every measured model, labeled and
// inspectable -- and the algorithm is unchanged (rank-footrule over
// distribution-free descriptors).
func TestSimilarComponentsSpanStoreCatalog(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(overgodb.Create(root))(t)
	ctx := t.Context()
	generator := rand.New(rand.NewSource(11))
	for index, name := range []string{"catalog-model-alpha", "catalog-model-beta"} {
		modelID := testutil.ArtifactID(t, artifact.KindModel, name)
		directory := filepath.Join(t.TempDir(), name)
		testcheck.NoError(t, os.MkdirAll(directory, 0o755))
		tensors := map[string][]float32{}
		shapes := map[string][]int{}
		facts := make([]modelartifact.TensorFact, 0, 3)
		for _, tensor := range []string{"blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "output_norm.weight"} {
			values := make([]float32, 64)
			for i := range values {
				values[i] = float32(generator.NormFloat64()) * float32(index+1)
			}
			tensors[tensor], shapes[tensor] = values, []int{8, 8}
			facts = append(facts, modelartifact.TensorFact{
				Name: tensor, Shape: []uint64{8, 8}, Storage: "f32", Bytes: 64 * 4,
			})
		}
		testcheck.NoError(t, safetensors.Save(filepath.Join(directory, "model.safetensors"), tensors, shapes, nil))
		inventory := testcheck.Must(modelartifact.NewTensorInventoryDocument(modelID, modelartifact.TensorFormatSafetensors, facts))(t)
		measurement := testcheck.Must(modelartifact.MeasureAtLocation(inventory, directory, modelartifact.MeasurementPolicy{
			MaxSamplesPerTensor: 64, MaxReadBytes: 1 << 20,
		}))(t)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       fmt.Sprintf("catalog/model/%d", index),
			Artifacts: []artifact.Descriptor{{ID: modelID}},
		}); err != nil {
			t.Fatal(err)
		}
		inventoryContent := testcheck.Must(inventory.Content())(t)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:      fmt.Sprintf("catalog/inventory/%d", index),
			Contents: []artifact.Content{inventoryContent},
			Lineage: []artifact.Lineage{{
				Child: inventory.ID, Parent: modelID, Relation: artifact.RelationDerivedFrom,
			}},
		}); err != nil {
			t.Fatal(err)
		}
		batch := testcheck.Must(measurement.Batch(fmt.Sprintf("catalog/measurement/%d", index)))(t)
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	testcheck.NoError(t, store.Close())

	handler := testcheck.Must(New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, OvergoDBPath: root}, &fakeGenerator{}))(t)
	t.Cleanup(func() { _ = handler.Close() })
	response := serveTestRequest(handler, http.MethodGet, "/analyze/tensors/similar?name=blk.0.attn_q.weight&k=5", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result analyzeTensorsSimilarResponse
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	if result.Pool != "store-catalog" {
		t.Fatalf("pool = %q, want store-catalog", result.Pool)
	}
	if result.Metric != "rank-footrule" {
		t.Fatalf("metric = %q; the pool source changes, the algorithm does not", result.Metric)
	}
	if result.Target.Model == "" || len(result.Neighbors) != 5 {
		t.Fatalf("target=%+v neighbors=%d", result.Target, len(result.Neighbors))
	}
	models := map[string]bool{}
	for _, neighbor := range result.Neighbors {
		if neighbor.Model == "" {
			t.Fatalf("neighbor %q lacks a model label", neighbor.Name)
		}
		models[neighbor.Model] = true
	}
	if len(models) < 2 {
		t.Fatalf("neighbors span %d model(s), want cross-model retrieval", len(models))
	}
}
