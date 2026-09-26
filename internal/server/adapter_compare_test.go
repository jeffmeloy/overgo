package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// adapterSessionRunner serves the hermetic session model with one output
// projection adapter loaded at scale 1; the adapter moves the logit of one
// token, so its greedy output differs from the base model's.
func adapterSessionRunner(t *testing.T) *inference.Runner {
	t.Helper()
	tensor := func(name string, shape []uint64, values []float32) gguf.TensorData {
		var data bytes.Buffer
		if err := binary.Write(&data, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
		return gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(data.Bytes())}
	}
	moved := make([]float32, 8)
	moved[7] = -100
	path := filepath.Join(t.TempDir(), "move-d.gguf")
	testutil.WriteGGUF(t, path, []gguf.Metadata{
		testutil.GGUFScalar("general.type", gguf.ValueTypeString, "adapter"),
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "llama"),
		testutil.GGUFScalar("adapter.type", gguf.ValueTypeString, "lora"),
		testutil.GGUFScalar("adapter.lora.alpha", gguf.ValueTypeFloat32, float32(1)),
	}, []gguf.TensorData{
		tensor("output.weight.lora_a", []uint64{8, 1}, []float32{1, 1, 1, 1, 1, 1, 1, 1}),
		tensor("output.weight.lora_b", []uint64{1, 8}, moved),
	})
	return openSessionRunner(t, 64, inference.LoRAConfig{Path: path, Scale: 1})
}

// responseText is a non-streamed response's message text.
func responseText(t *testing.T, body []byte) (string, string) {
	t.Helper()
	var decoded struct {
		ID     string `json:"id"`
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("response %s: %v", body, err)
	}
	var text strings.Builder
	for _, item := range decoded.Output {
		for _, part := range item.Content {
			text.WriteString(part.Text)
		}
	}
	return decoded.ID, text.String()
}

// TestAdapterComparisonTurn: one prompt and sampling run against the base
// model (an empty lora list) and the loaded adapter gives two different
// outputs, the adapter's matching the stored turn the loaded scales made;
// neither comparison run is stored, and an adapter not loaded is refused.
func TestAdapterComparisonTurn(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, adapterSessionRunner(t))
	listed := serveTestRequest(handler, http.MethodGet, "/lora-adapters", "")
	if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), `"path":`) || !strings.Contains(listed.Body.String(), `move-d.gguf`) {
		t.Fatalf("adapters status=%d body=%s", listed.Code, listed.Body.String())
	}
	run := func(extra string) (int, string, string) {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/v1/responses",
			`{"model":"`+testModelID+`","input":"ab","temperature":0,"seed":7,"ignore_eos":true,"max_output_tokens":6`+extra+`}`)
		id, text := responseText(t, response.Body.Bytes())
		return response.Code, id, text
	}
	code, turnID, turn := run(`,"store":true`)
	if code != http.StatusOK || turn == "" {
		t.Fatalf("stored turn status=%d text=%q", code, turn)
	}
	code, baseID, base := run(`,"store":false,"lora":[]`)
	if code != http.StatusOK {
		t.Fatalf("base status=%d", code)
	}
	code, adapterID, adapted := run(`,"store":false,"lora":[{"id":0,"scale":1}]`)
	if code != http.StatusOK {
		t.Fatalf("adapter status=%d", code)
	}
	if adapted != turn || base == adapted {
		t.Fatalf("turn=%q base=%q adapter=%q; want the adapter to repeat the turn and the base to differ", turn, base, adapted)
	}
	if _, found, err := runrecord.ResolveInteraction(t.Context(), store, turnID); err != nil || !found {
		t.Fatalf("stored turn found=%v err=%v", found, err)
	}
	for _, id := range []string{baseID, adapterID} {
		if _, found, err := runrecord.ResolveInteraction(t.Context(), store, id); err != nil || found {
			t.Fatalf("comparison run %s stored=%v err=%v", id, found, err)
		}
	}
	if code, _, _ := run(`,"store":false,"lora":[{"id":3,"scale":1}]`); code != http.StatusBadRequest {
		t.Fatalf("an adapter not loaded answered %d", code)
	}
}
