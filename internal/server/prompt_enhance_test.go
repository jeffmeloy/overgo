package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
)

// TestPromptEnhanceRewritesUnderTheInstruction pins the enhance route: the
// served chat model sees the fixed instruction as the system turn and the
// trimmed prompt as the user turn, the rewrite comes back beside the
// original, and the record is stored under its schema with both and the
// instruction; an empty prompt is refused.
func TestPromptEnhanceRewritesUnderTheInstruction(t *testing.T) {
	generator := &fakeGenerator{pieces: []string{"a red square, ", "studio lighting"}}
	handler := newTestHandlerWithRepository(t, generator)
	page := serveTestRequest(handler, http.MethodPost, "/generation/enhance", `{"prompt":" a red square "}`)
	var answer promptEnhanceResponse
	if page.Code != http.StatusOK || json.Unmarshal(page.Body.Bytes(), &answer) != nil {
		t.Fatalf("status=%d body=%s", page.Code, page.Body.String())
	}
	if answer.Original != "a red square" || answer.Enhanced != "a red square, studio lighting" || answer.Instruction != PromptEnhanceInstruction ||
		answer.Model != testModelID || answer.Record == "" {
		t.Fatalf("answer = %+v", answer)
	}
	generator.mu.Lock()
	messages := generator.chatMessages
	generator.mu.Unlock()
	if len(messages) != 2 || messages[0].Role != inference.ChatRoleSystem || messages[0].Content != PromptEnhanceInstruction ||
		messages[1].Role != inference.ChatRoleUser || messages[1].Content != "a red square" {
		t.Fatalf("the served model saw %+v", messages)
	}
	id, err := artifact.ParseID(answer.Record)
	if err != nil {
		t.Fatal(err)
	}
	descriptor, reader, found, err := handler.repository.OpenContent(t.Context(), id)
	if err != nil || !found || descriptor.Schema != promptEnhancementSchema || descriptor.MediaType != artifact.JSONMediaType {
		t.Fatalf("record = %+v found=%v err=%v", descriptor, found, err)
	}
	var stored promptEnhancement
	if err := json.NewDecoder(reader).Decode(&stored); err != nil || stored != answer.promptEnhancement {
		t.Fatalf("stored record = %+v, %v; answered %+v", stored, err, answer.promptEnhancement)
	}
	if refused := serveTestRequest(handler, http.MethodPost, "/generation/enhance", `{"prompt":"  "}`); refused.Code != http.StatusBadRequest {
		t.Fatalf("an empty prompt answered %d: %s", refused.Code, refused.Body.String())
	}
}
