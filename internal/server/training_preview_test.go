package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/trainingworkflow"
)

// TestDatasetPreviewShowsTrainedTokens: the training route pages a dataset
// as the objective encodes it before any run: each preference pair's chosen
// and rejected sequences with the shared prompt prefix masked and the
// responses trained, the record count for paging, and every sequence's length.
func TestDatasetPreviewShowsTrainedTokens(t *testing.T) {
	t.Parallel()
	fixture := newDPOTrainingFixture(t)
	handler, err := New(Config{Repository: fixture.store}, &workspaceTestRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: fixture.workspace})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	request := func(position int, dataset any) (int, trainingworkflow.Preview) {
		t.Helper()
		body, err := json.Marshal(map[string]any{"task": "training", "recipe": fixture.definition.ID,
			"input": map[string]any{"dataset": dataset}, "position": position, "limit": 1})
		if err != nil {
			t.Fatal(err)
		}
		response := serveTestRequest(handler, http.MethodPost, "/training/preview", string(body))
		var preview trainingworkflow.Preview
		if response.Code == http.StatusOK {
			if err := json.Unmarshal(response.Body.Bytes(), &preview); err != nil {
				t.Fatal(err)
			}
		}
		return response.Code, preview
	}
	code, preview := request(0, fixture.dataset)
	if code != http.StatusOK || preview.Records != 2 || len(preview.Page) != 1 || preview.Page[0].ID != "pair" || len(preview.Page[0].Sequences) != 2 {
		t.Fatalf("preview status=%d %+v", code, preview)
	}
	for index, sequence := range preview.Page[0].Sequences {
		masked, trained := 0, 0
		for position, token := range sequence.Tokens {
			switch {
			case token.Trained:
				trained++
			case trained != 0:
				t.Fatalf("%s: masked token %d follows a trained one", sequence.Label, position)
			default:
				masked++
			}
		}
		if masked == 0 || trained == 0 || preview.Lengths[index] != len(sequence.Tokens) {
			t.Fatalf("%s: masked=%d trained=%d length=%d of %d", sequence.Label, masked, trained, preview.Lengths[index], len(sequence.Tokens))
		}
	}
	// The lengths cover every sequence of the dataset, not just the page.
	if len(preview.Lengths) != 4 || preview.Page[0].Sequences[0].Label != "chosen" || preview.Page[0].Sequences[1].Label != "rejected" {
		t.Fatalf("lengths = %v, sequences = %+v", preview.Lengths, preview.Page[0].Sequences)
	}
	// Past the last record the page is empty; the count still pages.
	if code, past := request(2, fixture.dataset); code != http.StatusOK || past.Records != 2 || len(past.Page) != 0 {
		t.Fatalf("past the end status=%d %+v", code, past)
	}
	if code, _ := request(0, "dataset:absent"); code != http.StatusBadRequest {
		t.Fatalf("an unknown dataset status=%d", code)
	}
}
