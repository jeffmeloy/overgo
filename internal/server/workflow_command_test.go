package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestWorkflowSubmissionCommand: a submission the run would admit answers
// the commands that reproduce it without running it -- a generation recipe
// as its cmd/recipe invocation on the model's file, and every submission as
// the request that posts it again, each word quoted for a POSIX shell -- and
// a submission the run would refuse is refused the same way.
func TestWorkflowSubmissionCommand(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "workflow-command-recipe")
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{},
		run:           testutil.ArtifactID(t, artifact.KindRun, "workflow-command-run"),
		repository:    store,
		capability: WorkflowCapability{
			Task: recipe.TaskSpeech, Recipe: recipeID, Location: "C:/models/it's.gguf",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []WorkflowControl{
				{Name: "text", Type: WorkflowControlText, Required: true},
				{Name: "seed", Type: WorkflowControlInteger},
			},
		},
	}
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	submission := `{"task":"speech","recipe":"` + recipeID.String() + `","input":{"text":"it's here","seed":17}}`
	answered := serveTestRequest(handler, http.MethodPost, "/generation/command", submission)
	var body struct {
		Commands []workflowCommand `json:"commands"`
	}
	if err := json.Unmarshal(answered.Body.Bytes(), &body); answered.Code != http.StatusOK || err != nil || len(body.Commands) != 2 {
		t.Fatalf("command status=%d body=%s", answered.Code, answered.Body.String())
	}
	if cli := body.Commands[0]; cli.Kind != "cli" ||
		cli.Text != `go run ./cmd/recipe run -task 'speech' -input '{"text":"it'\''s here","seed":17}' 'C:/models/it'\''s.gguf'` {
		t.Fatalf("cli command = %+v", cli)
	}
	if again := body.Commands[1]; again.Kind != "http" ||
		again.Text != `curl -X POST 'http://example.com/generation/run' -H 'Content-Type: application/json' -d '{"task":"speech","recipe":"`+recipeID.String()+`","input":{"text":"it'\''s here","seed":17}}'` {
		t.Fatalf("http command = %+v", again)
	}
	if generator.executions != 0 {
		t.Fatalf("answering the command ran the workflow %d times", generator.executions)
	}
	refused := serveTestRequest(handler, http.MethodPost, "/generation/command", `{"task":"speech","recipe":"`+recipeID.String()+`","input":{"seed":17}}`)
	if refused.Code != http.StatusBadRequest {
		t.Fatalf("a submission missing its required text answered %d %s", refused.Code, refused.Body.String())
	}
}
