package mediacapability

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestVQAControlsDeclareImageQuestionAndBudget pins the page form of the
// VQA request: the stored image and the question are required controls,
// the decode budget is an optional integer.
func TestVQAControlsDeclareImageQuestionAndBudget(t *testing.T) {
	controls, refusal := Controls(modelrecipe.ModuleVQAPrepare, "")
	if refusal != "" {
		t.Fatal(refusal)
	}
	names := make([]string, 0, len(controls))
	for _, control := range controls {
		names = append(names, control.Name+":"+control.Type)
		if required := control.Name != "max_tokens"; control.Required != required {
			t.Errorf("control %s required=%v", control.Name, control.Required)
		}
	}
	if joined := strings.Join(names, ","); joined != "image:artifact,question:text,max_tokens:integer" {
		t.Fatalf("VQA controls = %s", joined)
	}
}

// TestVQAExecutorRefusesBeforeTheDevice pins the refusals the executor
// reaches without a device: a request without an image and a request
// naming an image the store does not hold are refused by name, and the
// catalog binds the executor to the vqa prepare module.
func TestVQAExecutorRefusesBeforeTheDevice(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "vqa-executor")
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskVQA, modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	execution := modelrecipe.CapabilityEvidenceSelection{Program: program}
	capability := Catalog[recipe.TaskVQA]
	if _, err := capability.Execute(t.Context(), store, t.TempDir(), execution, `{"question":"what?"}`); err == nil || !strings.Contains(err.Error(), "image") {
		t.Fatalf("a request without an image ran: %v", err)
	}
	absent := testutil.ArtifactID(t, artifact.KindFile, "absent-image")
	if _, err := capability.Execute(t.Context(), store, t.TempDir(), execution, `{"image":"`+absent.String()+`","question":"what?"}`); err == nil || !strings.Contains(err.Error(), "absent") {
		t.Fatalf("a request naming an absent image ran: %v", err)
	}
}

// TestTextOutputPublishesPlainText pins the text answer's published form:
// a string output becomes a plain-text output artifact.
func TestTextOutputPublishesPlainText(t *testing.T) {
	content, err := OutputContent("there")
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.ID.Kind() != artifact.KindOutput || content.Descriptor.MediaType != modelrecipe.TextAnswerMediaType || string(content.Data) != "there" {
		t.Fatalf("text output = %+v %q", content.Descriptor, content.Data)
	}
}
