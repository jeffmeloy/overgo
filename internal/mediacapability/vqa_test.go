package mediacapability

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/vqaserve"
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
	// A stored image reaches the real prepare stage, whose missing model
	// configuration fails on the host. Its published run must survive the
	// capability's error handoff, with the original image and question.
	testutil.PublishArtifact(t, store, modelID)
	image, err := vqaserve.ImageContract.ContentBytes([]byte("fixture image"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "vqa/terminal-image", Contents: []artifact.Content{image}}); err != nil {
		t.Fatal(err)
	}
	value, err := capability.Execute(t.Context(), store, t.TempDir(), execution, `{"image":"`+image.Descriptor.ID.String()+`","question":"what?"}`)
	measured, ok := value.(capabilityruntime.Measured)
	if err == nil || !ok {
		t.Fatalf("prepare failure lost its measurement: %T, %v", value, err)
	}
	run, err := runrecord.RequireExactRun(t.Context(), store, measured.Run)
	question, questionErr := artifact.JSONContent(vqaserve.QuestionContract, "what?")
	if err != nil || questionErr != nil || run.Recipe != definition.ID || run.Outcome != runrecord.OutcomeFailed || len(run.Inputs) != 2 || len(run.Outputs) != 0 {
		t.Fatalf("prepare failure run %+v (%v, %v)", run, err, questionErr)
	}
	if !slices.Contains(run.Inputs, image.Descriptor.ID) || !slices.Contains(run.Inputs, question.Descriptor.ID) {
		t.Fatalf("prepare failure run lost its image or question: %+v", run)
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
