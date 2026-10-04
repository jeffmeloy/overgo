package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/peft"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/trainingworkflow"
)

// sftLetters renders a conversation within the tiny model's four-letter
// vocabulary: a role's letter, then its content, for each message, and the
// assistant's letter alone as a generation prompt.
func sftLetters(messages []dataset.ChatMessage, generationPrompt bool) (string, error) {
	var text strings.Builder
	for _, message := range messages {
		text.WriteString(map[string]string{"user": "b", "assistant": "c"}[message.Role] + message.Content)
	}
	if generationPrompt {
		text.WriteString("c")
	}
	return text.String(), nil
}

// stepCounter is a reporter that counts the steps a run reports.
type stepCounter struct {
	testReporter
	steps *atomic.Int64
}

func (counter stepCounter) Progress(done uint64, _ *uint64) { counter.steps.Store(int64(done)) }

// sftFixture is a served tiny dense model, a store holding a registered
// dataset of twelve conversations, and the workspace that fine-tunes the model
// on them.
type sftFixture struct {
	store         *overgodb.Store
	roots         dataroot.Roots
	workspace     *SFTWorkspace
	model         artifact.ID
	conversations artifact.ID
}

func newSFTFixture(t *testing.T) sftFixture {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	roots := dataroot.Roots{Models: filepath.Join(root, "models"), Datasets: filepath.Join(root, "datasets"), Checkpoints: filepath.Join(root, "checkpoints")}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	served := filepath.Join(roots.Models, "served")
	model := writeWorkspaceModel(t, served, weights, shapes)
	// A served model declares its context length, which bounds each training window,
	// and the token that ends a sequence, which separates the windows' records.
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"max_position_embeddings":16,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	for name, document := range map[string]string{"config.json": config, "generation_config.json": `{"eos_token_id":0}`} {
		if err := os.WriteFile(filepath.Join(served, name), []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	directory := filepath.Join(root, "conversations")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	for index := range 12 {
		user, assistant := strings.Repeat("ab", 4), strings.Repeat("cd", 4)
		if index >= 6 {
			user, assistant = strings.Repeat("ba", 4), strings.Repeat("dc", 4)
		}
		data, err := json.Marshal(dataset.Conversation{Messages: []dataset.ChatMessage{
			{Role: "user", Content: user[:1+index%6]}, {Role: "assistant", Content: assistant[:1+index%6]},
		}})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, fmt.Sprintf("%02d.json", index+1)), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	registered, err := dataset.RegisterDirectoryDatasetAs(ctx, store, "chats", directory, dataset.ModalityStructured)
	if err != nil {
		t.Fatal(err)
	}
	workspace, err := NewSFTWorkspace(store, roots, SFTTrainer{Model: model, File: served, Render: sftLetters})
	if err != nil {
		t.Fatal(err)
	}
	return sftFixture{store: store, roots: roots, workspace: workspace, model: model, conversations: registered.Dataset}
}

// input is the run's input naming the fixture's conversations.
func (fixture sftFixture) input(output string) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"dataset":%q,"steps":4,"output":%q}`, fixture.conversations.String(), output))
}

// adapterSteps is how many updates a LoRA run takes: a rank-2 adapter moves the
// tiny model's answers to the held-out replies in sixteen steps, where in four
// it answers as the base does and there is no verdict to give.
const adapterSteps = 16

// inputWithRank is the input of a LoRA run of the given rank.
func (fixture sftFixture) inputWithRank(output string, rank int) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"dataset":%q,"steps":%d,"rank":%d,"output":%q}`, fixture.conversations.String(), adapterSteps, rank, output))
}

// inputOnPrompts is the input of a run that also trains on the prompts.
func (fixture sftFixture) inputOnPrompts(output string) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"dataset":%q,"steps":4,"train_on_prompts":true,"output":%q}`, fixture.conversations.String(), output))
}

// TestSFTWorkspaceFineTunesTheServedModel: the Train tab offers the served
// model on conversations, whole or as a LoRA adapter of a rank the page asks
// for, and says how a run learns;
// previewing pages the conversations as rendered; running it renders a
// registered conversation dataset, trains the model on the host and records
// one run with the model and conversations in and the trained model and the
// verdict out; a name that is not one path component is refused as a failed
// run, never written outside the checkpoints root.
func TestSFTWorkspaceFineTunesTheServedModel(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newSFTFixture(t)
	workspace := fixture.workspace
	if other, err := workspace.WorkflowCapabilities(ctx, workflowExport); err != nil || len(other) != 0 {
		t.Fatalf("an export listing held %v (%v), want none", other, err)
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowTraining)
	if err != nil || len(capabilities) != 2 {
		t.Fatalf("training capabilities = %+v, %v, want fine-tuning the whole model and a LoRA adapter", capabilities, err)
	}
	capability, adapterCapability := capabilities[0], capabilities[1]
	var adapterControls []string
	for _, control := range adapterCapability.Controls {
		adapterControls = append(adapterControls, control.Name)
	}
	if adapterCapability.Name != "served, LoRA adapter on conversations" || strings.Join(adapterControls, ",") != "dataset,steps,rank,train_on_prompts,output" || adapterCapability.Recipe == capability.Recipe ||
		adapterCapability.Controls[2].Bounds == nil || adapterCapability.Controls[2].Bounds.Default != sftLoRADefaultRank || !strings.Contains(adapterCapability.Note, "PEFT adapter") {
		t.Fatalf("LoRA capability = %+v with controls %v, want its own recipe, a rank of default %d and a note naming the PEFT adapter", adapterCapability, adapterControls, sftLoRADefaultRank)
	}
	var controls []string
	for _, control := range capability.Controls {
		controls = append(controls, control.Name)
	}
	if capability.Task != recipe.TaskTraining || !capability.Host || capability.Name != "served, on conversations" || strings.Join(controls, ",") != "dataset,steps,train_on_prompts,output" {
		t.Fatalf("capability = %+v with controls %v", capability, controls)
	}
	if prompts := capability.Controls[2]; prompts.Type != workflowControlBoolean || prompts.Required {
		t.Fatalf("train on prompts is %+v, want an optional yes or no, left off to learn the replies", prompts)
	}
	if capability.Note != sftNote || capability.Controls[0].Format != sftDatasetFormat || capability.Controls[0].Label == "" {
		t.Fatalf("the dataset control declares %+v and the note %q, want the conversation format, a label and how a run learns", capability.Controls[0], capability.Note)
	}

	// The route lists a capability only when it validates; a format belongs to a dataset control alone.
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		t.Fatalf("the Train tab would refuse the listing: %v", err)
	}
	changed := capability
	changed.Controls = slices.Clone(capability.Controls)
	changed.Controls[1].Format = sftDatasetFormat
	if err := validateWorkflowCapabilities([]WorkflowCapability{changed}); err == nil {
		t.Fatal("a dataset format on a number of steps was accepted")
	}

	page, err := workspace.PreviewWorkflow(ctx, WorkflowTraining, capability.Task, capability.Recipe, fixture.input("tuned"), 1, 2)
	if preview, ok := page.(trainingworkflow.Preview); err != nil || !ok || preview.Records != 12 || len(preview.Page) != 2 || preview.Page[0].ID != "02.json" {
		t.Fatalf("preview = (%+v, %v), want the second and third of twelve conversations", page, err)
	}
	if _, err := workspace.PreviewWorkflow(ctx, WorkflowTraining, capability.Task, testutil.ArtifactID(t, artifact.KindRecipe, "another"), fixture.input("tuned"), 0, 1); err == nil {
		t.Fatal("a preview for another recipe was admitted")
	}
	// The preview marks the tokens a run learns: the replies alone, or every token when asked to train on the prompts.
	learned := func(input json.RawMessage) (trained, masked int) {
		t.Helper()
		page, err := workspace.PreviewWorkflow(ctx, WorkflowTraining, capability.Task, capability.Recipe, input, 0, 1)
		if err != nil {
			t.Fatal(err)
		}
		for _, token := range page.(trainingworkflow.Preview).Page[0].Sequences[0].Tokens {
			if token.Trained {
				trained++
			} else {
				masked++
			}
		}
		return trained, masked
	}
	if trained, masked := learned(fixture.input("tuned")); trained == 0 || masked == 0 {
		t.Fatalf("the preview shows %d trained and %d masked tokens, want the reply learned and the prompt read", trained, masked)
	}
	if trained, masked := learned(fixture.inputOnPrompts("tuned")); trained == 0 || masked != 0 {
		t.Fatalf("training on prompts previews %d trained and %d masked tokens, want every token trained", trained, masked)
	}

	steps := &atomic.Int64{}
	completion, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, capability, fixture.input("tuned"), stepCounter{steps: steps})
	if err != nil || len(completion.Outputs) != 2 || steps.Load() != 4 {
		t.Fatalf("run = (%+v, %v) after %d steps, want the trained model and its verdict after four", completion, err, steps.Load())
	}
	run, err := runrecord.RequireRun(ctx, fixture.store, completion.Run)
	// A run record keeps its artifacts in identity order, not the order they were named in.
	if err != nil || run.Outcome != runrecord.OutcomeSucceeded || len(run.Inputs) != 2 || !slices.Contains(run.Inputs, fixture.model) || !slices.Contains(run.Inputs, fixture.conversations) ||
		len(run.Outputs) != 2 || !slices.Contains(run.Outputs, completion.Outputs[0]) || !slices.Contains(run.Outputs, completion.Outputs[1]) {
		t.Fatalf("recorded run = %+v (%v), want the model and conversations in and %v out", run, err, completion.Outputs)
	}
	if _, err := os.Stat(filepath.Join(fixture.roots.Checkpoints, "tuned")); err != nil {
		t.Fatalf("the trained model was not written under the checkpoints root: %v", err)
	}

	// A run may train on the prompts as well; it records the same way.
	every, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, capability, fixture.inputOnPrompts("tuned-every-token"), stepCounter{steps: &atomic.Int64{}})
	if err != nil || len(every.Outputs) != 2 || every.Outputs[1] == completion.Outputs[1] {
		t.Fatalf("a run on the prompts = (%+v, %v), want a verdict of its own", every, err)
	}

	// The LoRA capability previews the same conversations and trains an adapter of the rank given, left
	// beside the trained model as a PEFT directory the whole-model run does not write.
	if page, err := workspace.PreviewWorkflow(ctx, WorkflowTraining, adapterCapability.Task, adapterCapability.Recipe, fixture.input("adapted"), 0, 1); err != nil || page.(trainingworkflow.Preview).Records != 12 {
		t.Fatalf("LoRA preview = (%+v, %v), want the same twelve conversations", page, err)
	}
	if _, err := os.Stat(filepath.Join(fixture.roots.Checkpoints, "tuned", trainingworkflow.LoRAAdapterDirectory)); err == nil {
		t.Fatal("fine-tuning the whole model left an adapter directory")
	}
	adapted, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, adapterCapability, fixture.inputWithRank("adapted", 2), stepCounter{steps: &atomic.Int64{}})
	if err != nil || len(adapted.Outputs) != 2 {
		t.Fatalf("LoRA run = (%+v, %v), want the trained model and its verdict", adapted, err)
	}
	adapter, err := peft.Load(filepath.Join(fixture.roots.Checkpoints, "adapted", trainingworkflow.LoRAAdapterDirectory))
	if err != nil || len(adapter.Modules) != 7 || adapter.Config.Rank != 2 || adapter.Config.Alpha != 4 {
		t.Fatalf("adapter = (%+v with %d modules, %v), want the seven projections at rank 2 and alpha 4", adapter.Config, len(adapter.Modules), err)
	}
	for name, scenario := range map[string]struct {
		capability WorkflowCapability
		input      json.RawMessage
		want       string
	}{
		"adapter without a rank":  {adapterCapability, fixture.input("no-rank"), "positive rank"},
		"rank on the whole model": {capability, fixture.inputWithRank("rank-on-model", 2), "belongs to the LoRA adapter"},
	} {
		if _, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, scenario.capability, scenario.input, testReporter{}); err == nil || !strings.Contains(err.Error(), scenario.want) {
			t.Errorf("%s: err = %v, want it to say %q", name, err, scenario.want)
		}
	}

	refused, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, capability, fixture.input("../escape"), testReporter{})
	failed, requireErr := runrecord.RequireRun(ctx, fixture.store, refused.Run)
	if err == nil || !strings.Contains(err.Error(), "one safe path component") || requireErr != nil || failed.Outcome != runrecord.OutcomeFailed {
		t.Fatalf("an escaping output = (%v, run %+v, %v), want a refusal recorded as a failed run", err, failed, requireErr)
	}
	if _, err := os.Stat(filepath.Join(fixture.roots.Checkpoints, "..", "escape")); err == nil {
		t.Fatal("an escaping output wrote outside the checkpoints root")
	}
}
