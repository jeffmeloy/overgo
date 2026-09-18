package speechsynth

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestSpeechRecovery(t *testing.T) {
	synth := loadArtifactSynthesizer(t)
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if store != nil {
			if err := store.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	model := testutil.ArtifactID(t, artifact.KindModel, "speech-recovery-identity-fixture")
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[artifact.ID]bool{}
	execute := func(request SynthesisRequest, stop bool) (Audio, artifact.ID) {
		t.Helper()
		content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.speech-input.v1"), request)
		if err != nil {
			t.Fatal(err)
		}
		key := capabilityruntime.RunKey(definition, content)
		operation, err := workflowruntime.ExecutionID(definition.ID, key)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		if stop {
			cancel(context.Canceled)
		}
		input := definition.Inputs[0]
		output, _, err := capabilityruntime.ExecuteMeasured[Audio](ctx, store, model, program, key, map[recipe.PortName]workflowruntime.Value{input.Name: workflowruntime.ArtifactValue(input.Data, request, content)}, func(runtime *workflowruntime.Runtime) error { return RegisterRuntime(runtime, model, synth) })
		if stop {
			if !errors.Is(err, context.Canceled) || len(output.PCM) != 0 {
				t.Fatalf("cancelled request returned output: %v", err)
			}
			return output, operation
		}
		if err != nil {
			t.Fatal(err)
		}
		receipt, found, err := runrecord.ResolveStageReceipt(t.Context(), store, operation, "generate")
		if err != nil || !found || receipt.Attempt != 1 {
			t.Fatalf("native generation replayed: found=%v receipt=%+v error=%v", found, receipt, err)
		}
		return output, operation
	}
	baseline := SynthesisRequest{Text: "Green always green.", Voice: "alba", MaxFrames: 2, Seed: 7}
	variants := []SynthesisRequest{baseline, baseline, baseline, baseline}
	variants[1].Text = "This is a different sentence."
	variants[2].Seed++
	voices, err := ListVoices(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, voice := range voices {
		if voice != "alba" {
			variants[3].Voice = voice
			break
		}
	}
	if variants[3].Voice == "alba" {
		t.Fatal("second exported voice required")
	}
	var config struct {
		Mimi struct {
			FrameRate float64 `json:"frame_rate"`
		} `json:"mimi"`
	}
	if err := jsonfile.Decode(filepath.Join(artifactDir(t), "pockettts_config.json"), &config); err != nil {
		t.Fatal(err)
	}
	completeRequest := SynthesisRequest{Text: "Hello. This speech was generated locally.", Voice: "alba", Seed: 7}
	tokens, err := synth.tokenizer.Encode(completeRequest.Text)
	if err != nil {
		t.Fatal(err)
	}
	// Same pinned reference budget as the native completion fixture; this is
	// test input, not a production document-duration policy.
	completeRequest.MaxFrames = int(math.Ceil((float64(len(tokens))/3 + 2) * config.Mimi.FrameRate))
	variants = append(variants, completeRequest)
	for index, request := range variants {
		first, operation := execute(request, false)
		if seen[operation] {
			t.Fatal("changed text, seed or voice reused the same operation")
		}
		seen[operation] = true
		if first.Complete != (index == len(variants)-1) {
			t.Fatalf("unexpected completion for variant %d: %v", index, first.Complete)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		store = nil
		store, err = overgodb.Open(directory)
		if err != nil {
			t.Fatal(err)
		}
		replay, replayed := execute(request, false)
		if replayed != operation || !reflect.DeepEqual(first, replay) {
			t.Fatal("exact replay changed native output")
		}
	}
	stopped := baseline
	stopped.Text = "This request is stopped before acceptance."
	_, cancelled := execute(stopped, true)
	_, retried := execute(stopped, false)
	if cancelled != retried {
		t.Fatal("retry changed request identity")
	}
	t.Run("cancel after durable generation", testSpeechRecoveryAfterGenerationCancellation)
	t.Log("actual native binding: exact output survives store reopen; text, seed and exported voice changes retain distinct operation identities; cancelled request returns no audio and retries successfully")
}

// Cancel at the durable store boundary while keeping the actual native binding.
type speechCancelAfterLatents struct {
	artifact.Repository
	cancel context.CancelCauseFunc
}

func (store *speechCancelAfterLatents) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	id, err := store.Repository.Commit(ctx, batch)
	if err == nil && store.cancel != nil {
		for _, content := range batch.Contents {
			if content.Descriptor.Schema == "overgo.speech-latents.v1" {
				store.cancel(context.Canceled)
				store.cancel = nil
				break
			}
		}
	}
	return id, err
}

func testSpeechRecoveryAfterGenerationCancellation(t *testing.T) {
	synth := loadArtifactSynthesizer(t)
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if store != nil {
			if err := store.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	model := testutil.ArtifactID(t, artifact.KindModel, "speech-cancel-after-generation-fixture")
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	request := SynthesisRequest{Text: "Green always green.", Voice: "alba", MaxFrames: 2, Seed: 7}
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.speech-input.v1"), request)
	if err != nil {
		t.Fatal(err)
	}
	key := capabilityruntime.RunKey(definition, content)
	operation, err := workflowruntime.ExecutionID(definition.ID, key)
	if err != nil {
		t.Fatal(err)
	}
	execute := func(ctx context.Context, repository artifact.Repository) (Audio, error) {
		input := definition.Inputs[0]
		output, _, err := capabilityruntime.ExecuteMeasured[Audio](ctx, repository, model, program, key, map[recipe.PortName]workflowruntime.Value{input.Name: workflowruntime.ArtifactValue(input.Data, request, content)}, func(runtime *workflowruntime.Runtime) error { return RegisterRuntime(runtime, model, synth) })
		return output, err
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)
	controlled := &speechCancelAfterLatents{Repository: store, cancel: cancel}
	output, err := execute(ctx, controlled)
	if controlled.cancel != nil {
		t.Fatal("native generation was not persisted")
	}
	if !errors.Is(err, context.Canceled) || len(output.PCM) != 0 {
		t.Fatalf("cancelled native decoding returned audio: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = nil
	store, err = overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	output, err = execute(t.Context(), store)
	if err != nil || len(output.PCM) == 0 {
		t.Fatalf("retry failed: %v", err)
	}
	receipt, found, err := runrecord.ResolveStageReceipt(t.Context(), store, operation, "generate")
	if err != nil || !found || receipt.Attempt != 1 {
		t.Fatalf("retry regenerated completed latents: found=%v receipt=%+v error=%v", found, receipt, err)
	}
	t.Log("actual RegisterRuntime: cancellation after durable generation exposes no PCM; store reopen and retry decode the retained latents without another generation attempt")
}
