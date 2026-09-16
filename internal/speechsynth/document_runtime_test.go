package speechsynth

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestSpeechDocumentOrderedRecording(t *testing.T) {
	synth, err := LoadSynthesizer(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := synth.PlanDocument(t.Context(), DocumentRequest{Text: strings.Repeat("Hello. This speech was generated locally. Each sentence belongs to this document. ", 4), Voice: "alba", Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Segments) < 2 {
		t.Fatal("fixture must exercise multiple segments")
	}
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
	modelID := testutil.ArtifactID(t, artifact.KindModel, "speech-document-recording")
	testutil.PublishArtifact(t, store, modelID)
	parent, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, modelID)
	if err != nil {
		t.Fatal(err)
	}
	base, err := modelrecipe.CompileCapability(parent)
	if err != nil {
		t.Fatal(err)
	}
	program, source, values, err := DocumentProgram(base, plan)
	if err != nil {
		t.Fatal(err)
	}
	key := capabilityruntime.RunKey(program.Definition(), source)
	operation, err := workflowruntime.ExecutionID(program.Definition().ID, key)
	if err != nil {
		t.Fatal(err)
	}
	stopContext, stop := context.WithCancelCause(t.Context())
	defer stop(nil)
	completed := map[recipe.NodeID]uint32{}
	_, stopped := capabilityruntime.Execute[DocumentWAV](stopContext, store, modelID, program, key, values, func(runtime *workflowruntime.Runtime) error {
		runtime.ObserveStages(func(receipt runrecord.StageReceipt) {
			if strings.HasSuffix(string(receipt.Node), "-generate") {
				completed[receipt.Node] = receipt.Attempt
				stop(context.Canceled)
			}
		})
		return RegisterDocumentRuntime(runtime, store, modelID, synth)
	})
	if !errors.Is(stopped, context.Canceled) || len(completed) == 0 {
		t.Fatalf("stop did not retain completed segment: %v %v", completed, stopped)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = nil
	store, err = overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	execute := func() DocumentWAV {
		t.Helper()
		wave, err := capabilityruntime.Execute[DocumentWAV](t.Context(), store, modelID, program, key, values, func(runtime *workflowruntime.Runtime) error {
			return RegisterDocumentRuntime(runtime, store, modelID, synth)
		})
		if err != nil {
			t.Fatal(err)
		}
		return wave
	}
	first := execute()
	decoded, status, err := media.DecodeAudio(t.Context(), first.Data, uint64(artifact.MaxContentBytes))
	if err != nil || len(decoded.Samples) == 0 {
		t.Fatalf("invalid completed WAV: %s %v", status, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = nil
	store, err = overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	replayed := execute()
	if !bytes.Equal(first.Data, replayed.Data) {
		t.Fatal("reopened recording changed")
	}
	for node, attempt := range completed {
		receipt, found, err := runrecord.ResolveStageReceipt(t.Context(), store, operation, node)
		if err != nil || !found || receipt.Attempt != attempt {
			t.Fatalf("completed segment %s regenerated: %+v %v", node, receipt, err)
		}
	}
	t.Logf("complete native document: %d segments, %d source bytes, %d WAV bytes; stopped after durable progress, reopened, resumed without regenerating completed speech, then replayed exact WAV", len(plan.Segments), len(plan.Request.Text), len(first.Data))
}
