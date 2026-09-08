package audioparity

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

func TestASRProductionTrainingAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": production CPU training runs as its exact mandatory acceptance")
	}
	l := newAdapterLifecycle(t)
	reference := l.update(t, l.batcher(t, nil))
	referenceWeights := l.adapter.WeightSnapshot()
	referenceOptimizer, err := l.adapter.OptimizerSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("independent library update complete: loss=%g parameters=%d", reference.Loss, len(referenceWeights))
	// Register metadata only. Audio remains in its exact original Parquet row.
	metadata, err := json.Marshal(dataset.SpeechRecord{Audio: mustAudioID(t, l.fixture.payload), Origin: l.origin, Target: l.fixture.record.Text})
	if err != nil {
		t.Fatal(err)
	}
	metadata = append(metadata, '\n') // One source-addressed JSONL metadata record.
	metadataID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, metadata)
	if err != nil {
		t.Fatal(err)
	}
	version, err := dataset.NewVersion([]dataset.Asset{{Name: "speech", Artifact: metadataID, Records: 1}})
	if err != nil {
		t.Fatal(err)
	}
	sourceID := fmt.Sprintf("%s/speech/0", version.ID)
	split, err := dataset.BuildGroupSplit(version.ID, []dataset.Record{{ID: sourceID, Group: l.fixture.record.RowID}}, 0,
		[]dataset.SplitPartition{{Name: "selected", Weight: 1}, {Name: "reserved", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	var membership dataset.Membership
	for _, value := range split.Memberships {
		if len(value.Records) != 0 {
			membership = value
		}
	}
	if len(membership.Records) != 1 {
		t.Fatal("real source selection is empty")
	}
	batch, err := split.PublicationBatch("production/source", nil)
	if err != nil {
		t.Fatal(err)
	}
	content, err := version.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, content, artifact.Content{Descriptor: artifact.Descriptor{ID: metadataID, Size: uint64(len(metadata))}, Data: metadata})
	batch.Lineage = append(batch.Lineage, version.Lineage()...)
	l.commit(t, batch, nil)
	inspection, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/audio-text-processor/v1"), l.policy)
	l.content(t, inspection, err)
	profile, found := l.base.PrimaryDependency(recipe.DependencyProcessorProfile)
	if !found {
		t.Fatal("base execution profile is absent")
	}
	spec := l.trainingSpec(t, version.ID, membership.ID, []artifact.ID{profile, l.transform.ID, inspection.Descriptor.ID},
		recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityAudio}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}})
	definition, err := recipe.RequireDefinition(t.Context(), l.store, spec.Recipe)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(t.Context(), l.store, "production/training-verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(t.Context(), l.store, definition, verification, recipe.EvidenceParity, "isolated production-command fixture; no canonical store promotion"); err != nil {
		t.Fatal(err)
	}
	lowercase := true
	request := trainingworkflow.AudioTrainingSpec{BaseRecipe: l.base.ID, MemoryBytes: adapterAcceptanceMemory, Inspection: l.policy, Lowercase: &lowercase}
	manifest := filepath.Join(t.TempDir(), "audio.json")
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifest, encoded, 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(t.TempDir(), "train.exe")
	baselineCommand(t, l.fixture.root, "go", "build", "-o", binary, "./cmd/train")
	if err := l.store.Close(); err != nil {
		t.Fatal(err)
	}
	run := func(name, steps, resume string) (trainingprogram.Checkpoint, string, string) {
		t.Helper()
		directory := filepath.Join(t.TempDir(), name)
		arguments := []string{"-store", l.storePath, "-recipe", definition.ID.String(), "-audio-manifest", manifest, "-out", directory, "-steps", steps}
		if resume != "" {
			arguments = append(arguments, "-resume", resume)
		}
		output := baselineCommand(t, l.fixture.root, binary, arguments...)
		if !strings.Contains(output, "backend=host-reference objective=ctc") || !strings.Contains(output, "source="+sourceID) ||
			!strings.Contains(output, fmt.Sprintf("raw_text_sha256=%x", sha256.Sum256([]byte(l.fixture.record.Text)))) {
			t.Fatalf("command did not execute the real source: %s", output)
		}
		checkpoint, err := trainingprogram.LoadCheckpoint(directory)
		if err != nil || checkpoint.Dataset != version.ID || checkpoint.Split != membership.ID {
			t.Fatalf("production checkpoint lineage: %v", err)
		}
		t.Logf("command %s complete: requested updates=%s cursor=%d checkpoint=%s", name, steps, checkpoint.Stream.Position, checkpoint.ID())
		return checkpoint, directory, output
	}
	first, firstDirectory, firstOutput := run("first", "1", "")
	weights, err := adaptertrain.LoadInputProjection(firstDirectory, l.binding, first.Weights, l.adapter.Program().Parameters()[0].Rows, adapterAcceptanceMemory)
	if err != nil || !slices.Equal(weights, referenceWeights) || !reflect.DeepEqual(first.Optimizer, referenceOptimizer) ||
		!strings.Contains(firstOutput, fmt.Sprintf("loss=%g", reference.Loss)) {
		t.Fatalf("production first update differs from the independently executed library path: %v", err)
	}
	uninterrupted, _, uninterruptedOutput := run("uninterrupted", "2", "")
	resumed, _, resumedOutput := run("resumed", "1", firstDirectory)
	if uninterrupted.Stream != resumed.Stream || resumed.Stream.Position != 2 || uninterrupted.Weights != resumed.Weights ||
		!reflect.DeepEqual(uninterrupted.Optimizer, resumed.Optimizer) || !reflect.DeepEqual(uninterrupted.RNG, resumed.RNG) {
		t.Fatal("fresh-process production resume changed stream, weights, optimizer or RNG")
	}
	lastSource := func(output string) string {
		var last string
		for line := range strings.SplitSeq(output, "\n") {
			if strings.HasPrefix(line, "audio source=") {
				last = line
			}
		}
		return last
	}
	if lastSource(resumedOutput) == "" || lastSource(uninterruptedOutput) != lastSource(resumedOutput) {
		t.Fatal("resumed source or full-precision loss differs")
	}
	l.store, err = overgodb.Open(l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := modelrecipe.AdaptedTranscriptionDefinition(l.base, resumed.ID())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recipe.RequireDefinition(t.Context(), l.store, candidate.ID); err != nil {
		t.Fatal(err)
	}
	status, found, err := modelrecipe.Status(t.Context(), l.store, candidate.ID)
	if err != nil || !found || status != recipe.StatusCandidate {
		t.Fatalf("training must publish without activating: status=%s err=%v", status, err)
	}
	commit := strings.TrimSpace(baselineCommand(t, l.fixture.root, "git", "rev-parse", "HEAD"))
	handler, _ := newAudioHTTPFixture(t, l.store, candidate, l.policy, commit)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, audioHTTPRequest(t, candidate, l.fixture.payload, true))
	var transcript struct {
		Text string `json:"text"`
	}
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &transcript) != nil || strings.TrimSpace(transcript.Text) == "" {
		t.Fatalf("production checkpoint HTTP reload: status=%d body=%s", response.Code, response.Body)
	}
	cancelled, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)
	cancelOutput := filepath.Join(t.TempDir(), "cancelled")
	interrupted, err := trainingworkflow.Execute(cancelled, trainingworkflow.Request{Repository: l.store, Observations: l.store, Recipe: definition.ID,
		Audio: &request, OutputDirectory: cancelOutput, Steps: 2, Progress: cancelAudioUpdate{cancel: cancel}})
	if !errors.Is(err, context.Canceled) || len(interrupted.Losses) != 1 || interrupted.StreamPosition != 1 || interrupted.Candidate.Valid() || interrupted.Checkpoint.ID().Valid() {
		t.Fatalf("cancelled production update: losses=%v cursor=%d candidate=%s err=%v", interrupted.Losses, interrupted.StreamPosition, interrupted.Candidate, err)
	}
	if _, err := os.Stat(cancelOutput); !os.IsNotExist(err) {
		t.Fatalf("cancelled update published checkpoint: %v", err)
	}
	observation, err := runrecord.RequireServingObservation(t.Context(), l.store, interrupted.Observation)
	if err != nil || observation.Outcome != runrecord.OutcomeCancelled || observation.Usage.InputTokens != 0 {
		t.Fatalf("cancelled audio observation=%+v err=%v", observation, err)
	}
	t.Log("production train command: pinned original Parquet audio, exact library first-update parity, fresh-process resumed source/loss/weights/optimizer/RNG, inactive candidate publication and authenticated HTTP reload; CPU only, no quality improvement or production promotion claimed")
}

type cancelAudioUpdate struct{ cancel context.CancelCauseFunc }

func (w cancelAudioUpdate) Write(data []byte) (int, error) {
	if strings.HasPrefix(string(data), "audio source=") {
		w.cancel(context.Canceled)
	}
	return len(data), nil
}

func mustAudioID(t *testing.T, data []byte) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		t.Fatal(err)
	}
	return id
}
