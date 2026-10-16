package trainingworkflow

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/speechsynth"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func speechDatasetFixture(t *testing.T) (*overgodb.Store, Request, string) {
	t.Helper()
	root := t.TempDir()
	directory := writeTinySpeech(t, filepath.Join(root, "model"))
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	commit := func(batch artifact.Batch) {
		t.Helper()
		if _, err := artifact.Publish(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
	}
	inventory, err := speechsynth.Inventory(directory)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := inventory.Batch("model")
	if err != nil {
		t.Fatal(err)
	}
	commit(batch)
	base, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "speech-base", base); err != nil {
		t.Fatal(err)
	}
	model, err := speechsynth.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	facts, err := speechTrainingFacts(model)
	if err != nil {
		t.Fatal(err)
	}
	conditioning, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "fixture/text-bos"), "text-bos")
	if err != nil {
		t.Fatal(err)
	}
	profile := trainingdata.SpeechLatentProfile{Model: base.Model, Transform: facts[0].Descriptor.ID, Noise: facts[1].Descriptor.ID,
		Conditioning: conditioning.Descriptor.ID, Width: 2, MaximumFrames: 3, MaximumTextBytes: 64, MaximumEncodedBytes: 2048}
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentTokenizer {
			profile.Tokenizer = component.Artifact
		}
		if component.Role == artifact.ComponentWeights {
			profile.Codec = component.Artifact
		}
	}
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	commit(artifact.Batch{Key: "processing", Contents: append(facts, conditioning, profileContent)})
	source := testutil.ArtifactID(t, artifact.KindFile, "independent fixture latent producer")
	var raw bytes.Buffer
	for index, text := range []string{"a b", "b c", "c a"} {
		if err := json.NewEncoder(&raw).Encode(trainingdata.SpeechLatentRecord{Profile: profileContent.Descriptor.ID, Source: source,
			SourceRecord: fmt.Sprintf("source-%d", index), Text: text, Frames: 2, Seed: int64(index + 7), Latents: []float32{0.5 + float32(index)/8, -0.25, 0.75, 0.125}}); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(root, "targets.jsonl")
	if err := os.WriteFile(path, raw.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	asset, err := artifact.IdentifyBytes(artifact.KindFile, raw.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	version, err := dataset.NewVersion([]dataset.Asset{{Name: "latents", Artifact: asset, Records: 3}})
	if err != nil {
		t.Fatal(err)
	}
	training, err := dataset.NewMembership(version.ID, 9, "train", []dataset.Record{{ID: dataset.RecordID(version.ID, "latents", 0), Group: "one"}, {ID: dataset.RecordID(version.ID, "latents", 1), Group: "two"}})
	if err != nil {
		t.Fatal(err)
	}
	heldout, err := dataset.NewMembership(version.ID, 9, "heldout", []dataset.Record{{ID: dataset.RecordID(version.ID, "latents", 2), Group: "three"}})
	if err != nil {
		t.Fatal(err)
	}
	versionContent, err := version.Content()
	if err != nil {
		t.Fatal(err)
	}
	trainContent, err := training.Content()
	if err != nil {
		t.Fatal(err)
	}
	heldContent, err := heldout.Content()
	if err != nil {
		t.Fatal(err)
	}
	commit(artifact.Batch{Key: "dataset", Contents: []artifact.Content{versionContent, trainContent, heldContent},
		Artifacts: []artifact.Descriptor{{ID: asset, Size: uint64(raw.Len())}, {ID: source}},
		Lineage:   append(version.Lineage(), training.Lineage(), heldout.Lineage()), Locations: []artifact.LocationEvent{{Action: artifact.LocationAdd, Location: artifact.Location{Artifact: asset, Kind: artifact.LocationFile, Value: path}}}})
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{Name: "fixture speech", Kind: trainingprogram.ObjectiveLatentSequence,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityAudio}},
		Dataset:   version.ID, Split: training.ID, Processors: []artifact.ID{profileContent.Descriptor.ID},
		Loss: testutil.ArtifactID(t, artifact.KindProfile, "loss"), Evaluation: testutil.ArtifactID(t, artifact.KindProfile, "evaluation"), Metric: trainingprogram.MetricAudioSNR,
		Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "synthetic training fixture; no model quality claim")}, Authority: trainingprogram.ObjectiveApproved})
	if err != nil {
		t.Fatal(err)
	}
	trainingRecipe, err := bootstrapRegisteredObjective(t.Context(), store, base.Model, objective)
	if err != nil {
		t.Fatal(err)
	}
	return store, Request{Repository: store, Recipe: trainingRecipe, Host: true, Steps: 2, OutputDirectory: filepath.Join(root, "checkpoint"),
		Speech: &SpeechTrainingSpec{BaseRecipe: base.ID, Profile: profile, MemoryBytes: 4096, Seed: 19, Shuffle: true}}, directory
}

func TestSpeechDatasetTraining(t *testing.T) {
	t.Parallel()
	store, request, _ := speechDatasetFixture(t)
	whole, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.Steps, request.OutputDirectory = 1, filepath.Join(t.TempDir(), "first")
	first, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ResumeDirectory, request.OutputDirectory = request.OutputDirectory, filepath.Join(t.TempDir(), "resumed")
	resumed, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if whole.Checkpoint.Weights != resumed.Checkpoint.Weights || !reflect.DeepEqual(whole.Checkpoint.Optimizer, resumed.Checkpoint.Optimizer) ||
		!reflect.DeepEqual(whole.Checkpoint.Stream, resumed.Checkpoint.Stream) || !reflect.DeepEqual(whole.Checkpoint.RNG, resumed.Checkpoint.RNG) ||
		whole.Losses[0] != first.Losses[0] || whole.Losses[1] != resumed.Losses[0] || resumed.StreamPosition != 2 {
		t.Fatalf("dataset continuation differs: whole=%+v first=%+v resumed=%+v", whole, first, resumed)
	}
	for _, failure := range []string{"order", "normalization", "tokenizer", "codec", "cancelled", "cancel-after-step", "existing-output"} {
		t.Run(failure, func(t *testing.T) {
			bad := request
			spec := *bad.Speech
			bad.Speech, bad.OutputDirectory = &spec, filepath.Join(t.TempDir(), "rejected")
			ctx := t.Context()
			switch failure {
			case "order":
				spec.Seed++
			case "normalization":
				spec.Profile.Transform = testutil.ArtifactID(t, artifact.KindProfile, "wrong")
			case "tokenizer":
				spec.Profile.Tokenizer = testutil.ArtifactID(t, artifact.KindFile, "wrong")
			case "codec":
				spec.Profile.Codec = testutil.ArtifactID(t, artifact.KindTensorSet, "wrong")
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "cancel-after-step":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				defer cancel()
				bad.Progress = speechCancelWriter{cancel}
			case "existing-output":
				bad.OutputDirectory = request.ResumeDirectory
			}
			head, sequence := store.Head()
			if _, err := Execute(ctx, bad); err == nil {
				t.Fatal("invalid continuation admitted")
			}
			if next, count := store.Head(); next != head || count != sequence {
				t.Fatal("rejected continuation mutated store")
			}
			if failure != "existing-output" {
				if _, err := os.Stat(bad.OutputDirectory); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("failed run published checkpoint: %v", err)
				}
			}
		})
	}
}

func TestSpeechCheckpointPublication(t *testing.T) {
	t.Parallel()
	store, request, directory := speechDatasetFixture(t)
	result, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	status, found, err := modelrecipe.Status(t.Context(), store, result.Candidate)
	if err != nil || !found || status != recipe.StatusCandidate {
		t.Fatalf("publication activated recipe: %s %v", status, err)
	}
	definition, err := recipe.RequireDefinition(t.Context(), store, result.Candidate)
	if err != nil {
		t.Fatal(err)
	}
	voiceDirectory := filepath.Join(directory, "embeddings")
	if err := os.Mkdir(voiceDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	header := []byte(`{"transformer.layers.0.self_attn/cache":{"dtype":"F32","shape":[2,1,1,2,2],"data_offsets":[0,32]},"transformer.layers.0.self_attn/offset":{"dtype":"I64","shape":[1],"data_offsets":[32,40]}}`)
	voice := append(binary.LittleEndian.AppendUint64(nil, uint64(len(header))), header...)
	voice = append(voice, make([]byte, 32)...)
	voice = binary.LittleEndian.AppendUint64(voice, 1)
	if err := os.WriteFile(filepath.Join(voiceDirectory, "fixture.safetensors"), voice, 0o600); err != nil {
		t.Fatal(err)
	}
	load := func() *speechsynth.Synthesizer {
		t.Helper()
		s, err := speechsynth.LoadSynthesizer(directory)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	read := func(s *speechsynth.Synthesizer) speechsynth.Audio {
		t.Helper()
		audio, err := s.Speak(t.Context(), speechsynth.DocumentRequest{Text: "a b.", Voice: "fixture", Seed: 7})
		if err != nil {
			t.Fatal(err)
		}
		return audio
	}
	untrained := read(load())
	synth := load()
	if err := synth.RestoreRecipe(t.Context(), store, definition); err != nil {
		t.Fatal(err)
	}
	trained := read(synth)
	if len(trained.PCM) == 0 || slices.Equal(untrained.PCM, trained.PCM) {
		t.Fatal("serving did not apply trained weights")
	}
	for _, failure := range []string{"topology", "lineage", "cancelled"} {
		t.Run(failure, func(t *testing.T) {
			candidate := definition
			var repository artifact.Repository = store
			ctx := t.Context()
			switch failure {
			case "topology":
				candidate.Model = testutil.ArtifactID(t, artifact.KindModel, "foreign")
			case "lineage":
				repository = speechMissingLineage{store}
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			if err := synth.RestoreRecipe(ctx, repository, candidate); err == nil {
				t.Fatal("invalid checkpoint binding served")
			}
			if !reflect.DeepEqual(read(synth), trained) {
				t.Fatal("refused reload mutated served model")
			}
		})
	}
	if err := os.Remove(filepath.Join(request.OutputDirectory, "momentum.bin")); err != nil {
		t.Fatal(err)
	}
	reloaded := load()
	if err := reloaded.RestoreRecipe(t.Context(), store, definition); err != nil {
		t.Fatal(err)
	}
	if again := read(reloaded); !reflect.DeepEqual(again, trained) {
		t.Fatal("serving reload differs")
	}
	if _, err := trainingprogram.LoadCheckpoint(request.OutputDirectory); err == nil {
		t.Fatal("resume ignored missing optimizer payload")
	}
	weightsPath := filepath.Join(request.OutputDirectory, trainingprogram.CheckpointWeights)
	weights, err := os.ReadFile(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	weights[len(weights)-1] ^= 1
	if err := os.WriteFile(weightsPath, weights, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := load().RestoreRecipe(t.Context(), store, definition); err == nil {
		t.Fatal("corrupt checkpoint served")
	}
	if entries, err := filepath.Glob(filepath.Join(filepath.Dir(request.OutputDirectory), ".overgo-checkpoint-*")); err != nil || len(entries) != 0 {
		t.Fatalf("publication retained staging: %v %v", entries, err)
	}
}

type speechCancelWriter struct{ cancel context.CancelFunc }

func (w speechCancelWriter) Write(data []byte) (int, error) {
	w.cancel()
	return len(data), nil
}

type speechMissingLineage struct{ artifact.Repository }

func (speechMissingLineage) Parents(context.Context, artifact.ID) ([]artifact.Lineage, error) {
	return nil, nil
}

// TestSpeechDatasetCommand retains the route's publication contract on the same dataset fixture.
func TestSpeechDatasetCommand(t *testing.T) {
	t.Parallel()
	store, request, _ := speechDatasetFixture(t)
	storePath := filepath.Join(filepath.Dir(request.OutputDirectory), "store")
	manifest := filepath.Join(filepath.Dir(request.OutputDirectory), "training.json")
	data, err := json.Marshal(request.Speech)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifest, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), "go", "run", "./cmd/train", "-route", "speech", "-recipe", request.Recipe.String(), "-manifest", manifest, "-out", request.OutputDirectory, "-store", storePath, "-host", "-steps", "2")
	command.Dir = testutil.RepoRoot(t)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("speech command: %v\n%s", err, output)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(request.OutputDirectory)
	if err != nil || checkpoint.Optimizer.Step != 2 || checkpoint.Stream.Position != 2 {
		t.Fatalf("command checkpoint=%+v %v", checkpoint, err)
	}
}
