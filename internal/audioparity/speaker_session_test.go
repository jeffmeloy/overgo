package audioparity

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

func verifySpeakerModel(t *testing.T, modelRoot string) {
	t.Helper()
	for _, file := range []struct {
		name, hash string
		bytes      uint64
	}{
		{"model.safetensors", "e8abcc5f3a82ff23134c98a37f70fef3f159611f394bb191a0ad0a6f4b052974", 494206256},
		{"config.json", "bb49355fc30581d6c320d3b7ac0f37f9d4210425bfe9b6745824687318ca63a3", 2096},
		{"processor_config.json", "6110cad4e4ee1e9c49a12b36a4047b320b34640e2f9570deff89279d74ec3efc", 413},
		{"README.md", "b7845ba2e8e724bf490fc0e613c350cba9377a5b6fe7fb6897ee3e441eaf4b1e", 16223},
	} {
		verifyASRFile(t, filepath.Join(modelRoot, file.name), alignmentFileID(t, file.hash), file.bytes)
	}
}

func verifySpeakerStoredSession(t *testing.T, root string) {
	modelRoot := filepath.Join(root, "model")
	verifySpeakerModel(t, modelRoot)
	sourceDirectory, commit := snapshotAudioSource(t)
	storePath := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publication := audioPublication{store: store}
	inventory, err := modelartifact.FromFiles(modelRoot, []modelartifact.FileSpec{
		{Path: filepath.Join(modelRoot, "model.safetensors"), Name: "weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(modelRoot, "config.json"), Name: "config.json", Role: artifact.ComponentConfig},
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := inventory.Batch("speaker/model")
	publication.commit(t, batch, err)
	licenseData, err := os.ReadFile(filepath.Join(modelRoot, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	licenseID, err := artifact.IdentifyBytes(artifact.KindFile, licenseData)
	if err != nil {
		t.Fatal(err)
	}
	publication.commit(t, artifact.Batch{Key: "speaker/license", Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: licenseID, Size: uint64(len(licenseData)), MediaType: "text/markdown"}, Data: licenseData}}}, nil)
	var declaration struct {
		Encoder  speechrecognition.Declaration
		Activity speechrecognition.ActivityBinding
	}
	readAudioFixtureJSON(t, "testdata/speaker_activity_declaration.json", &declaration)
	profile := speechrecognition.SpeakerProfile{Model: inventory.Manifest.ID, Inventory: inventory.TensorInventory.ID, License: licenseID, MaximumOffset: 1e-3, Encoder: declaration.Encoder, Activity: declaration.Activity,
		Boundary: speechrecognition.SpeakerBoundary{Threshold: .5, FrameSamples: 1280, TickSamples: 160, FinalBoundary: "last-tick-clipped-to-source"}}
	readAudioFixtureJSON(t, "testdata/speaker_frontend.json", &profile.Frontend)
	profile, err = speechrecognition.NewSpeakerProfile(profile)
	if err != nil {
		t.Fatal(err)
	}
	batch, err = profile.Batch("speaker/profile")
	publication.commit(t, batch, err)
	definition, err := modelrecipe.DiarizationDefinition(profile.Model, profile.ID, profile.Inventory)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "speaker/recipe", definition); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		t.Fatal(err)
	}
	batch, err = environment.Batch("speaker/environment")
	publication.commit(t, batch, err)
	session, err := speechrecognition.LoadSession(t.Context(), store, definition.ID, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())
	corpus, annotations := verifySpeakerCorpus(t, root, &publication)
	var oracle []struct {
		File  string
		Turns []recipecontract.SpeechTurn
	}
	readAudioFixtureJSON(t, "testdata/speaker_reference_turns.json", &oracle)
	if corpus.Selected != 3 || corpus.Scanned != corpus.Selected || len(corpus.Clips) != corpus.Selected || len(oracle) != corpus.Selected {
		t.Fatal("selection denominator differs")
	}
	var policy dataset.AudioInspectionPolicy
	readAudioFixtureJSON(t, "../dataset/testdata/audio_inspection_policy.json", &policy)
	completed := 0
	var commandInputs []map[string]any
	var expectedScores []evaluation.SpeechTurnScore
	var expectedWords [][]speechrecognition.AttributedWord
	for i, clip := range corpus.Clips {
		name := filepath.Base(clip.File)
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(root, name))
			if err != nil {
				t.Fatal(err)
			}
			id, err := artifact.IdentifyBytes(artifact.KindFile, data)
			if err != nil || id.DigestHex() != clip.SHA256 {
				t.Fatal("audio differs")
			}
			publication.commit(t, artifact.Batch{Key: "speaker/source/" + name, Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(data))}}}, nil)
			policy.MaximumEncodedBytes = uint64(len(data))
			policy.MaximumSamples = clip.End - clip.Start
			lease, err := session.Lease(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			predicted, run, err := lease.Diarize(t.Context(), data, dataset.AudioPayloadOrigin{Container: id}, policy, speechrecognition.RunBinding{Key: "speaker/run/" + name, CodeCommit: commit, Environment: environment.ID})
			if releaseErr := lease.Release(); releaseErr != nil {
				t.Fatal(releaseErr)
			}
			if err != nil {
				t.Fatal(err)
			}
			if run.Recipe != definition.ID || run.CodeCommit != commit || run.Environment != environment.ID || run.Outcome != runrecord.OutcomeSucceeded || len(run.Outputs) != 1 {
				t.Fatalf("run binding: %+v", run)
			}
			persisted, err := speechrecognition.RequireSpeechTurns(t.Context(), store, run.Outputs[0])
			if err != nil || !reflect.DeepEqual(predicted, persisted) {
				t.Fatalf("stored output differs: %v", err)
			}
			if oracle[i].File != name || !reflect.DeepEqual(predicted.Turns, oracle[i].Turns) {
				t.Fatalf("reference turns differ: got %+v want %+v", predicted.Turns, oracle[i].Turns)
			}
			reference := recipecontract.SpeechTurns{Source: predicted.Source}
			for _, s := range clip.Segments {
				reference.Turns = append(reference.Turns, recipecontract.SpeechTurn{Speaker: s.Speaker, Span: recipecontract.SampleSpan{Start: s.Start, End: s.End}})
			}
			score, err := evaluation.ScoreSpeechTurns(t.Context(), reference, predicted, recipecontract.SampleSpan{End: clip.End - clip.Start}, adapterAcceptanceMemory)
			if err != nil {
				t.Fatal(err)
			}
			referenceContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo/speech-turns/v1"), reference)
			if err != nil {
				t.Fatal(err)
			}
			archiveID := alignmentFileID(t, "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d")
			publication.commit(t, artifact.Batch{Key: "speaker/reference/" + name, Contents: []artifact.Content{referenceContent}, Lineage: artifact.DependencyLineage(referenceContent.Descriptor.ID, archiveID, reference.Source.Audio, reference.Source.Profile)}, nil)
			items := speakerConditioningWords(t, annotations, clip)
			texts := make([]string, len(items))
			for i, item := range items {
				texts[i] = item.Text
			}
			transcriptContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema), recipecontract.Transcription{Source: predicted.Source, Text: strings.Join(texts, " "), Language: "en"})
			if err != nil {
				t.Fatal(err)
			}
			alignment := recipecontract.TimestampedAlignment{Source: predicted.Source, Transcription: transcriptContent.Descriptor.ID, Items: items}
			if err := alignment.Validate(); err != nil {
				t.Fatal(err)
			}
			alignmentContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo/audio-alignment/v1"), alignment)
			if err != nil {
				t.Fatal(err)
			}
			lineage := append(artifact.DependencyLineage(transcriptContent.Descriptor.ID, archiveID, predicted.Source.Audio, predicted.Source.Profile), artifact.DependencyLineage(alignmentContent.Descriptor.ID, transcriptContent.Descriptor.ID, archiveID)...)
			publication.commit(t, artifact.Batch{Key: "speaker/annotation-conditioned-words/" + name, Contents: []artifact.Content{transcriptContent, alignmentContent}, Lineage: lineage}, nil)
			words, err := speechrecognition.AttributeWords(t.Context(), alignment, predicted, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i, word := range words {
				if word.Word != items[i] {
					t.Fatal("composition altered aligned word")
				}
			}
			commandInputs = append(commandInputs, map[string]any{"audio": dataset.AudioPayloadReference{Path: filepath.Join(root, name), Audio: id, Origin: dataset.AudioPayloadOrigin{Container: id}}, "reference": referenceContent.Descriptor.ID, "span": recipecontract.SampleSpan{End: clip.End - clip.Start}, "alignment": alignmentContent.Descriptor.ID})
			expectedScores = append(expectedScores, score)
			expectedWords = append(expectedWords, words)
			encoded, err := json.Marshal(score)
			if err != nil {
				t.Fatal(err)
			}
			t.Log(string(encoded))
			completed++
		})
	}
	if completed != corpus.Selected {
		t.Fatal("incomplete selected cases")
	}
	verifySpeakerSessionControls(t, root, session, &publication, policy, commit, environment.ID)
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	verifySpeakerCommand(t, sourceDirectory, commit, storePath, definition.ID, profile, policy, commandInputs, expectedScores, expectedWords)
	fmt.Printf("HONESTY speaker stored CPU sessions=%d/%d; fresh native command parity; annotation-conditioned word preservation; AMI training overlap; no held-out generalization or GPU credit\n", completed, corpus.Selected)
}
