package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

func TestTrainSpeechDatasetPreview(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	wave := testutil.MonoPCM16WAV(16000, []int16{8192, -8192})
	audio, _ := artifact.IdentifyBytes(artifact.KindFile, wave)
	path := filepath.Join(t.TempDir(), "source.wav")
	testcheck.NoError(t, os.WriteFile(path, wave, 0600))
	metadata := testcheck.Must(json.Marshal(dataset.SpeechRecord{Audio: audio, Origin: dataset.AudioPayloadOrigin{Container: audio}, Target: "isolated training target"}))(t)
	metadataID, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, metadata)
	version := testcheck.Must(dataset.NewVersion([]dataset.Asset{{Name: "speech", Artifact: metadataID, Records: 1}}))(t)
	split := testcheck.Must(dataset.BuildGroupSplit(version.ID, []dataset.Record{{ID: fmt.Sprintf("%s/speech/0", version.ID), Group: "source"}}, 0, []dataset.SplitPartition{{Name: "selected", Weight: 1}, {Name: "reserved", Weight: 1}}))(t)
	var membership dataset.Membership
	for _, candidate := range split.Memberships {
		if len(candidate.Records) != 0 {
			membership = candidate
		}
	}
	batch := testcheck.Must(split.PublicationBatch("preview/fixture", nil))(t)
	content := testcheck.Must(version.Content())(t)
	batch.Contents = append(batch.Contents, content, artifact.Content{Descriptor: artifact.Descriptor{ID: metadataID, Size: uint64(len(metadata))}, Data: metadata})
	batch.Lineage = append(batch.Lineage, version.Lineage()...)
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: audio, Size: uint64(len(wave))})
	batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: artifact.Location{Artifact: audio, Kind: artifact.LocationFile, Value: path}, Action: artifact.LocationAdd})
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	request := trainingDataPreviewManifest{Dataset: version.ID, Split: membership.ID, MemoryBytes: uint64(len(metadata)), Policy: dataset.AudioInspectionPolicy{MaximumEncodedBytes: uint64(len(wave)), MaximumSamples: 2, ClipThreshold: 1, Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 1.0 / 32768, MaximumAbsoluteDCOffset: 1}}}
	encoded := testcheck.Must(json.Marshal(request))(t)
	manifest := filepath.Join(t.TempDir(), "preview.json")
	testcheck.NoError(t, os.WriteFile(manifest, encoded, 0600))
	var output bytes.Buffer
	testcheck.NoError(t, previewTrainingData(t.Context(), store, manifest, &output))
	var result struct {
		Preview           trainingdata.PreviewResult `json:"preview"`
		ModelLoaded       bool                       `json:"model_loaded"`
		ParametersUpdated bool                       `json:"parameters_updated"`
	}
	testcheck.NoError(t, json.Unmarshal(output.Bytes(), &result))
	if result.ModelLoaded || result.ParametersUpdated || len(result.Preview.Examples) != 1 || result.Preview.State.Position != 1 {
		t.Fatalf("preview scope: %+v", result)
	}
	values := result.Preview.Examples[0].Values
	if len(values) != 2 || values[0].Modality != recipecontract.ModalityAudio || values[0].Role != trainingdata.RoleInput || values[0].SampleRate != 16000 || values[0].Bytes != 8 || values[1].Role != trainingdata.RoleTarget || values[1].Text != "isolated training target" {
		t.Fatalf("typed preview: %+v", values)
	}
	if bytes.Contains(output.Bytes(), []byte(`"data"`)) {
		t.Fatal("preview exposed raw PCM")
	}
}
