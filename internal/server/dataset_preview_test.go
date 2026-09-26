package server

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

type datasetPreviewResolver struct {
	dataset *trainingdata.Dataset
}

func (resolver datasetPreviewResolver) OpenDataset(context.Context, string) (*trainingdata.Dataset, error) {
	return resolver.dataset, nil
}

func TestDatasetPreviewUsesProductionProcessor(t *testing.T) {
	t.Parallel()
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "gui-preview-dataset")
	splitID := testutil.ArtifactID(t, artifact.KindDatasetShard, "gui-preview-split")
	processorID := testutil.ArtifactID(t, artifact.KindProfile, "gui-preview-processor")
	var calls atomic.Uint64
	processor := func(_ context.Context, record trainingdata.RawRecord) (trainingdata.Example, error) {
		calls.Add(1)
		return trainingdata.Example{ID: record.ID, Group: record.Group, Values: []trainingdata.Value{{
			Role: trainingdata.RoleInput, Modality: recipecontract.ModalityText,
			Encoding: trainingdata.EncodingUTF8, Data: append([]byte("processed:"), record.Data...),
		}}}, nil
	}
	dataset, err := trainingdata.MaterializeRecords(trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
	}, processorID, []trainingdata.RawRecord{{ID: "record", Data: []byte("source")}}, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityText}, Process: processor,
	})
	if err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{DatasetPreview: datasetPreviewResolver{dataset: dataset}}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	response := serveTestRequest(handler, http.MethodPost, "/datasets/preview", `{"name":"fixture","position":0,"limit":1}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var preview trainingdata.PreviewResult
	if err := json.Unmarshal(response.Body.Bytes(), &preview); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 || len(preview.Examples) != 1 || len(preview.Examples[0].Values) != 1 ||
		preview.Examples[0].Values[0].Text != "processed:source" {
		t.Fatalf("calls=%d preview=%+v", calls.Load(), preview)
	}
}

// TestDatasetPreviewOpensRegisteredDataset: with no launcher resolver the
// store answers the preview, a registered text dataset reads every record
// with no split, an unregistered name is refused by name, and a server
// without a store says so.
func TestDatasetPreviewOpensRegisteredDataset(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	corpus := t.TempDir()
	if err := os.WriteFile(filepath.Join(corpus, "notes.txt"), []byte("the sea is wide"), 0o644); err != nil {
		t.Fatal(err)
	}
	directory, err := json.Marshal(corpus)
	if err != nil {
		t.Fatal(err)
	}
	if registered := serveTestRequest(handler, http.MethodPost, "/library/register",
		`{"kind":"dataset","name":"notes","directory":`+string(directory)+`}`); registered.Code != http.StatusOK {
		t.Fatalf("register status=%d body=%s", registered.Code, registered.Body.String())
	}
	response := serveTestRequest(handler, http.MethodPost, "/datasets/preview", `{"name":"notes","position":0,"limit":1}`)
	var preview trainingdata.PreviewResult
	if err := json.Unmarshal(response.Body.Bytes(), &preview); response.Code != http.StatusOK || err != nil {
		t.Fatalf("preview status=%d body=%s", response.Code, response.Body.String())
	}
	if len(preview.Examples) != 1 || len(preview.Examples[0].Values) != 1 || preview.Examples[0].Values[0].Text != "the sea is wide" {
		t.Fatalf("preview=%+v", preview)
	}
	unregistered := serveTestRequest(handler, http.MethodPost, "/datasets/preview", `{"name":"absent","position":0,"limit":1}`)
	if unregistered.Code != http.StatusBadRequest || !strings.Contains(unregistered.Body.String(), `dataset \"absent\" is not registered`) {
		t.Fatalf("unregistered status=%d body=%s", unregistered.Code, unregistered.Body.String())
	}
	storeless, err := New(Config{}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer storeless.Close()
	if refused := serveTestRequest(storeless, http.MethodPost, "/datasets/preview", `{"name":"notes","position":0,"limit":1}`); refused.Code != http.StatusNotImplemented {
		t.Fatalf("storeless status=%d body=%s", refused.Code, refused.Body.String())
	}
}
