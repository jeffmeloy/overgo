package server

import (
	"context"
	"encoding/json"
	"net/http"
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
