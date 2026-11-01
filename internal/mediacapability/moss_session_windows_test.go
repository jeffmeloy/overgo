//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestMOSSResidentSession(t *testing.T) {
	cudatest.Require(t)
	root := testutil.ExtractTarGzip(t, filepath.Join("..", "..", "fixtures", "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	source, err := Catalog[recipe.TaskImageGen].Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(source.Manifests) != 3 {
		t.Fatal("component manifests missing")
	}
	definition, facts, err := source.Define(source.Inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := source.Inventory.Batch(t.Name())
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, facts...)
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	if err = modelrecipe.PublishFacts(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	execution, err := modelrecipe.CompileCandidateExecution(t.Context(), store, program)
	if err != nil {
		t.Fatal(err)
	}
	director := pictureService(t)
	serve := func(ctx context.Context, request latentimage.Request) (capabilityruntime.Measured, error) {
		b, err := json.Marshal(request)
		if err != nil {
			return capabilityruntime.Measured{}, err
		}
		output, err := director.Executor()(capabilityruntime.FreshRun(ctx), store, root, execution, string(b))
		if err != nil {
			return capabilityruntime.Measured{}, err
		}
		measured, ok := output.(capabilityruntime.Measured)
		if !ok {
			return measured, errors.New("missing measured output")
		}
		return measured, nil
	}
	check := func(measured capabilityruntime.Measured, request latentimage.Request) []byte {
		t.Helper()
		image, ok := measured.Output.(latentimage.EncodedImage)
		if !ok || image.Width != request.Width || image.Height != request.Height {
			t.Fatal("image extent or type differs")
		}
		content, err := latentimage.PNGContent(image)
		if err != nil {
			t.Fatal(err)
		}
		held, found, err := artifact.ReadContent(t.Context(), store, content.Descriptor.ID)
		if err != nil || !found || !bytes.Equal(held.Data, image.Data) {
			t.Fatal("PNG not retained", err)
		}
		if len(measured.Phases) != len(program.Stages()) || !measured.Input.Descriptor.ID.Valid() {
			t.Fatal("stage or input lineage missing")
		}
		run, err := runrecord.RequireExactRun(t.Context(), store, measured.Run)
		if err != nil || run.Outcome != runrecord.OutcomeSucceeded {
			t.Fatal("successful run absent", err)
		}
		return image.Data
	}
	first := latentimage.Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 42}
	measured, err := serve(t.Context(), first)
	if err != nil {
		t.Fatal(err)
	}
	baseline := check(measured, first)
	// This exact PNG identifies the composition already compared with the native
	// float authority; native numerical acceptance remains in latentimage tests.
	content, err := latentimage.PNGContent(measured.Output.(latentimage.EncodedImage))
	if err != nil || content.Descriptor.ID.DigestHex() != "99062f8a13f008e4c3f14f45e5a2b063e2ce4e910a59f90b3b2bdc48d32fdeda" {
		t.Fatal("serving changed verified composition", err)
	}
	for _, change := range []func(*latentimage.Request){
		func(r *latentimage.Request) { r.Prompt = "green tree" },
		func(r *latentimage.Request) { r.NegativePrompt = "green" },
		func(r *latentimage.Request) { r.Seed++ },
	} {
		request := first
		change(&request)
		measured, err = serve(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Equal(check(measured, request), baseline) {
			t.Fatal("request change did not reach generator")
		}
		if measured.Load != 0 {
			t.Fatal("compatible request reloaded model")
		}
	}
	// Two requests share one loaded session and cannot overwrite one another.
	var wg sync.WaitGroup
	outputs := make([]capabilityruntime.Measured, 2)
	errorsSeen := make([]error, len(outputs))
	for i := range outputs {
		wg.Go(func() { ; outputs[i], errorsSeen[i] = serve(t.Context(), first) })
	}
	wg.Wait()
	for i, output := range outputs {
		if errorsSeen[i] != nil {
			t.Fatal(errorsSeen[i])
		}
		if !bytes.Equal(check(output, first), baseline) {
			t.Fatal("concurrent request changed output")
		}
	}
	if snapshot := director.Snapshot(); snapshot.Loads != 1 || snapshot.Active != 0 || snapshot.Entries != 1 {
		t.Fatal("unbounded or unreleased session", snapshot)
	}
	ctx, cancel := context.WithCancel(t.Context())
	left := 100
	_, err = serve(checkedContext{ctx, cancel, &left}, first)
	cancel()
	if !errors.Is(err, context.Canceled) || left > 0 {
		t.Fatal("request cancellation lost", err, left)
	}
	measured, err = serve(t.Context(), first)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(check(measured, first), baseline) || measured.Load <= 0 {
		t.Fatal("cancelled session was not replaced cleanly")
	}
	portrait := latentimage.Request{Prompt: "green tree", Width: 32, Height: 48, Steps: 3, Seed: 7}
	measured, err = serve(t.Context(), portrait)
	if err != nil {
		t.Fatal(err)
	}
	check(measured, portrait)
	if measured.Load <= 0 || director.Snapshot().Entries != 1 {
		t.Fatal("extent change did not replace bounded session")
	}
	if err = director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if snapshot := director.Snapshot(); snapshot.Active != 0 || snapshot.Entries != 0 || !snapshot.Closed {
		t.Fatal("close leaked session", snapshot)
	}
}
