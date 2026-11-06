//go:build windows

package mediacapability

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/mosstrain"
	"overgo/internal/overgodbtest"
	"overgo/internal/peft"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// picturePublished draws a request from a model directory through the image
// capability, as a registered candidate of its own.
func picturePublished(t *testing.T, root string, request latentimage.Request) ([]byte, string) {
	t.Helper()
	pictures := catalogPictures(t, root)
	director := pictureService(t)
	raw := testcheck.Must(json.Marshal(request))(t)
	output := testcheck.Must(director.Executor()(capabilityruntime.FreshRun(t.Context()), pictures.store, root, pictures.execution, string(raw)))(t)
	measured, ok := output.(capabilityruntime.Measured)
	if !ok {
		t.Fatal("missing measured output")
	}
	image, ok := measured.Output.(latentimage.EncodedImage)
	if !ok {
		t.Fatal("missing image")
	}
	testcheck.NoError(t, director.Close(t.Context()))
	return image.Data, pictures.execution.Program.Definition().Model.String()
}

// catalogPictures is a store a picture by the model in root is served from,
// its recipe defined from the directory as the catalog defines it.
func catalogPictures(t *testing.T, root string) servedPictures {
	t.Helper()
	source := testcheck.Must(Catalog[recipe.TaskImageGen].Resolve(t.Context(), root))(t)
	definition, facts, err := source.Define(source.Inventory.Manifest.ID)
	testcheck.NoError(t, err)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	store := overgodbtest.Open(t, t.TempDir())
	batch := testcheck.Must(source.Inventory.Batch(t.Name()))(t)
	batch.Contents = append(batch.Contents, facts...)
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	if err = modelrecipe.PublishFacts(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	return servedPictures{store: store, directory: root, execution: modelrecipetest.CandidateExecution(t, store, program)}
}

// TestMOSSTrainedModelServes holds a published model to the capability the base
// serves through, under an identity of its own: an adapter of zero merges to
// the base's very picture, and one that changes the weights changes it.
func TestMOSSTrainedModelServes(t *testing.T) {
	cudatest.Require(t)
	root := testutil.ExtractTarGzip(t, filepath.Join("..", "..", "fixtures", "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	spec := testcheck.Must(latentimage.ReadRotaryTransformer(root))(t)
	options := mosstrain.Options{Rank: 2, Alpha: 2}
	config := peft.Config{BaseModel: root, Rank: options.Rank, Alpha: options.Alpha}
	request := latentimage.Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 42}
	base, baseModel := picturePublished(t, root, request)

	silent := t.TempDir()
	testcheck.NoError(t, mosstrain.PublishServed(root, filepath.Join(silent, "model"), peft.Adapter{Config: config, Modules: mosstrain.InitialAdapter(spec, options, 1)}))
	same, _ := picturePublished(t, filepath.Join(silent, "model"), request)
	if !bytes.Equal(same, base) {
		t.Fatal("an adapter of zero changed the base's picture")
	}

	modules := mosstrain.InitialAdapter(spec, options, 1)
	for name, module := range modules {
		for index := range module.Up {
			module.Up[index] = 0.1
		}
		modules[name] = module
	}
	trained := t.TempDir()
	testcheck.NoError(t, mosstrain.PublishServed(root, filepath.Join(trained, "model"), peft.Adapter{Config: config, Modules: modules}))
	different, trainedModel := picturePublished(t, filepath.Join(trained, "model"), request)
	if bytes.Equal(different, base) {
		t.Fatal("a trained adapter left the base's picture")
	}
	if trainedModel == baseModel {
		t.Fatal("a trained model took the base's identity")
	}
}
