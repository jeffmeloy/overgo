//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func compactGeneratorFixture(t *testing.T) (string, *safetensors.Source) {
	t.Helper()
	root := testutil.ExtractTarGzip(t, filepath.Join("..", "..", "fixtures", "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	id, err := artifact.IdentifyFile(artifact.KindFile, filepath.Join(root, "reference.safetensors"))
	if err != nil || id.DigestHex() != "084332711dc6be4bae81d54cec4fd5319beb5d209d61855706c16874e9d50d86" {
		t.Fatal("reference identity", err)
	}
	s := testcheck.Must(safetensors.OpenFile(filepath.Join(root, "reference.safetensors")))(t)
	t.Cleanup(func() { s.Close() })
	for _, metadata := range s.Metadata {
		var files map[string]string
		if err = json.Unmarshal([]byte(metadata["files"]), &files); err != nil {
			t.Fatal(err)
		}
		for name, want := range files {
			if !filepath.IsLocal(name) {
				t.Fatal(name)
			}
			id, e := artifact.IdentifyFile(artifact.KindFile, filepath.Join(root, name))
			if e != nil || id.DigestHex() != want {
				t.Fatal("component identity", name, e)
			}
		}
	}
	return root, s
}

func TestMOSSGeneratorComposition(t *testing.T) {
	cudatest.Require(t)
	root, source := compactGeneratorFixture(t)
	var contract struct {
		Comparison struct {
			Tolerances map[string]mossStageTolerance `json:"dtype_tolerances"`
		}
	}
	testcheck.NoError(t, readJSON(filepath.Join("..", "..", "fixtures", "moss", "oracle-contract.json"), &contract))
	identity := testcheck.Must(artifact.IdentifyFile(artifact.KindFile, filepath.Join(root, "reference.safetensors")))(t)
	for _, test := range []struct {
		name, prompt, negative string
		h, w                   int
		seed                   int64
	}{{"rectangle", "blue cube", "red", 32, 48, 42}, {"portrait", "green tree", "", 48, 32, 7}} {
		t.Run(test.name, func(t *testing.T) {
			request := Request{Prompt: test.prompt, NegativePrompt: test.negative, Height: test.h, Width: test.w, Seed: test.seed, Steps: 3}
			var report bytes.Buffer
			if err := compareGeneration(t.Context(), root, test.name, test.name+".", request, identity, source, contract.Comparison.Tolerances, &report); err != nil {
				t.Log(report.String())
				t.Fatal(err)
			}
			t.Log(report.String())
			if bytes.Count(report.Bytes(), []byte("\n")) != 42 {
				t.Fatal("native boundary coverage")
			}
		})
	}
}

// TestMOSSGeneratorRefusesPicturesBeyondTheCard holds an admitted request to a
// failed operation where its shapes overflow or its activations exceed the
// card, before any host allocation of its extent.
func TestMOSSGeneratorRefusesPicturesBeyondTheCard(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	for _, extent := range []int{1 << 34, 1 << 24} {
		request := Request{Prompt: "blue cube", Width: extent, Height: extent, Steps: 1}
		if _, err := LoadGenerator(t.Context(), root, profile, request); err == nil {
			t.Fatal("picture beyond the card admitted", extent)
		}
	}
}

func TestMOSSGeneratorLifecycle(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	request := Request{Prompt: "blue cube", Width: 48, Height: 32, Steps: 1, Seed: 42}
	for _, stage := range []string{"prepare", "integrate", "decode"} {
		t.Run(stage, func(t *testing.T) {
			g, e := LoadGenerator(t.Context(), root, profile, request)
			if e != nil {
				t.Fatal(e)
			}
			defer g.Close(context.Background())
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			g.observe = func(name string, _ []float32) error {
				if stage == "prepare" && name == "conditioning.0.hidden" || stage == "integrate" && name == "prediction.0" || stage == "decode" && name == "pixels" {
					cancel()
				}
				return nil
			}
			session, e := g.prepare(ctx, request)
			if e == nil {
				session, e = g.integrate(ctx, session)
			}
			if e == nil {
				_, e = g.decode(ctx, session)
			}
			if !errors.Is(e, context.Canceled) {
				t.Fatal("cancellation lost", e)
			}
			if g.completed || g.rotary.latent != nil || g.rotary.runtime.StaticBytes() != 0 {
				t.Fatal("cancelled request published completion or retained temporary resources")
			}
			if e = g.Reset(t.Context(), request); e == nil {
				t.Fatal("failed request reused as completed")
			}
			if e = g.Close(ctx); e != nil {
				t.Fatal(e)
			}
			if g.rotary != nil {
				t.Fatal("closed state retained")
			}
			if e = g.Close(ctx); e != nil {
				t.Fatal(e)
			}
		})
	}
	for _, change := range []func(*Request){func(r *Request) { r.Width++ }, func(r *Request) { r.DynamicShiftMu = 1 }, func(r *Request) { r.NumTrainTimesteps = 7 }, func(r *Request) { r.Steps = -1 }} {
		invalid := request
		change(&invalid)
		if g, e := LoadGenerator(t.Context(), root, profile, invalid); e == nil {
			g.Close(context.Background())
			t.Fatal("invalid request accepted", invalid)
		}
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, e := LoadGenerator(cancelled, root, profile, request); !errors.Is(e, context.Canceled) {
		t.Fatal("cancelled load accepted", e)
	}
}

func TestResidentWeightShapeRefusal(t *testing.T) {
	cudatest.Require(t)
	root := t.TempDir()
	testcheck.NoError(t, safetensors.Save(filepath.Join(root, "weights.safetensors"), map[string][]float32{"weight": {1, 2, 3, 4, 5, 6}}, map[string][]int{"weight": {2, 3}}, nil))
	b := tensor.NewBuilder()
	weight := b.Input("weight", dtype.F32, tensor.MustShape(2, 3))
	out := b.Scale(weight, 2)
	session, err := graphruntime.NewResidentSession(0)
	testcheck.NoError(t, err)
	defer session.Close(context.Background())
	if _, err = compileResidentWeights(t.Context(), session, "shape refusal", root, tensor.WeightBindings{weight}, nil, out); err == nil {
		t.Fatal("equal bytes with transposed dimensions accepted")
	}
	if session.StaticBytes() != 0 {
		t.Fatal("refused shape retained weights")
	}
}

func TestMOSSProfileBindsSourcePolicy(t *testing.T) {
	root, _ := compactGeneratorFixture(t)
	profile, err := ResolveProfile(root)
	testcheck.NoError(t, err)
	if profile.Prompt.MaxTokens != 4096 || !profile.Prompt.EnableThinking || profile.Rotary == nil || profile.Rotary.Guidance != 3 || profile.Rotary.SpatialScale != 16 || profile.Sampling.DefaultSteps != 50 {
		t.Fatal("source policy differs")
	}
	path := filepath.Join(root, "tokenizer", "chat_template.jinja")
	if err = os.WriteFile(path, []byte("b{{ messages[0]['content'] }}"), 0600); err != nil {
		t.Fatal(err)
	}
	changed := testcheck.Must(ResolveProfile(root))(t)
	if profile.ID == changed.ID {
		t.Fatal("template change absent from profile identity")
	}
	path = filepath.Join(root, "scheduler", "scheduler_config.json")
	data := testcheck.Must(os.ReadFile(path))(t)
	var config map[string]any
	if err = json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	config["invert_sigmas"] = true
	data = testcheck.Must(json.Marshal(config))(t)
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = ResolveProfile(root); err == nil {
		t.Fatal("unsupported scheduler policy accepted")
	}
}

func TestMOSSGeneratorWorkflow(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile, err := ResolveProfile(root)
	testcheck.NoError(t, err)
	inventory := testcheck.Must(modelartifact.FromHFPath(root))(t)
	definition := testcheck.Must(modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentImagePrepare, inventory.Manifest.ID, profile.ID))(t)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	store := overgodbtest.Open(t, t.TempDir())
	content := testcheck.Must(profile.Content())(t)
	batch := testcheck.Must(inventory.Batch(t.Name()))(t)
	batch.Contents = append(batch.Contents, content)
	if _, err = store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	request := Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 42}
	input := testcheck.Must(capabilityruntime.InputContent("overgo.image-gen-input.v1", request))(t)
	runtime := testcheck.Must(workflowruntime.NewForProgram(store, program))(t)
	g := testcheck.Must(LoadGenerator(t.Context(), root, profile, request))(t)
	defer g.Close(context.Background())
	if err = RegisterRuntime(runtime, inventory.Manifest.ID, g); err != nil {
		t.Fatal(err)
	}
	operation := testcheck.Must(workflowruntime.ExecutionID(definition.ID, t.Name()))(t)
	result := testcheck.Must(runtime.ExecuteProgram(t.Context(), t.Name(), operation, nil, program, map[recipe.PortName]workflowruntime.Value{definition.Inputs[0].Name: workflowruntime.ArtifactValue(definition.Inputs[0].Data, request, input)}))(t)
	if result.Run.Outcome != runrecord.OutcomeSucceeded || result.Run.Recipe != definition.ID || len(result.Run.Outputs) != 1 || len(result.NodeWalls) != len(program.Stages()) {
		t.Fatal("incomplete workflow lineage", result.Run)
	}
	image, ok := result.Outputs[definition.Outputs[0].Name].Single()
	if !ok {
		t.Fatal("image missing")
	}
	encoded, ok := image.Value.(EncodedImage)
	if !ok || encoded.Width != request.Width || encoded.Height != request.Height {
		t.Fatal("workflow image differs")
	}
	held, present, err := artifact.ReadContent(t.Context(), store, result.Run.Outputs[0])
	if err != nil || !present || !bytes.Equal(held.Data, encoded.Data) {
		t.Fatal("workflow image was not durably published", err)
	}
}
