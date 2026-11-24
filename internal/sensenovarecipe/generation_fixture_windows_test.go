//go:build windows

package sensenovarecipe

import (
	"context"
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/testutil"
)

// The shared fixtures of the SenseNova generation tests: the request-resources measurement, whose bytes its retained
// record names and which stays untagged until it is measured again, builds with them beside the tagged suites.

const senseNovaProductionPNG = "6ef4065018d328c2945e6de0a8b87fe3eb2aec44039b43207398969712446de0"

// senseNovaPicture is the digest a SenseNova picture is pinned by: its pixels', the SHA-256 of its canonical PNG
// (testutil.CanonicalPNG), which senseNovaProductionPNG names whatever level the picture is written at (one-png-writer).
func senseNovaPicture(t testing.TB, encoded []byte) string {
	t.Helper()
	return fmt.Sprintf("%x", sha256.Sum256(testutil.CanonicalPNG(t, encoded)))
}

func newProductionGenerationFixture(t testing.TB) (*overgodb.Store, recipe.Program, *Generator) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	modelID := testutil.ArtifactID(t, artifact.KindModel, "sensenova-production")
	testutil.PublishArtifact(t, store, modelID)
	flowProfile, err := routedlm.InspectFlowProfile(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	flowContent, err := flowProfile.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/sensenova/runtime-flow-profile", Contents: []artifact.Content{flowContent},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, flowProfile.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	generator, err := LoadGenerator(ctx, store, senseNovaModelDir, definition)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Errorf("close: %v", err)
		}
	})
	return store, program, generator
}

const senseNovaGenerationGold = `..\..\fixtures\sensenova\generation_native_256.json`

type generationLeadershipOracle struct {
	Schema string `json:"schema"`
	Source struct {
		InferencePath  string `json:"inference_path"`
		InferenceSHA   string `json:"inference_sha256"`
		ModelingPath   string `json:"modeling_path"`
		ModelingSHA    string `json:"modeling_sha256"`
		ConfigSHA      string `json:"config_sha256"`
		TensorIndexSHA string `json:"tensor_index_sha256"`
	} `json:"source"`
	Request struct {
		Prompt        string
		Width, Height int
		Steps         int
		Seed          int64
		CFGScale      float32 `json:"cfg_scale"`
		TimestepShift float64 `json:"timestep_shift"`
	} `json:"request"`
	TextInputs struct {
		Conditional   []int `json:"conditional_ids"`
		Unconditional []int `json:"unconditional_ids"`
	} `json:"text_inputs"`
	Steps []generationStep `json:"steps"`
}

// senseNovaModelDir is the real SenseNova artifact under the model roots,
// resolved once from the repository root two directories up: a value, not a
// function of a test, because the pinned production-oracle test reads it as
// one.
var senseNovaModelDir = func() string {
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		panic(err)
	}
	return roots.ResolveModelPath("SenseNova-U1-8B-MoT-Infographic-V3")
}()

type seededLatentOracle struct {
	Request struct {
		Width  int   `json:"width"`
		Height int   `json:"height"`
		Seed   int64 `json:"seed"`
	} `json:"request"`
	Steps []struct {
		Z struct {
			Shape    []int     `json:"shape"`
			Elements int       `json:"elements"`
			Indices  []int64   `json:"indices"`
			Values   []float32 `json:"values"`
		} `json:"z"`
	} `json:"steps"`
}

type generationStep struct {
	Timestep              float64          `json:"timestep"`
	NextTimestep          float64          `json:"next_timestep"`
	Z                     generationSample `json:"z"`
	ConditionalBoundary   generationSample `json:"conditional_boundary"`
	UnconditionalBoundary generationSample `json:"unconditional_boundary"`
	GuidedVelocity        generationSample `json:"guided_velocity"`
	NextZ                 generationSample `json:"next_z"`
}

type generationSample struct {
	Elements int       `json:"elements"`
	Indices  []int     `json:"indices"`
	Values   []float32 `json:"values"`
}
