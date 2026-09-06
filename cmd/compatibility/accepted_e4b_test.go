package main

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// TestAcceptedE4BModalities reads the frozen evidence selection only. It never
// launches inference or promotes a partial or fixture-only model result.
func TestAcceptedE4BModalities(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for canonical E4B evidence acceptance")
	}
	root := testutil.RepoRoot(t)
	var spec modelValidationSpecification
	if err := jsonfile.Decode(filepath.Join(root, "docs", "verification", "e4b-validation.json"), &spec); errors.Is(err, fs.ErrNotExist) {
		t.Skip("integration: the E4B validation specification is absent from this tree; the producer's lane carries it")
	} else if err != nil {
		t.Fatal(err)
	}
	if spec.Model.String() != "model:sha256:fb09299dd00edd7ffdcf8cb48e475d2a9c9e30a22c51f79f6d4d793e983c557b" ||
		spec.Projector.String() != "projector:sha256:fd74ad68955dc65da2ccd2498b6ae39b05685120d2cc9ed1bd3c41f4c9933686" {
		t.Fatal("E4B model/projector identity differs")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := checkModelValidation(t.Context(), store, spec, e4bValidationCells()); err != nil {
		t.Fatal(err)
	}
	inference, active, err := modelrecipe.ActiveRecord(t.Context(), store, spec.Model, recipe.TaskInference)
	if err != nil || !active {
		t.Fatalf("E4B inference binding is not promoted: %v", err)
	}
	definition, bound := inference.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !bound {
		t.Fatal("E4B inference definition is absent")
	}
	projection, active, err := modelrecipe.ActiveRecord(t.Context(), store, spec.Model, recipe.TaskProjection)
	if err != nil || !active {
		t.Fatalf("E4B complete projection binding is not promoted: %v", err)
	}
	for _, cell := range spec.Cells {
		if cell.ModelDefinition != definition || !strings.Contains(cell.Name, "/text/") && cell.Recipe != projection.Definition.ID {
			t.Fatalf("%s does not name the promoted binding", cell.Name)
		}
	}
	mask, err := runrecord.VerifyGateRun(t.Context(), store, projection.Definition.ID, spec.MaskGate, spec.MaskRun)
	if err != nil {
		t.Fatal(err)
	}
	nativeHidden, err := os.ReadFile(testutil.FixturePath(t, "e4b_vision", "hidden_mask_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	wantMask := fmt.Sprintf("native_hidden_sha256=%x;policies=2;interventions=2;layers=2", sha256.Sum256(nativeHidden))
	if mask.Gate.CodeCommit != spec.CodeCommit || len(mask.Gate.Steps) != 1 || mask.Gate.Steps[0].Name != "image-mask-causality" ||
		mask.Gate.Steps[0].Outcome != runrecord.StepSucceeded || mask.Gate.Steps[0].Evidence != wantMask {
		t.Fatal("E4B mask evidence differs from clean source and native capture")
	}
	oracle, err := os.ReadFile(testutil.FixturePath(t, "e4b_multimodal_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, cell := range spec.Cells {
		if (strings.HasPrefix(cell.Name, "protocol/") || strings.HasPrefix(cell.Name, "resources/")) && cell.OracleSHA256 != fmt.Sprintf("%x", sha256.Sum256(oracle)) {
			t.Fatal("protocol proof differs from the pinned native oracle")
		}
	}
	t.Log("E4B: 18 protocol cells, six dataset-quality cells and six resource cells; exact source/case/metric authorities verified; new model executions=0")
}
