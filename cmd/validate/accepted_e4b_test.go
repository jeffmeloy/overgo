package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestAcceptedE4BModalitiesNameTheServedDefinition reads the frozen evidence selection only. It never launches
// inference or promotes a partial or fixture-only model result. The protocol and resource cells and the mask name the
// definition E4B serves and its promoted projection, re-run on it (e4b-tensor-cores); the dataset-quality cells name the
// definition and commit their external adapter measured, the tensor-core prompt attention between the two held to the
// F32 paths by TestPromptTensorCoresHoldHeldOut.
func TestAcceptedE4BModalitiesNameTheServedDefinition(t *testing.T) {
	t.Parallel()
	testskip.Short(t)
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT for canonical E4B evidence acceptance")
	}
	root := testutil.RepoRoot(t)
	var spec modelValidationSpecification
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(root, "docs", "verification", "e4b-validation.json"), &spec))
	if spec.Model.String() != "model:sha256:fb09299dd00edd7ffdcf8cb48e475d2a9c9e30a22c51f79f6d4d793e983c557b" ||
		spec.Projector.String() != "projector:sha256:fd74ad68955dc65da2ccd2498b6ae39b05685120d2cc9ed1bd3c41f4c9933686" {
		t.Fatal("E4B model/projector identity differs")
	}
	roots, err := dataroot.Resolve(root)
	testcheck.NoError(t, err)
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	testcheck.NoError(t, err)
	defer store.Close()
	testcheck.NoError(t, checkModelValidation(t.Context(), store, spec, e4bValidationCells()))
	served, active, err := modelrecipe.ActiveRecord(t.Context(), store, spec.Model, recipe.TaskInference)
	if err != nil || !active {
		t.Fatalf("E4B's inference binding is not promoted: %v", err)
	}
	definition, bound := served.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !bound {
		t.Fatal("E4B inference definition is absent")
	}
	// The dataset-quality cells' adapter ran under the inference recipe the validation first activated.
	measuredID, err := artifact.ParseID("recipe:sha256:d6ba6736fff6cba4ba0fec3b89c996a42105d6643f5e61b64370343a1915a39e")
	testcheck.NoError(t, err)
	measured, err := recipe.RequireDefinition(t.Context(), store, measuredID)
	testcheck.NoError(t, err)
	measuredDefinition, bound := measured.PrimaryDependency(recipe.DependencyDefinition)
	if measured.Model != spec.Model || measured.Task != recipe.TaskInference || !bound {
		t.Fatal("the quality cells' inference recipe is not E4B's")
	}
	projection, active, err := modelrecipe.ActiveRecord(t.Context(), store, spec.Model, recipe.TaskProjection)
	if err != nil || !active {
		t.Fatalf("E4B complete projection binding is not promoted: %v", err)
	}
	for _, cell := range spec.Cells {
		want, commit := definition, spec.CodeCommit
		if strings.HasPrefix(cell.Name, "quality/") {
			want, commit = measuredDefinition, cell.CodeCommit
		}
		if cell.ModelDefinition != want || commit == "" || !strings.Contains(cell.Name, "/text/") && cell.Recipe != projection.Definition.ID {
			t.Fatalf("%s does not name the promoted binding", cell.Name)
		}
	}
	mask, err := runrecord.VerifyGateRun(t.Context(), store, projection.Definition.ID, spec.MaskGate, spec.MaskRun)
	testcheck.NoError(t, err)
	nativeHidden, err := os.ReadFile(testutil.FixturePath(t, "e4b_vision", "hidden_mask_golden.json"))
	testcheck.NoError(t, err)
	wantMask := fmt.Sprintf("native_hidden_sha256=%x;policies=2;interventions=2;layers=2", sha256.Sum256(nativeHidden))
	if mask.Gate.CodeCommit != spec.CodeCommit || len(mask.Gate.Steps) != 1 || mask.Gate.Steps[0].Name != "image-mask-causality" ||
		mask.Gate.Steps[0].Outcome != runrecord.StepSucceeded || mask.Gate.Steps[0].Evidence != wantMask {
		t.Fatal("E4B mask evidence differs from clean source and native capture")
	}
	oracle, err := os.ReadFile(testutil.FixturePath(t, "e4b_multimodal_golden.json"))
	testcheck.NoError(t, err)
	for _, cell := range spec.Cells {
		if (strings.HasPrefix(cell.Name, "protocol/") || strings.HasPrefix(cell.Name, "resources/")) && cell.OracleSHA256 != fmt.Sprintf("%x", sha256.Sum256(oracle)) {
			t.Fatal("protocol proof differs from the pinned native oracle")
		}
	}
	t.Log("E4B: 18 protocol cells, six dataset-quality cells and six resource cells; exact source/case/metric authorities verified; new model executions=0")
}
