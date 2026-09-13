package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

type imageVideoInventory struct {
	Version  uint16                    `json:"version"`
	Census   artifact.ID               `json:"census"`
	Tasks    []recipe.Task             `json:"tasks"`
	Cells    []imageVideoInventoryCell `json:"cells"`
	Inactive []imageVideoInactive      `json:"inactive"`
}

type imageVideoInventoryCell struct {
	Name        string                `json:"name"`
	Model       artifact.ID           `json:"model"`
	Task        recipe.Task           `json:"task"`
	Recipe      artifact.ID           `json:"recipe"`
	References  []imageVideoReference `json:"references"`
	Limitations []string              `json:"limitations"`
}

type imageVideoReference struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type imageVideoInactive struct {
	Recipe artifact.ID   `json:"recipe"`
	Status recipe.Status `json:"status"`
}

func checkImageVideoCoverage(value imageVideoInventory, census capabilityCensus) error {
	if value.Version != artifact.InitialDocumentVersion || value.Census != census.ID {
		return errors.New("media inventory: census binding differs")
	}
	wantTasks := []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen}
	tasks := slices.Clone(value.Tasks)
	slices.Sort(tasks)
	slices.Sort(wantTasks)
	if !slices.Equal(tasks, wantTasks) {
		return errors.New("media inventory: task scope differs")
	}
	type cellKey struct {
		model, definition artifact.ID
		task              recipe.Task
	}
	want := map[cellKey]bool{}
	for _, model := range census.Models {
		for _, capability := range model.Capabilities {
			if !slices.Contains(tasks, capability.Task) {
				continue
			}
			if !model.Present || capability.Stale != "" {
				return fmt.Errorf("media inventory: unavailable or stale required model %s", model.Model)
			}
			want[cellKey{model.Model, capability.Recipe, capability.Task}] = true
		}
	}
	if len(want) == 0 || len(value.Cells) != len(want) {
		return errors.New("media inventory: required cell coverage differs")
	}
	for _, cell := range value.Cells {
		key := cellKey{cell.Model, cell.Recipe, cell.Task}
		if !want[key] {
			return errors.New("media inventory: duplicate or substituted model/task/recipe")
		}
		delete(want, key)
		if cell.Name == "" || len(cell.References) == 0 || len(cell.Limitations) == 0 {
			return errors.New("media inventory: reference scope or limitations absent")
		}
	}
	return nil
}

func checkImageVideoReference(root string, reference imageVideoReference) error {
	want, err := hex.DecodeString(reference.SHA256)
	if err != nil || len(want) != sha256.Size || reference.Path == "" {
		return errors.New("media inventory: invalid reference identity")
	}
	path := reference.Path
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	hash := sha256.New()
	_, readErr := io.Copy(hash, file)
	if err := errors.Join(readErr, file.Close()); err != nil {
		return err
	}
	if !slices.Equal(hash.Sum(nil), want) {
		return fmt.Errorf("media inventory: reference %s changed", reference.Path)
	}
	return nil
}

func checkImageVideoDefinitions(ctx context.Context, store *overgodb.Store, value imageVideoInventory) error {
	for _, cell := range value.Cells {
		active, found, err := modelrecipe.ActiveRecord(ctx, store, cell.Model, cell.Task)
		if err != nil || !found || active.Definition.ID != cell.Recipe {
			return fmt.Errorf("media inventory: active recipe differs for %s: %v", cell.Name, err)
		}
		if _, err := runrecord.VerifyEvidence(ctx, store, cell.Recipe, active.Event.Evidence); err != nil {
			return err
		}
		if len(active.Definition.Inputs) == 0 || len(active.Definition.Outputs) == 0 {
			return fmt.Errorf("media inventory: input/output contract absent for %s", cell.Name)
		}
		manifest, found, err := store.Manifest(ctx, cell.Model)
		if err != nil || !found {
			return fmt.Errorf("media inventory: component manifest absent for %s: %v", cell.Name, err)
		}
		if err := manifest.Validate(); err != nil {
			return err
		}
		for _, component := range manifest.Components {
			if _, found, err := store.Artifact(ctx, component.Artifact); err != nil || !found {
				return fmt.Errorf("media inventory: component %s absent: %v", component.Artifact, err)
			}
		}
		for _, dependency := range active.Definition.Dependencies {
			if _, found, err := store.Artifact(ctx, dependency.Artifact); err != nil || !found {
				return fmt.Errorf("media inventory: recipe dependency %s absent: %v", dependency.Artifact, err)
			}
		}
	}
	want := map[artifact.ID]recipe.Status{}
	for _, inactive := range value.Inactive {
		if _, duplicate := want[inactive.Recipe]; duplicate || inactive.Status == recipe.StatusActive {
			return errors.New("media inventory: duplicate or active entry in inactive inventory")
		}
		want[inactive.Recipe] = inactive.Status
	}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{recipe.DefinitionDocumentContract()}, Order: overgodb.DocumentOldestFirst,
	}, recipe.ParseDefinition, func(_ overgodb.DocumentView, definition recipe.Definition) error {
		if !slices.Contains(value.Tasks, definition.Task) {
			return nil
		}
		status, found, err := modelrecipe.Status(ctx, store, definition.ID)
		if err != nil || !found {
			return fmt.Errorf("media inventory: lifecycle absent for %s: %v", definition.ID, err)
		}
		if status != recipe.StatusActive {
			if want[definition.ID] != status {
				return fmt.Errorf("media inventory: inactive recipe disposition differs for %s", definition.ID)
			}
			delete(want, definition.ID)
		}
		return nil
	})
	if err != nil {
		return err
	}
	if len(want) != 0 {
		return errors.New("media inventory: inactive recipe absent from store")
	}
	return nil
}

func TestImageVideoInventoryAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": exact media inventory acceptance checks the private snapshot")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: image_video_gen inventory requires its explicit data root")
	}
	var value imageVideoInventory
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_inventory.json"), &value); err != nil {
		t.Fatal(err)
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
	census, err := capabilityCensusCodec.Require(t.Context(), store, value.Census)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkCapabilityCensus(t.Context(), store, census, len(census.Models)); err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoCoverage(value, census); err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoDefinitions(t.Context(), store, value); err != nil {
		t.Fatal(err)
	}
	for _, cell := range value.Cells {
		for _, reference := range cell.References {
			if err := checkImageVideoReference(root, reference); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("%s %s model=%s recipe=%s references=%d; limitations=%v", cell.Name, cell.Task, cell.Model, cell.Recipe, len(cell.References), cell.Limitations)
	}
	for _, name := range []string{"omitted cell", "duplicate cell", "foreign task", "wrong recipe", "wrong census", "missing reference"} {
		t.Run(name, func(t *testing.T) {
			changed := value
			changed.Cells = slices.Clone(value.Cells)
			switch name {
			case "omitted cell":
				changed.Cells = changed.Cells[1:]
			case "duplicate cell":
				changed.Cells[1] = changed.Cells[0]
			case "foreign task":
				changed.Cells[0].Task = recipe.TaskVQA
			case "wrong recipe":
				changed.Cells[0].Recipe = changed.Cells[1].Recipe
			case "wrong census":
				changed.Census = artifact.ID{}
			case "missing reference":
				changed.Cells[0].References = nil
			}
			if err := checkImageVideoCoverage(changed, census); err == nil {
				t.Fatal("invalid media coverage accepted")
			}
		})
	}
	t.Run("changed reference", func(t *testing.T) {
		directory := t.TempDir()
		reference := imageVideoReference{Path: "sample", SHA256: fmt.Sprintf("%x", sha256.Sum256([]byte("original")))}
		if err := os.WriteFile(filepath.Join(directory, reference.Path), []byte("changed"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := checkImageVideoReference(directory, reference); err == nil {
			t.Fatal("changed reference accepted")
		}
	})
	t.Logf("frozen census=%s producer=%s store sequence=%d; cells=%d inactive=%d; no model execution", value.Census, census.ProducerCommit, census.StoreSequence, len(value.Cells), len(value.Inactive))
}
