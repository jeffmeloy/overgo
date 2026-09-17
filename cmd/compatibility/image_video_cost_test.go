package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const imageVideoCostProfileSHA256 = "143b869ce8ab2a3d0967b9f9c89d05f21e26ea36a60201c1479f31a48745f6b4"

type mediaCostPipeline struct {
	Model         artifact.ID   `json:"model"`
	Task          recipe.Task   `json:"task"`
	Recipe        artifact.ID   `json:"recipe"`
	Nodes         []recipe.Node `json:"nodes"`
	RuntimeOwner  string        `json:"runtime_owner"`
	Preparation   string        `json:"preparation"`
	Integration   string        `json:"integration"`
	Decode        string        `json:"decode"`
	Publication   string        `json:"publication"`
	ResourceScope string        `json:"resource_scope"`
}
type mediaCostProfile struct {
	Version   uint16              `json:"version"`
	Census    artifact.ID         `json:"inventory_census"`
	Source    string              `json:"baseline_source"`
	Quality   string              `json:"quality_protocol_sha256"`
	Resources string              `json:"resource_protocol_sha256"`
	Sources   map[string]string   `json:"source_sha256"`
	Pipelines []mediaCostPipeline `json:"pipelines"`
	Wan       struct {
		Profile      artifact.ID                `json:"profile"`
		Geometry     latentvideo.LatentGeometry `json:"geometry"`
		Latent       uint64                     `json:"latent_bytes"`
		Conditioning uint64                     `json:"step_conditioning_bytes"`
		Shared       uint64                     `json:"step_shared_input_bytes"`
		TwoBranch    uint64                     `json:"two_branch_input_bytes"`
		Avoided      uint64                     `json:"predicted_avoided_duplicate_upload_bytes"`
		Readback     uint64                     `json:"predicted_head_readback_bytes"`
		Scope        string                     `json:"scope"`
	} `json:"derived_wan_boundaries"`
	Candidates []struct {
		ID          string   `json:"id"`
		Owners      []string `json:"owners"`
		Change      string   `json:"change"`
		Effect      string   `json:"effect"`
		Cohort      string   `json:"cohort"`
		Limitations string   `json:"limitations"`
	} `json:"candidates"`
}

func checkMediaCostCoverage(cost mediaCostProfile, inventory imageVideoInventory) error {
	if cost.Version != 1 || cost.Census != inventory.Census || cost.Quality != imageVideoProtocolSHA256 || cost.Resources != imageVideoResourceProtocolSHA256 || len(cost.Source) != 40 || len(cost.Pipelines) != len(inventory.Cells) {
		return errors.New("cost profile identity or denominator differs")
	}
	seen := map[artifact.ID]bool{}
	for _, p := range cost.Pipelines {
		if seen[p.Recipe] || !slices.ContainsFunc(inventory.Cells, func(c imageVideoInventoryCell) bool {
			return c.Model == p.Model && c.Task == p.Task && c.Recipe == p.Recipe
		}) {
			return errors.New("cost profile recipe coverage differs")
		}
		seen[p.Recipe] = true
		if p.RuntimeOwner == "" || cost.Sources[p.RuntimeOwner] == "" || p.Preparation == "" || p.Integration == "" || p.Decode == "" || p.Publication == "" || p.ResourceScope == "" || len(p.Nodes) == 0 {
			return errors.New("cost profile omits a stage or source owner")
		}
	}
	seenCandidates := map[string]bool{}
	for _, c := range cost.Candidates {
		if c.ID == "" || seenCandidates[c.ID] || len(c.Owners) == 0 || c.Change == "" || c.Effect == "" || c.Cohort == "" || c.Limitations == "" {
			return errors.New("cost candidate lacks attribution or limitations")
		}
		seenCandidates[c.ID] = true
		for _, owner := range c.Owners {
			if cost.Sources[owner] == "" {
				return errors.New("cost candidate source is unbound")
			}
		}
	}
	if len(seenCandidates) == 0 {
		return errors.New("cost profile has no candidate attribution")
	}
	return nil
}
func checkMediaCostBytes(cost mediaCostProfile, profile latentvideo.Profile, config latentvideo.DenoiserConfig) error {
	request := profile.Generation
	geometry, err := config.CompileLatentGeometry(request.Frames, request.Width, request.Height)
	if err != nil {
		return err
	}
	latent, err := tensor.MustShape(uint64(geometry.Elements())).Bytes(dtype.F32)
	if err != nil {
		return err
	}
	conditioning, err := tensor.MustShape(uint64(media.PairedShiftScaleGateWidth(config.Dim) + config.Dim)).Bytes(dtype.F32)
	if err != nil {
		return err
	}
	w := cost.Wan
	if w.Profile != profile.ID || w.Geometry != geometry || w.Latent != latent || w.Conditioning != conditioning || w.Shared != latent+conditioning || w.TwoBranch != w.Shared*tensor.PairedExtent || w.Avoided != w.Shared*uint64(request.Steps) || w.Readback != latent*tensor.PairedExtent*uint64(request.Steps) || w.Scope == "" {
		return errors.New("cost profile transfer accounting differs from declared shapes")
	}
	return nil
}
func TestImageVideoCostAttributionAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": cost attribution reads the private snapshot and model metadata")
	}
	root := testutil.RepoRoot(t)
	var cost mediaCostProfile
	data, err := os.ReadFile(filepath.Join(root, "docs/image_video_costs.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(data, imageVideoCostProfileSHA256); err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_costs.json"), &cost); err != nil {
		t.Fatal(err)
	}
	var inventory imageVideoInventory
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_inventory.json"), &inventory); err != nil {
		t.Fatal(err)
	}
	if err := checkMediaCostCoverage(cost, inventory); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, p := range cost.Pipelines {
		active, found, err := modelrecipe.ActiveRecord(t.Context(), store, p.Model, p.Task)
		if err != nil || !found {
			t.Fatalf("recipe unavailable: %v", err)
		}
		actual, _ := json.Marshal(active.Definition.Nodes)
		retained, _ := json.Marshal(p.Nodes)
		if active.Definition.ID != p.Recipe || !bytes.Equal(actual, retained) {
			t.Fatal("compiled stages differ from cost attribution")
		}
	}
	for path, digest := range cost.Sources {
		var output bytes.Buffer
		receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{Path: "git", Args: []string{"show", cost.Source + ":" + path}, Dir: root, Env: gitauthority.ReaderEnvironment(), Stdout: &output})
		if err != nil || receipt.ExitCode != 0 || fmt.Sprintf("%x", sha256.Sum256(output.Bytes())) != digest {
			t.Fatalf("source identity differs: %s: %v", path, err)
		}
	}
	directory := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	profile, err := latentvideo.ResolveProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	config, err := latentvideo.LoadDenoiserConfig(directory, profile.Policy)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaCostBytes(cost, profile, config); err != nil {
		t.Fatal(err)
	}
	for _, defect := range []string{"missing stage", "duplicate cell", "omitted cell", "changed quality"} {
		bad := cost
		bad.Pipelines = slices.Clone(cost.Pipelines)
		switch defect {
		case "missing stage":
			bad.Pipelines[0].Decode = ""
		case "duplicate cell":
			bad.Pipelines[1] = bad.Pipelines[0]
		case "omitted cell":
			bad.Pipelines = bad.Pipelines[1:]
		case "changed quality":
			bad.Quality = "changed"
		}
		if checkMediaCostCoverage(bad, inventory) == nil {
			t.Fatalf("accepted %s", defect)
		}
	}
	bad := cost
	bad.Wan.Avoided++
	if checkMediaCostBytes(bad, profile, config) == nil {
		t.Fatal("accepted incorrect transfer bytes")
	}
	var changed map[string]any
	if err := json.Unmarshal(data, &changed); err != nil {
		t.Fatal(err)
	}
	changed["boundaries"] = nil
	changedData, err := json.Marshal(changed)
	if err != nil {
		t.Fatal(err)
	}
	if checkMediaProtocolIdentity(changedData, imageVideoCostProfileSHA256) == nil {
		t.Fatal("accepted omitted lifetime and copy attribution")
	}
	t.Logf("%d actual pipelines; %d source identities; predicted duplicate step upload=%d bytes, remaining head readback=%d bytes; metadata only, no model execution", len(cost.Pipelines), len(cost.Sources), cost.Wan.Avoided, cost.Wan.Readback)
}
