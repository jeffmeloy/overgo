package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

const imageVideoCompositionSHA256 = "6974a8c383d7a64cc0a02d79b41d2303ccea7760e0fa097572e013209e26461a"

type mediaCompositionBundle struct {
	Version             uint16        `json:"version"`
	Source              string        `json:"source_base"`
	Protocol            artifact.ID   `json:"protocol"`
	Acquisition         artifact.ID   `json:"native_acquisition"`
	Harness             artifact.ID   `json:"harness"`
	NativeTest          artifact.ID   `json:"native_test"`
	OwnerTest           artifact.ID   `json:"owner_test"`
	ScopeEvidence       artifact.ID   `json:"scope_evidence"`
	Failed              []artifact.ID `json:"failed_attempts"`
	RetainedRun         artifact.ID   `json:"retained_run"`
	RetainedAcquisition artifact.ID   `json:"retained_acquisition"`
	Cells               []struct {
		Model       artifact.ID `json:"model"`
		Recipe      artifact.ID `json:"recipe"`
		Task        recipe.Task `json:"task"`
		Disposition string      `json:"disposition"`
	} `json:"cells"`
	Components []struct {
		Node      recipe.NodeID `json:"node"`
		Directory string        `json:"directory"`
		Model     artifact.ID   `json:"model"`
		Members   int           `json:"members"`
	} `json:"components"`
	Scope string `json:"scope"`
}

func TestImageVideoCompositionAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": retained composition evidence")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: explicit media data root required")
	}
	var bundle mediaCompositionBundle
	path := filepath.Join(root, "docs/image_video_composition.json")
	if err := jsonfile.DecodeStrict(path, &bundle); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoCompositionSHA256); err != nil {
		t.Fatal(err)
	}
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Source == "" || bundle.Scope == "" {
		t.Fatal("incomplete composition bundle")
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
	read := func(id artifact.ID) []byte {
		t.Helper()
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatal("missing composition evidence", id, err)
		}
		return content.Data
	}
	for _, id := range []artifact.ID{bundle.NativeTest, bundle.OwnerTest} {
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if report.PassedTests == 0 {
			t.Fatal("empty composition test")
		}
	}
	if len(bundle.Failed) == 0 {
		t.Fatal("failed verification receipt missing")
	}
	for _, id := range bundle.Failed {
		report, err := testevidence.GoTestJSONReport(string(read(id)))
		if err != nil || len(report.Failed) == 0 {
			t.Fatal("failed attempt is not retained as a failure", err)
		}
	}
	var inventory imageVideoInventory
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_inventory.json"), &inventory); err != nil {
		t.Fatal(err)
	}
	if len(bundle.Cells) != len(inventory.Cells) {
		t.Fatal("composition scope omits inventory cells")
	}
	var scope []struct {
		Model        artifact.ID         `json:"model"`
		Recipe       artifact.ID         `json:"recipe"`
		Dependencies []recipe.Dependency `json:"dependencies"`
	}
	if err := json.Unmarshal(read(bundle.ScopeEvidence), &scope); err != nil {
		t.Fatal(err)
	}
	if len(scope) != len(inventory.Cells) {
		t.Fatal("scope evidence omits recipes")
	}
	var composed recipe.Definition
	for index, cell := range inventory.Cells {
		entry := bundle.Cells[index]
		if entry.Model != cell.Model || entry.Recipe != cell.Recipe || entry.Task != cell.Task {
			t.Fatal("scope differs from inventory")
		}
		active, found, err := modelrecipe.ActiveRecord(t.Context(), store, cell.Model, cell.Task)
		if err != nil || !found || active.Definition.ID != cell.Recipe {
			t.Fatal("active recipe changed", err)
		}
		definition, err := recipe.RequireDefinition(t.Context(), store, cell.Recipe)
		if err != nil {
			t.Fatal(err)
		}
		if scope[index].Model != cell.Model || scope[index].Recipe != cell.Recipe || !slices.Equal(scope[index].Dependencies, definition.Dependencies) {
			t.Fatal("retained scope differs from actual dependencies")
		}
		modelCount := 0
		for _, dependency := range definition.Dependencies {
			switch dependency.Role {
			case recipe.DependencyModel:
				modelCount++
			case recipe.DependencyProfile, recipe.DependencyFlowProfile:
			default:
				t.Fatal("uncovered composition dependency", dependency)
			}
		}
		if modelCount == 1 {
			if entry.Disposition != "single-model-native" {
				t.Fatal("incorrect single-model disposition")
			}
			for _, node := range definition.Nodes {
				if node.ModelSlot != 0 {
					t.Fatal("unexpected single-model slot")
				}
			}
		} else {
			if entry.Disposition != "native-component-chain" || composed.ID.Valid() {
				t.Fatal("uncovered or duplicate composition")
			}
			composed = definition
		}
	}
	if !composed.ID.Valid() || len(bundle.Components) == 0 {
		t.Fatal("required native component chain missing")
	}
	parent, found, err := store.Manifest(t.Context(), composed.Model)
	if err != nil || !found {
		t.Fatal("parent manifest missing", err)
	}
	if err := parent.Validate(); err != nil {
		t.Fatal(err)
	}
	var bindings []modelrecipe.ComponentBinding
	for _, component := range bundle.Components {
		var members []artifact.Component
		for _, member := range parent.Components {
			if strings.HasPrefix(member.Name, component.Directory+"/") {
				members = append(members, member)
			}
		}
		expected, err := artifact.NewManifest(artifact.KindModel, members)
		if err != nil {
			t.Fatal(err)
		}
		child, found, err := store.Manifest(t.Context(), component.Model)
		if err != nil || !found {
			t.Fatal("child manifest missing", err)
		}
		if err := child.Validate(); err != nil {
			t.Fatal(err)
		}
		if len(members) == 0 || len(members) != component.Members || child.ID != expected.ID || !slices.Equal(child.Components, expected.Components) {
			t.Fatal("component files, identities, roles or ordinals differ from native parent")
		}
		bindings = append(bindings, modelrecipe.ComponentBinding{Node: component.Node, Model: component.Model})
	}
	profile, ok := composed.PrimaryDependency(recipe.DependencyProfile)
	if !ok {
		t.Fatal("profile absent")
	}
	rebuilt, err := modelrecipe.GenerationDefinitionWithComponents(modelrecipe.ModuleLatentImagePrepare, composed.Model, profile, bindings)
	if err != nil || rebuilt.ID != composed.ID {
		t.Fatal("typed ports, stage binding or profile differ from native chain", err)
	}
	program, err := modelrecipe.CompileCapability(composed)
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	stages := program.Stages()
	if len(stages) != len(bindings) {
		t.Fatal("unexpected conversion stage")
	}
	for index, stage := range stages {
		if stage.Node.ID != bindings[index].Node || workflow.ModuleModel(stage.Module.ID, composed.Model) != bindings[index].Model {
			t.Fatal("compiled execution order or component selection changed")
		}
	}
	var protocol struct {
		Source              string      `json:"source_base"`
		Model               artifact.ID `json:"model"`
		Recipe              artifact.ID `json:"recipe"`
		Input               artifact.ID `json:"input"`
		Output              artifact.ID `json:"output"`
		RetainedRun         artifact.ID `json:"retained_run"`
		RetainedAcquisition artifact.ID `json:"retained_acquisition"`
		Quality             string      `json:"quality_protocol_sha256"`
		Harness             string      `json:"harness_sha256"`
		Acceptance          string      `json:"acceptance"`
	}
	protocolRaw := read(bundle.Protocol)
	if err := json.Unmarshal(protocolRaw, &protocol); err != nil {
		t.Fatal(err)
	}
	if protocol.Source != bundle.Source || protocol.Model != composed.Model || protocol.Recipe != composed.ID || protocol.Quality != imageVideoProtocolSHA256 || protocol.RetainedRun != bundle.RetainedRun || protocol.RetainedAcquisition != bundle.RetainedAcquisition || protocol.Acceptance == "" || protocol.Harness != fmt.Sprintf("%x", sha256.Sum256(read(bundle.Harness))) {
		t.Fatal("native comparison protocol changed")
	}
	retained, err := runrecord.RequireExactRun(t.Context(), store, bundle.RetainedRun)
	if err != nil {
		t.Fatal(err)
	}
	if retained.Recipe != composed.ID || retained.Outcome != runrecord.OutcomeSucceeded || !slices.Equal(retained.Inputs, []artifact.ID{protocol.Input}) || !slices.Equal(retained.Outputs, []artifact.ID{protocol.Output}) {
		t.Fatal("native comparison uses unrelated input or output")
	}
	var prior struct {
		Source   string          `json:"source_base"`
		Run      artifact.ID     `json:"validation_run"`
		Output   artifact.ID     `json:"output"`
		Resolved json.RawMessage `json:"resolved_request"`
	}
	if err := json.Unmarshal(read(bundle.RetainedAcquisition), &prior); err != nil {
		t.Fatal(err)
	}
	if prior.Run != retained.ID || prior.Source != retained.CodeCommit || prior.Output != protocol.Output {
		t.Fatal("retained acquisition binding differs")
	}
	var native struct {
		Source      string                `json:"source_base"`
		Protocol    string                `json:"protocol_sha256"`
		Model       artifact.ID           `json:"model"`
		Recipe      artifact.ID           `json:"recipe"`
		Input       artifact.ID           `json:"input"`
		Output      artifact.ID           `json:"output"`
		Exact       bool                  `json:"exact_png_match"`
		Width       int                   `json:"width"`
		Height      int                   `json:"height"`
		Min         float64               `json:"minimum"`
		Max         float64               `json:"maximum"`
		Resolved    json.RawMessage       `json:"resolved_request"`
		Environment runrecord.Environment `json:"environment"`
		Load        uint64                `json:"load_ns"`
		Request     uint64                `json:"request_ns"`
		Close       uint64                `json:"close_ns"`
		Total       uint64                `json:"load_through_close_ns"`
		Allocated   uint64                `json:"allocated_bytes"`
		Peak        uint64                `json:"peak_host_bytes"`
		Before      driver.ExecutionStats `json:"load_counters"`
		After       driver.ExecutionStats `json:"after_counters"`
		Memory      driver.MemoryStats    `json:"after_memory"`
		Closed      driver.MemoryStats    `json:"closed_memory"`
	}
	if err := json.Unmarshal(read(bundle.Acquisition), &native); err != nil {
		t.Fatal(err)
	}
	if native.Source != bundle.Source || native.Protocol != fmt.Sprintf("%x", sha256.Sum256(protocolRaw)) || native.Model != protocol.Model || native.Recipe != protocol.Recipe || native.Input != protocol.Input || native.Output != protocol.Output || !native.Exact || string(native.Resolved) != string(prior.Resolved) {
		t.Fatal("native acquisition differs from fixed composed request")
	}
	var request struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	}
	if err := json.Unmarshal(read(protocol.Input), &request); err != nil {
		t.Fatal(err)
	}
	if native.Width != request.Width || native.Height != request.Height || math.IsNaN(native.Min) || math.IsInf(native.Min, 0) || math.IsNaN(native.Max) || math.IsInf(native.Max, 0) || native.Min > native.Max || native.Load == 0 || native.Request == 0 || native.Close == 0 || native.Total < native.Load+native.Request+native.Close || native.Allocated == 0 || native.Peak == 0 || native.Memory.PeakBytes == 0 || native.Closed.CurrentBytes != 0 || native.After.KernelLaunches+native.After.GraphLaunches <= native.Before.KernelLaunches+native.Before.GraphLaunches {
		t.Fatal("native measurements or cleanup incomplete")
	}
	if _, err := runrecord.NewEnvironment(native.Environment); err != nil {
		t.Fatal(err)
	}
	output, found, err := artifact.ReadContent(t.Context(), store, protocol.Output)
	if err != nil || !found {
		t.Fatal("matched output missing", err)
	}
	_, pixels, ok := sampleFile(output)
	if !ok || len(pixels) == 0 {
		t.Fatal("matched PNG is not exportable")
	}
	var merged mediaMergedEvidence
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{prior.Source, bundle.Source, "HEAD", ""} {
		if err := checkMediaRuntimeAtRevision(root, revision, merged.RuntimePaths, merged.RuntimeSHA256); err != nil {
			t.Fatal(err)
		}
	}
}

func TestImageVideoCompositionRejectsChangedEvidence(t *testing.T) {
	path := filepath.Join(testutil.RepoRoot(t), "docs/image_video_composition.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"cells", "components", "retained_run", "native_acquisition"} {
		t.Run(field, func(t *testing.T) {
			var changed map[string]any
			if err := json.Unmarshal(raw, &changed); err != nil {
				t.Fatal(err)
			}
			delete(changed, field)
			data, err := json.Marshal(changed)
			if err != nil {
				t.Fatal(err)
			}
			if err := checkMediaProtocolIdentity(data, imageVideoCompositionSHA256); err == nil {
				t.Fatal("composition evidence omission accepted")
			}
		})
	}
}
