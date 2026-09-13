package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const imageVideoMergedSHA256 = "f6073ad6651b6c1cb705cf249ac9b0b69e8e8f110a60fd1922ba27eab85cb373"

type mediaMergedEvidence struct {
	Version       uint16             `json:"version"`
	Local         string             `json:"local_parent"`
	Incoming      string             `json:"incoming_parent"`
	Merged        string             `json:"merged_commit"`
	Scope         string             `json:"scope"`
	RuntimePaths  []string           `json:"runtime_paths"`
	RuntimeSHA256 string             `json:"runtime_sha256"`
	Inventory     artifact.ID        `json:"inventory_census"`
	Checks        []mediaSharedCheck `json:"checks"`
}

func mediaRuntimeIdentity(root, revision string, paths []string) (string, error) {
	if len(paths) == 0 {
		return "", errors.New("missing runtime scope")
	}
	command := exec.Command("git", append([]string{"ls-tree", "-r", revision, "--"}, paths...)...)
	command.Dir = root
	raw, err := command.Output()
	if err != nil {
		return "", err
	}
	var production []string
	for line := range strings.SplitSeq(string(raw), "\n") {
		_, path, found := strings.Cut(line, "\t")
		if !found || strings.HasSuffix(path, "_test.go") {
			continue
		}
		if strings.HasSuffix(path, ".go") || strings.HasSuffix(path, ".cu") || strings.HasSuffix(path, ".cuh") || path == "kernels/manifest.json" || path == "go.mod" || path == "go.sum" {
			production = append(production, line)
		}
	}
	if len(production) == 0 {
		return "", errors.New("empty runtime source identity")
	}
	return fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(production, "\n")))), nil
}

func compareMediaRuntimeIdentity(expected, actual string) error {
	if expected == "" || actual != expected {
		return errors.New("retained generation source scope changed")
	}
	return nil
}

// A later correction does not invalidate unchanged image or Wan acquisitions.
// Permit only the recorded LiveEdit encoder-range correction: every other
// production file and every other byte of the corrected file must still match.
// An empty revision checks the working tree before gate publication.
func checkMediaRuntimeAtRevision(root, revision string, paths []string, expected string) error {
	if revision != "" {
		actual, err := mediaRuntimeIdentity(root, revision, paths)
		if err != nil {
			return err
		}
		if err := compareMediaRuntimeIdentity(expected, actual); err == nil {
			return nil
		}
	}
	var bundle struct {
		Correction struct {
			Source string `json:"source_base"`
			Path   string `json:"path"`
			Before string `json:"before_git_sha256"`
			After  string `json:"after_git_sha256"`
		} `json:"pixel_range_correction"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_conditioned_videos.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &bundle); err != nil {
		return err
	}
	fix := bundle.Correction
	if fix.Source == "" || fix.Path != "internal/latentvideo/production_cuda_windows.go" || fix.Before == "" || fix.After == "" {
		return errors.New("missing exact LiveEdit correction identity")
	}
	base, err := mediaRuntimeIdentity(root, fix.Source, paths)
	if err != nil {
		return err
	}
	if err := compareMediaRuntimeIdentity(expected, base); err != nil {
		return err
	}
	git := func(args ...string) ([]byte, error) {
		command := exec.Command("git", args...)
		command.Dir = root
		return command.Output()
	}
	before, err := git("show", fix.Source+":"+fix.Path)
	if err != nil {
		return err
	}
	var after []byte
	if revision == "" {
		after, err = os.ReadFile(filepath.Join(root, fix.Path))
		after = []byte(strings.ReplaceAll(string(after), "\r\n", "\n"))
	} else {
		after, err = git("show", revision+":"+fix.Path)
	}
	if err != nil {
		return err
	}
	old := "func (r *LiveEditRuntime) Generate(ctx context.Context, request ReferenceEditRequest) (EncodedVideo, error) {\n\tsink, err := NewGIFEncoder(r.profile.SampleFPS, UnitPixels)"
	want := strings.Replace(string(before), old, strings.Replace(old, "UnitPixels)", "SignedUnitPixels)", 1), 1)
	if strings.Count(string(before), old) != 1 || string(after) != want || fmt.Sprintf("%x", sha256.Sum256(before)) != fix.Before || fmt.Sprintf("%x", sha256.Sum256(after)) != fix.After {
		return errors.New("source change exceeds the declared LiveEdit pixel-range correction")
	}
	args := []string{"diff", "--name-only", fix.Source}
	if revision != "" {
		args = append(args, revision)
	}
	args = append(args, "--")
	changed, err := git(append(args, paths...)...)
	if err != nil {
		return err
	}
	for path := range strings.SplitSeq(strings.TrimSpace(string(changed)), "\n") {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		if strings.HasSuffix(path, ".go") || strings.HasSuffix(path, ".cu") || strings.HasSuffix(path, ".cuh") || path == "kernels/manifest.json" || path == "go.mod" || path == "go.sum" {
			if path != fix.Path {
				return fmt.Errorf("unreconciled generation source change: %s", path)
			}
		}
	}
	return nil
}

func TestImageVideoMergedCapabilitiesAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": merged media evidence requires the private store")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: explicit image/video data root required")
	}
	path := filepath.Join(root, "docs/image_video_merged.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoMergedSHA256); err != nil {
		t.Fatal(err)
	}
	var value mediaMergedEvidence
	if err := jsonfile.DecodeStrict(path, &value); err != nil {
		t.Fatal(err)
	}
	if value.Version != artifact.InitialDocumentVersion || !gitauthority.ValidObjectID(value.Local) || !gitauthority.ValidObjectID(value.Incoming) || !gitauthority.ValidObjectID(value.Merged) || value.Scope == "" || len(value.Checks) == 0 {
		t.Fatal("incomplete merged media evidence")
	}
	for _, revision := range []string{value.Local, value.Merged, "HEAD"} {
		if err := checkMediaRuntimeAtRevision(root, revision, value.RuntimePaths, value.RuntimeSHA256); err != nil {
			t.Fatalf("%s: %v", revision, err)
		}
	}
	if err := compareMediaRuntimeIdentity(value.RuntimeSHA256, ""); err == nil {
		t.Fatal("accepted missing runtime identity")
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
	for _, check := range value.Checks {
		content, found, err := artifact.ReadContent(t.Context(), store, check.Evidence)
		if err != nil || !found {
			t.Fatalf("missing %s: %v", check.Name, err)
		}
		report, err := testevidence.GoTestJSONReport(string(content.Data))
		if err != nil {
			t.Fatal(err)
		}
		if err := testevidence.RequireComplete(report); err != nil {
			t.Fatal(err)
		}
		if check.Name == "" || check.Command == "" || report.PassedPackages == 0 || report.PassedTests == 0 {
			t.Fatal("empty merged-source check")
		}
	}
	var inventory imageVideoInventory
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_inventory.json"), &inventory); err != nil {
		t.Fatal(err)
	}
	if inventory.Census != value.Inventory {
		t.Fatal("changed frozen media denominator")
	}
	census, err := capabilityCensusCodec.Require(t.Context(), store, inventory.Census)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoCoverage(inventory, census); err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoDefinitions(t.Context(), store, inventory); err != nil {
		t.Fatal(err)
	}
	scope, err := resolveMediaReportScope("image-video")
	if err != nil {
		t.Fatal(err)
	}
	entries, truncated, err := discovery.CapabilityCatalogForTasks(t.Context(), store, mediaCatalogLimit, discovery.LoadMemo(t.Context(), store), scope.Tasks...)
	if err != nil || truncated {
		t.Fatalf("current media catalog incomplete: %v", err)
	}
	if _, err := loadMediaSampleSelection(t.Context(), root, store, scope, entries); err != nil {
		t.Fatal(err)
	}
	cells := 0
	for _, entry := range entries {
		cells += len(entry.Capabilities)
	}
	if cells != len(inventory.Cells) {
		t.Fatal("current active media coverage changed")
	}
	bad := inventory
	bad.Cells = slices.Clone(inventory.Cells[1:])
	if err := checkImageVideoCoverage(bad, census); err == nil {
		t.Fatal("accepted omitted media cell")
	}
	t.Logf("master %s merged at %s; %d active media cells retain their recipe/input bindings and unchanged generation source scope", value.Incoming, value.Merged, cells)
}
