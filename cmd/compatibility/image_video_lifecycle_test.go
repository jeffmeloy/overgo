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
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const imageVideoLifecycleSHA256 = "3c61e885b7c979140ffdcb8956e26836d9223cfd4dc3779ef8d137cc1e718159"

type mediaLifecycleChange struct {
	Before string `json:"before_sha256"`
	After  string `json:"after_sha256"`
	Reason string `json:"reason"`
}

type mediaLifecycleCheck struct {
	Name     string      `json:"name"`
	Evidence artifact.ID `json:"evidence"`
	Protocol artifact.ID `json:"protocol"`
	Command  string      `json:"command"`
	Required []string    `json:"required_tests"`
	Scope    string      `json:"scope"`
}

type mediaLifecycleBundle struct {
	SurfaceReview artifact.ID                     `json:"source_size_review"`
	Version       uint16                          `json:"version"`
	Source        string                          `json:"source_base"`
	Protocol      artifact.ID                     `json:"protocol"`
	Patch         artifact.ID                     `json:"source_patch"`
	Changes       map[string]mediaLifecycleChange `json:"source_changes"`
	Checks        []mediaLifecycleCheck           `json:"checks"`
	Failed        []artifact.ID                   `json:"failed_attempts"`
	Harnesses     map[string]artifact.ID          `json:"test_sources"`
	Scope         string                          `json:"scope"`
}

func readMediaLifecycleBundle(root string) (mediaLifecycleBundle, error) {
	var bundle mediaLifecycleBundle
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_lifecycle.json"))
	if err != nil {
		return bundle, err
	}
	if err := checkMediaProtocolIdentity(raw, imageVideoLifecycleSHA256); err != nil {
		return bundle, err
	}
	err = json.Unmarshal(raw, &bundle)
	return bundle, err
}

func mediaLifecycleProductionPath(path string) bool {
	return !strings.HasSuffix(path, "_test.go") && (strings.HasSuffix(path, ".go") || strings.HasSuffix(path, ".cu") || strings.HasSuffix(path, ".cuh") || path == "kernels/manifest.json" || path == "go.mod" || path == "go.sum")
}

// Reconcile only the complete, frozen lifecycle patch. Prior acquisitions retain
// their original revisions and numerical scope; later source edits need new proof.
func checkMediaLifecycleSource(root, revision string, paths []string) (string, error) {
	base, prior := checkMediaLifecyclePatch(root, revision, paths)
	if prior == nil {
		return base, nil
	}
	timingBase, err := checkMediaTimingSource(root, revision, paths)
	if err != nil {
		return "", errors.Join(prior, err)
	}
	return checkMediaLifecyclePatch(root, timingBase, paths)
}

func checkMediaLifecyclePatch(root, revision string, paths []string) (string, error) {
	bundle, err := readMediaLifecycleBundle(root)
	if err != nil {
		return "", err
	}
	if !gitauthority.ValidObjectID(bundle.Source) || len(bundle.Changes) == 0 {
		return "", errors.New("missing lifecycle source closure")
	}
	git := func(args ...string) ([]byte, error) {
		command := exec.Command("git", args...)
		command.Dir = root
		return command.Output()
	}
	changed, err := mediaRuntimeChanges(root, bundle.Source, revision, paths)
	if err != nil {
		return "", err
	}
	for _, path := range changed {
		if _, found := bundle.Changes[path]; !found {
			return "", fmt.Errorf("unreconciled lifecycle source: %s", path)
		}
	}
	for path, change := range bundle.Changes {
		included := false
		for _, scope := range paths {
			included = included || path == scope || strings.HasPrefix(path, strings.TrimSuffix(scope, "/")+"/")
		}
		if !included {
			continue
		}
		before, err := git("show", bundle.Source+":"+path)
		if change.Before == "" {
			if err == nil {
				return "", fmt.Errorf("lifecycle addition already exists: %s", path)
			}
		} else if err != nil || fmt.Sprintf("%x", sha256.Sum256(before)) != change.Before {
			return "", fmt.Errorf("lifecycle prior source differs: %s", path)
		}
		var after []byte
		if revision == "" {
			after, err = os.ReadFile(filepath.Join(root, path))
			after = []byte(strings.ReplaceAll(string(after), "\r\n", "\n"))
		} else {
			after, err = git("show", revision+":"+path)
		}
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(after)) != change.After {
			return "", fmt.Errorf("lifecycle current source differs: %s", path)
		}
	}
	return bundle.Source, nil
}

func TestImageVideoLifecycleAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": retained media lifecycle evidence")
	}
	root := testutil.RepoRoot(t)
	bundle, err := readMediaLifecycleBundle(root)
	if err != nil {
		t.Fatal(err)
	}
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Scope == "" || len(bundle.Checks) == 0 || len(bundle.Failed) == 0 {
		t.Fatal("incomplete lifecycle evidence")
	}
	var merged mediaMergedEvidence
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	paths := append(slices.Clone(merged.RuntimePaths), "internal/workflowruntime", "cmd/dit-train-probe")
	if _, err := checkMediaLifecycleSource(root, "", paths); err != nil {
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
	read := func(id artifact.ID) []byte {
		t.Helper()
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatal("missing lifecycle evidence", id, err)
		}
		return content.Data
	}
	var protocol struct {
		Source     string   `json:"source_base"`
		Acceptance []string `json:"acceptance"`
	}
	if err := json.Unmarshal(read(bundle.Protocol), &protocol); err != nil || protocol.Source != bundle.Source || len(protocol.Acceptance) == 0 {
		t.Fatal("lifecycle protocol differs", err)
	}
	var surfaceReview struct{ Source, Reason string }
	if err := json.Unmarshal(read(bundle.SurfaceReview), &surfaceReview); err != nil || surfaceReview.Source != bundle.Source || surfaceReview.Reason == "" {
		t.Fatal("missing reviewed source-size rationale", err)
	}
	var patch map[string]struct{ Before, After string }
	if err := json.Unmarshal(read(bundle.Patch), &patch); err != nil || len(patch) != len(bundle.Changes) {
		t.Fatal("incomplete lifecycle source patch", err)
	}
	for path, change := range bundle.Changes {
		entry, found := patch[path]
		if !found || change.Reason == "" || change.After != fmt.Sprintf("%x", sha256.Sum256([]byte(entry.After))) {
			t.Fatal("lifecycle patch differs", path)
		}
		if (change.Before == "" && entry.Before != "") || (change.Before != "" && change.Before != fmt.Sprintf("%x", sha256.Sum256([]byte(entry.Before)))) {
			t.Fatal("lifecycle prior patch differs", path)
		}
	}
	covered := map[string]bool{}
	var assertions map[string]map[string][]string
	if err := jsonfile.DecodeStrict(filepath.Join(root, mediaAssertionsPath), &assertions); err != nil {
		t.Fatal(err)
	}
	for _, check := range bundle.Checks {
		if check.Name == "" || check.Command == "" || check.Scope == "" || len(check.Required) == 0 {
			t.Fatal("incomplete lifecycle check", check.Name)
		}
		raw := read(check.Evidence)
		requireMediaTestReceipt(t, check.Name, raw, check.Required, assertions)
		var acquisition struct {
			Source string            `json:"source_base"`
			Files  map[string]string `json:"production_source_sha256"`
		}
		if err := json.Unmarshal(read(check.Protocol), &acquisition); err != nil || acquisition.Source != bundle.Source {
			t.Fatal("acquisition source protocol differs", check.Name, err)
		}
		for path, hash := range acquisition.Files {
			if change, found := bundle.Changes[path]; found && change.After == hash {
				covered[path] = true
			}
		}
	}
	for path := range bundle.Changes {
		if !covered[path] {
			t.Fatal("final source lacks a passing acquisition identity", path)
		}
	}
	for _, failure := range bundle.Failed {
		report, err := testevidence.GoTestJSONReport(string(read(failure)))
		if err == nil && testevidence.RequireComplete(report) == nil {
			t.Fatal("failed attempt was relabeled successful", failure)
		}
	}
	if len(bundle.Harnesses) == 0 {
		t.Fatal("missing lifecycle test sources")
	}
	for digest, id := range bundle.Harnesses {
		if fmt.Sprintf("%x", sha256.Sum256(read(id))) != digest {
			t.Fatal("lifecycle test source identity differs", id)
		}
	}
	if err := checkMediaRuntimeAtRevision(root, "", merged.RuntimePaths, merged.RuntimeSHA256); err != nil {
		t.Fatal(err)
	}
}

func TestImageVideoLifecycleRejectsChangedEvidence(t *testing.T) {
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_lifecycle.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"source_base", "source_changes", "source_patch", "protocol", "checks", "failed_attempts", "test_sources", "source_size_review"} {
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
			if err := checkMediaProtocolIdentity(data, imageVideoLifecycleSHA256); err == nil {
				t.Fatal("accepted missing lifecycle evidence", field)
			}
		})
	}
}
