package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const imageVideoLifecycleSHA256 = "3c61e885b7c979140ffdcb8956e26836d9223cfd4dc3779ef8d137cc1e718159"

type mediaLifecycleCheck struct {
	Name     string      `json:"name"`
	Evidence artifact.ID `json:"evidence"`
	Protocol artifact.ID `json:"protocol"`
	Command  string      `json:"command"`
	Required []string    `json:"required_tests"`
	Scope    string      `json:"scope"`
}

// mediaSourceChange is one source file of the lifecycle bundle's patch, by the
// hashes of its bytes before and after and the reason it changed.
type mediaSourceChange struct {
	Before string `json:"before_sha256"`
	After  string `json:"after_sha256"`
	Reason string `json:"reason"`
}

type mediaLifecycleBundle struct {
	SurfaceReview artifact.ID                  `json:"source_size_review"`
	Version       uint16                       `json:"version"`
	Source        string                       `json:"source_base"`
	Protocol      artifact.ID                  `json:"protocol"`
	Patch         artifact.ID                  `json:"source_patch"`
	Changes       map[string]mediaSourceChange `json:"source_changes"`
	Checks        []mediaLifecycleCheck        `json:"checks"`
	Failed        []artifact.ID                `json:"failed_attempts"`
	Harnesses     map[string]artifact.ID       `json:"test_sources"`
	Scope         string                       `json:"scope"`
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

func TestImageVideoLifecycleAcceptance(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "retained media lifecycle evidence")
	root := testutil.RepoRoot(t)
	bundle, err := readMediaLifecycleBundle(root)
	if err != nil {
		t.Fatal(err)
	}
	if bundle.Version != artifact.InitialDocumentVersion || bundle.Scope == "" || len(bundle.Checks) == 0 || len(bundle.Failed) == 0 {
		t.Fatal("incomplete lifecycle evidence")
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
}

func TestImageVideoLifecycleRejectsChangedEvidence(t *testing.T) {
	t.Parallel()
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
