package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const mediaWorkflowruntimeSHA256 = "6ea7f37c4f6096b37a3058dd9a5a171e66400e650c2b52ab77667919e30311d4"

// mediaWorkflowruntimeBase is the commit at which the media runtime chain was
// last whole: every later reviewed delta peels back to it.
const mediaWorkflowruntimeBase = "043805ef7f349dd94c1d2e6a7224fe0ccef34ef8"

// checkMediaWorkflowruntimeSource peels the two reviewed output-neutral deltas
// the harness landings made under internal/workflowruntime after the
// executor/hostmath base: an idempotent commit routed through its one owner,
// and a dead wrapper removed. Neither touches a generation path; the media
// runtime paths include the package because the media lifecycle workflow runs
// through it, and the lifecycle receipt re-observes that workflow's own tests
// at the current surface. It returns the pre-delta base and whether it
// peeled; a production change that is not one of these two is left for the
// caller to reconcile.
func checkMediaWorkflowruntimeSource(root, revision string, paths []string) (string, bool, error) {
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_workflowruntime_reconciliation.json"))
	if err != nil {
		return "", false, err
	}
	if err := checkMediaProtocolIdentity(raw, mediaWorkflowruntimeSHA256); err != nil {
		return "", false, err
	}
	var proof struct {
		Base     string                          `json:"base"`
		Changes  map[string]mediaLifecycleChange `json:"source_changes"`
		Evidence artifact.ID                     `json:"evidence"`
		Required map[string][]string             `json:"required_tests"`
		Scope    string                          `json:"scope"`
	}
	if err := json.Unmarshal(raw, &proof); err != nil {
		return "", false, err
	}
	if proof.Base != mediaWorkflowruntimeBase || len(proof.Changes) == 0 || len(proof.Required) == 0 || proof.Scope == "" || !proof.Evidence.Valid() {
		return "", false, errors.New("missing workflowruntime reconciliation closure")
	}
	changed, err := mediaRuntimeChanges(root, proof.Base, revision, paths)
	if err != nil {
		return "", false, err
	}
	if len(changed) == 0 {
		return proof.Base, false, nil
	}
	for _, path := range changed {
		if _, ok := proof.Changes[path]; !ok {
			return "", false, nil // not this reconciliation; the caller handles it
		}
	}
	git := func(args ...string) ([]byte, error) {
		command := exec.Command("git", args...)
		command.Dir = root
		return command.Output()
	}
	for path, change := range proof.Changes {
		included := false
		for _, scope := range paths {
			included = included || path == scope || strings.HasPrefix(path, strings.TrimSuffix(scope, "/")+"/")
		}
		if !included {
			continue
		}
		before, err := git("show", proof.Base+":"+path)
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(before)) != change.Before {
			return "", false, fmt.Errorf("reconciliation prior source differs: %s", path)
		}
		var after []byte
		if revision == "" {
			after, err = os.ReadFile(filepath.Join(root, path))
			after = []byte(strings.ReplaceAll(string(after), "\r\n", "\n"))
		} else {
			after, err = git("show", revision+":"+path)
		}
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(after)) != change.After {
			return "", false, fmt.Errorf("reconciliation current source differs: %s", path)
		}
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		return "", false, err
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		return "", false, err
	}
	defer store.Close()
	content, found, err := artifact.ReadContent(context.Background(), store, proof.Evidence)
	if err != nil {
		return "", false, err
	}
	if !found {
		return "", false, errors.New("missing workflowruntime output-neutrality receipt")
	}
	if err := checkMediaTestReceipt("workflowruntime reconciliation", content.Data, proof.Required); err != nil {
		return "", false, err
	}
	return proof.Base, true, nil
}

// TestMediaWorkflowruntimeReconciliation proves the reviewed deltas peel to the
// declared base at the current surface and that dropping any bound field breaks
// the fixture identity.
func TestMediaWorkflowruntimeReconciliation(t *testing.T) {
	root := testutil.RepoRoot(t)
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	paths := append([]string{"internal/workflowruntime"}, merged.RuntimePaths...)
	base, peeled, err := checkMediaWorkflowruntimeSource(root, "", paths)
	if err != nil {
		t.Fatal(err)
	}
	if !peeled || base != mediaWorkflowruntimeBase {
		t.Fatalf("reviewed deltas did not peel to the base: peeled=%v base=%s", peeled, base)
	}
	if !gitauthority.ValidObjectID(base) {
		t.Fatalf("base is not a commit: %s", base)
	}
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_workflowruntime_reconciliation.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"base", "source_changes", "evidence", "required_tests", "scope"} {
		var altered map[string]any
		if err := json.Unmarshal(raw, &altered); err != nil {
			t.Fatal(err)
		}
		delete(altered, field)
		encoded, err := json.Marshal(altered)
		if err != nil {
			t.Fatal(err)
		}
		if checkMediaProtocolIdentity(encoded, mediaWorkflowruntimeSHA256) == nil {
			t.Fatalf("missing %s accepted", field)
		}
	}
}
