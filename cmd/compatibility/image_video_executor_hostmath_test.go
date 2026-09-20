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

const mediaExecutorHostmathSHA256 = "d4b8bb18c031b65509a5ac6eae3c421bacd58dfaa32584129c1213165cad9bd6"

// checkMediaExecutorHostmathSource peels the two reviewed output-neutral runtime
// deltas that moved the media source identity past the rotary base: the executor
// device-byte accounting and the host linear-column reorder. It returns the
// pre-delta base and whether it peeled, so the existing chain reconciles the
// rest. Each delta is bound by its before/after source identity plus a receipt
// that re-observes the affected kernels at the current surface; a production
// change that is not one of these two is left for the caller to reconcile.
func checkMediaExecutorHostmathSource(root, revision string, paths []string) (string, bool, error) {
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_executor_hostmath_reconciliation.json"))
	if err != nil {
		return "", false, err
	}
	if err := checkMediaProtocolIdentity(raw, mediaExecutorHostmathSHA256); err != nil {
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
	if !gitauthority.ValidObjectID(proof.Base) || len(proof.Changes) == 0 || len(proof.Required) == 0 || proof.Scope == "" {
		return "", false, errors.New("missing executor/hostmath reconciliation closure")
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
		return "", false, errors.New("missing executor/hostmath output-neutrality receipt")
	}
	if err := checkMediaTestReceipt("executor/hostmath reconciliation", content.Data, proof.Required); err != nil {
		return "", false, err
	}
	return proof.Base, true, nil
}

// TestMediaExecutorHostmathReconciliation proves the reviewed deltas peel to the
// declared base at the current surface and that dropping any bound field breaks
// the fixture identity.
func TestMediaExecutorHostmathReconciliation(t *testing.T) {
	root := testutil.RepoRoot(t)
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	base, peeled, err := checkMediaExecutorHostmathSource(root, "", merged.RuntimePaths)
	if err != nil {
		t.Fatal(err)
	}
	if !peeled || base != "7abbfa4e29c343f897e56f8eb167c19e0ccae1d4" {
		t.Fatalf("reviewed deltas did not peel to the base: peeled=%v base=%s", peeled, base)
	}
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_executor_hostmath_reconciliation.json"))
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
		if checkMediaProtocolIdentity(encoded, mediaExecutorHostmathSHA256) == nil {
			t.Fatalf("missing %s accepted", field)
		}
	}
}
