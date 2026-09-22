package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
)

// mediaLayer is one reviewed output-neutral delta in the media runtime
// identity chain: a document naming its base commit, the before and after
// identity of every production file it reconciles, and the receipt of the
// tests that re-observe the affected paths at the current surface. The chain
// peels layers newest first, each returning its base for the next.
type mediaLayer struct {
	name     string // names the layer in refusals
	document string // repository-relative reconciliation document
	sha256   string // the document's accepted identity
	base     string // the commit the layer peels back to
}

// checkMediaLayerSource peels one layer: when every production change under
// paths between the layer's base and revision is one the document reviews,
// with the exact prior and current source identities it declares, and the
// bound receipt shows the required tests passing, it returns the base and
// true. A change outside the layer is left for the caller to reconcile; the
// layer then answers false without error.
func checkMediaLayerSource(root, revision string, paths []string, layer mediaLayer) (string, bool, error) {
	raw, err := os.ReadFile(filepath.Join(root, layer.document))
	if err != nil {
		return "", false, err
	}
	if err := checkMediaProtocolIdentity(raw, layer.sha256); err != nil {
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
	if proof.Base != layer.base || len(proof.Changes) == 0 || len(proof.Required) == 0 || proof.Scope == "" || !proof.Evidence.Valid() {
		return "", false, fmt.Errorf("missing %s reconciliation closure", layer.name)
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
		return "", false, fmt.Errorf("missing %s output-neutrality receipt", layer.name)
	}
	if err := checkMediaTestReceipt(layer.name+" reconciliation", content.Data, proof.Required); err != nil {
		return "", false, err
	}
	return proof.Base, true, nil
}

// requireMediaLayerPeels proves a layer peels to its declared base at the
// given revision and that dropping any bound field of its document breaks the
// document's identity. Callers peel the newer layers first and pass the
// revision they reached.
func requireMediaLayerPeels(t *testing.T, root, revision string, paths []string, layer mediaLayer) {
	t.Helper()
	base, peeled, err := checkMediaLayerSource(root, revision, paths, layer)
	if err != nil {
		t.Fatal(err)
	}
	if !peeled || base != layer.base {
		t.Fatalf("reviewed deltas did not peel to the base: peeled=%v base=%s", peeled, base)
	}
	if !gitauthority.ValidObjectID(base) {
		t.Fatalf("base is not a commit: %s", base)
	}
	raw, err := os.ReadFile(filepath.Join(root, layer.document))
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
		if checkMediaProtocolIdentity(encoded, layer.sha256) == nil {
			t.Fatalf("missing %s accepted", field)
		}
	}
}

// peelNewerMediaLayers peels the layers newer than the one under test, newest
// first, and returns the revision the older layer must answer for.
func peelNewerMediaLayers(t *testing.T, root string, paths []string, newer ...mediaLayer) string {
	t.Helper()
	revision := ""
	for _, layer := range newer {
		peeledBase, peeled, err := checkMediaLayerSource(root, revision, paths, layer)
		if err != nil {
			t.Fatal(err)
		}
		if peeled {
			revision = peeledBase
		}
	}
	return revision
}
