package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// mediaReviewedDeltasPath is the one registry of reviewed output-neutral
// source deltas between the pinned media runtime identity and the current
// tree. An output-neutral edit to a pinned path lands as one entry here plus
// the receipt that re-observes it.
const mediaReviewedDeltasPath = "docs/media_reviewed_deltas.json"

// mediaReviewedDeltasSHA256 is the registry's accepted canonical identity.
const mediaReviewedDeltasSHA256 = "7003854371976bbe7bc9eb6a8def9a5d444ce4e3a04cff18339e5bd72ddc756d"

// Receipt kinds: what re-observes a delta's neutrality.
const (
	mediaReceiptGoTest      = "go-test-json"         // store go test -json receipt naming the required tests
	mediaReceiptSourceBound = "source-bound-go-test" // store receipt acquired at the delta's source commit
	mediaReceiptRotary      = "rotary-ast"           // the rotary syntax proof, run against the source commit
	mediaReceiptDocument    = "document"             // a canonically bound evidence document declaring the changes
)

// catFileHeaderFields counts a git cat-file --batch blob header: object, type, size.
const catFileHeaderFields = 3

// mediaSourceChange is one file's reviewed move: its content sha256 before
// and after ("" = absent) and why output is unchanged.
type mediaSourceChange struct {
	Before string `json:"before_sha256"`
	After  string `json:"after_sha256"`
	Reason string `json:"reason"`
}

// mediaReviewedDelta is one reviewed landing. A document delta's changes are
// read from its evidence document, never copied into the registry.
type mediaReviewedDelta struct {
	ID             string                       `json:"id"`
	Receipt        string                       `json:"receipt"`
	Evidence       string                       `json:"evidence"`
	EvidenceSHA256 string                       `json:"evidence_sha256"`
	Source         string                       `json:"source"` // commit holding every after state; required by source-bound and rotary receipts
	Required       map[string][]string          `json:"required_tests"`
	Scope          string                       `json:"scope"`
	Changes        map[string]mediaSourceChange `json:"changes"`
}

// mediaDeltaRegistry chains deltas in landing order from base, the commit
// whose runtime identity the media evidence pins. Paths is the scope the
// registry answers for.
type mediaDeltaRegistry struct {
	Version uint16               `json:"version"`
	Base    string               `json:"base"`
	Paths   []string             `json:"paths"`
	Deltas  []mediaReviewedDelta `json:"deltas"`
}

// mediaDeltaRegistries caches one validated registry per repository root:
// the evidence behind it is read once per process, not once per check.
var mediaDeltaRegistries sync.Map // root -> func() (mediaDeltaRegistry, error)

// loadMediaDeltaRegistry returns root's validated registry. An absent
// registry is empty: nothing is reviewed, so any identity move is refused.
func loadMediaDeltaRegistry(root string) (mediaDeltaRegistry, error) {
	load, _ := mediaDeltaRegistries.LoadOrStore(root, sync.OnceValues(func() (mediaDeltaRegistry, error) {
		registry, err := readMediaDeltaRegistry(root)
		if err != nil || registry.Base == "" {
			return registry, err
		}
		return registry, checkMediaDeltaEvidence(root, registry)
	}))
	return load.(func() (mediaDeltaRegistry, error))()
}

// readMediaDeltaRegistry decodes the bound registry and resolves document
// deltas to the changes their documents declare. Evidence is not read.
func readMediaDeltaRegistry(root string) (mediaDeltaRegistry, error) {
	var registry mediaDeltaRegistry
	raw, err := os.ReadFile(filepath.Join(root, mediaReviewedDeltasPath))
	if errors.Is(err, fs.ErrNotExist) {
		return registry, nil
	}
	if err != nil {
		return registry, err
	}
	if err := checkMediaProtocolIdentity(raw, mediaReviewedDeltasSHA256); err != nil {
		return registry, fmt.Errorf("%s: %w", mediaReviewedDeltasPath, err)
	}
	if err := strictjson.DecodeBytes(raw, &registry); err != nil {
		return registry, err
	}
	if registry.Version != artifact.InitialDocumentVersion || !gitauthority.ValidObjectID(registry.Base) || len(registry.Paths) == 0 || len(registry.Deltas) == 0 {
		return registry, errors.New("incomplete reviewed delta registry")
	}
	seen := map[string]bool{}
	for index := range registry.Deltas {
		delta := &registry.Deltas[index]
		if delta.ID == "" || seen[delta.ID] || delta.Scope == "" || len(delta.Required) == 0 {
			return registry, fmt.Errorf("incomplete or duplicate reviewed delta %q", delta.ID)
		}
		seen[delta.ID] = true
		if delta.Receipt == mediaReceiptDocument {
			if len(delta.Changes) != 0 {
				return registry, fmt.Errorf("reviewed delta %s copies its document's changes", delta.ID)
			}
			document, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(delta.Evidence)))
			if err != nil {
				return registry, err
			}
			if err := checkMediaProtocolIdentity(document, delta.EvidenceSHA256); err != nil {
				return registry, fmt.Errorf("reviewed delta %s: %s: %w", delta.ID, delta.Evidence, err)
			}
			if delta.Changes, err = mediaDocumentChanges(document); err != nil {
				return registry, fmt.Errorf("reviewed delta %s: %w", delta.ID, err)
			}
		}
		if len(delta.Changes) == 0 {
			return registry, fmt.Errorf("reviewed delta %s changes nothing", delta.ID)
		}
		for path, change := range delta.Changes {
			if !mediaLifecycleProductionPath(path) || !mediaPathInScope(path, registry.Paths) || change.Before == change.After || change.Reason == "" ||
				!mediaOptionalDigest(change.Before) || !mediaOptionalDigest(change.After) {
				return registry, fmt.Errorf("reviewed delta %s: malformed change %s", delta.ID, path)
			}
		}
	}
	return registry, nil
}

// mediaDocumentChanges reads the change set an evidence document declares:
// a source_changes map, or one pixel_range_correction.
func mediaDocumentChanges(raw []byte) (map[string]mediaSourceChange, error) {
	var document struct {
		Changes    map[string]mediaSourceChange `json:"source_changes"`
		Correction *struct {
			Path   string `json:"path"`
			Before string `json:"before_git_sha256"`
			After  string `json:"after_git_sha256"`
			Scope  string `json:"scope"`
		} `json:"pixel_range_correction"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, err
	}
	switch {
	case len(document.Changes) != 0 && document.Correction == nil:
		return document.Changes, nil
	case len(document.Changes) == 0 && document.Correction != nil:
		fix := document.Correction
		return map[string]mediaSourceChange{fix.Path: {Before: fix.Before, After: fix.After, Reason: fix.Scope}}, nil
	}
	return nil, errors.New("evidence document declares no single source change set")
}

// mediaOptionalDigest accepts a sha256 hex digest or "" (absent).
func mediaOptionalDigest(value string) bool {
	return value == "" || artifact.ValidHexDigest(value)
}

// mediaReceiptStore reads delta receipts from the read-only retained store,
// opened on first use so a registry of documents alone never opens it.
type mediaReceiptStore struct {
	root  string
	store *overgodb.Store
}

func (s *mediaReceiptStore) read(delta mediaReviewedDelta) ([]byte, error) {
	id, err := artifact.ParseID(delta.Evidence)
	if err != nil {
		return nil, fmt.Errorf("reviewed delta %s: %w", delta.ID, err)
	}
	if s.store == nil {
		roots, err := dataroot.Resolve(s.root)
		if err != nil {
			return nil, err
		}
		if s.store, err = overgodb.OpenReadOnly(retainedReferenceStore(roots.Store)); err != nil {
			return nil, err
		}
	}
	content, found, err := artifact.ReadContent(context.Background(), s.store, id)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("missing %s output-neutrality receipt", delta.ID)
	}
	return content.Data, nil
}

func (s *mediaReceiptStore) close() error {
	if s.store == nil {
		return nil
	}
	return s.store.Close()
}

// checkMediaDeltaEvidence re-reads every delta's receipt; see checkMediaDeltaReceipt.
func checkMediaDeltaEvidence(root string, registry mediaDeltaRegistry) error {
	receipts := &mediaReceiptStore{root: root}
	for _, delta := range registry.Deltas {
		if err := checkMediaDeltaReceipt(receipts, registry.Paths, delta); err != nil {
			return errors.Join(err, receipts.close())
		}
	}
	return receipts.close()
}

// checkMediaDeltaReceipt holds one delta to its evidence: a store receipt must
// pass the required tests, a source commit must hold every after state, and
// the rotary delta must pass its syntax proof from the proof base. A document
// delta was bound when read.
func checkMediaDeltaReceipt(receipts *mediaReceiptStore, scope []string, delta mediaReviewedDelta) error {
	root := receipts.root
	name := delta.ID + " reviewed delta"
	paths := slices.Sorted(maps.Keys(delta.Changes))
	if delta.Source != "" {
		if !gitauthority.ValidObjectID(delta.Source) {
			return fmt.Errorf("%s: source is not a commit", name)
		}
		after, err := mediaSourceDigests(root, delta.Source, paths)
		if err != nil {
			return err
		}
		for _, path := range paths {
			if after[path] != delta.Changes[path].After {
				return fmt.Errorf("%s: source commit differs at %s", name, path)
			}
		}
	}
	switch delta.Receipt {
	case mediaReceiptGoTest:
		data, err := receipts.read(delta)
		if err != nil {
			return err
		}
		return checkMediaTestReceipt(name, data, delta.Required)
	case mediaReceiptSourceBound:
		data, err := receipts.read(delta)
		if err != nil {
			return err
		}
		var receipt struct{ Source, Output string }
		if delta.Source == "" || json.Unmarshal(data, &receipt) != nil || receipt.Source != delta.Source {
			return fmt.Errorf("%s: missing source-bound receipt", name)
		}
		return checkMediaTestReceipt(name, []byte(receipt.Output), delta.Required)
	case mediaReceiptRotary:
		if delta.Source == "" || delta.Evidence != "" {
			return fmt.Errorf("%s: the syntax proof binds a source commit only", name)
		}
		before, err := mediaSourceDigests(root, mediaRotaryBase, paths)
		if err != nil {
			return err
		}
		for _, path := range paths {
			if before[path] != delta.Changes[path].Before {
				return fmt.Errorf("%s: proof base differs at %s", name, path)
			}
		}
		if _, err := checkMediaRotarySource(root, delta.Source, scope); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	case mediaReceiptDocument:
		return nil
	}
	return fmt.Errorf("%s: unknown receipt kind %q", name, delta.Receipt)
}

// checkMediaRuntimeAtRevision accepts a revision whose runtime identity is the
// expected one, or whose every production change under paths since the
// registry base is a landed prefix of the reviewed delta chain. An empty
// revision checks the working tree before gate publication.
func checkMediaRuntimeAtRevision(root, revision string, paths []string, expected string) error {
	if revision != "" {
		actual, err := mediaRuntimeIdentity(root, revision, paths)
		if err != nil {
			return err
		}
		if compareMediaRuntimeIdentity(expected, actual) == nil {
			return nil
		}
	}
	registry, err := loadMediaDeltaRegistry(root)
	if err != nil {
		return err
	}
	if registry.Base == "" {
		return errors.New("retained generation source scope changed and no reviewed delta answers for it")
	}
	anchor, err := mediaRuntimeIdentity(root, registry.Base, paths)
	if err != nil {
		return err
	}
	if err := compareMediaRuntimeIdentity(expected, anchor); err != nil {
		return fmt.Errorf("reviewed delta base: %w", err)
	}
	unreviewed, err := mediaUnreviewedChanges(root, registry, revision, paths)
	if err != nil {
		return err
	}
	if len(unreviewed) != 0 {
		return fmt.Errorf("unreviewed generation source change: %s", strings.Join(unreviewed, ", "))
	}
	return nil
}

// mediaUnreviewedChanges names the production changes under paths between the
// registry base and revision that no landed prefix of the chain explains.
func mediaUnreviewedChanges(root string, registry mediaDeltaRegistry, revision string, paths []string) ([]string, error) {
	_, unreviewed, err := mediaDeltaStage(root, registry, revision, paths)
	return unreviewed, err
}

// mediaDeltaStage walks the chain from each in-scope file's content at the
// base and returns how many deltas have landed at revision: the last stage at
// which every in-scope file the chain touches holds exactly the chained
// content. A half-landed delta matches no stage, so its files are named. A
// change the chain never touches is always named. A delta whose before state
// is not the chain's content at that point breaks the chain and is an error.
func mediaDeltaStage(root string, registry mediaDeltaRegistry, revision string, paths []string) (int, []string, error) {
	changed, err := mediaRuntimeChanges(root, registry.Base, revision, paths)
	if err != nil {
		return 0, nil, err
	}
	var touched []string
	for _, delta := range registry.Deltas {
		for path := range delta.Changes {
			if mediaPathInScope(path, paths) && !slices.Contains(touched, path) {
				touched = append(touched, path)
			}
		}
	}
	slices.Sort(touched)
	unreviewed := slices.DeleteFunc(slices.Clone(changed), func(path string) bool {
		_, found := slices.BinarySearch(touched, path)
		return found
	})
	stage := 0
	if len(touched) != 0 {
		state, err := mediaSourceDigests(root, registry.Base, touched)
		if err != nil {
			return 0, nil, err
		}
		current, err := mediaSourceDigests(root, revision, touched)
		if err != nil {
			return 0, nil, err
		}
		stage = -1
		if maps.Equal(state, current) {
			stage = 0
		}
		for index, delta := range registry.Deltas {
			for path, change := range delta.Changes {
				if !mediaPathInScope(path, paths) {
					continue
				}
				if change.Before != state[path] {
					return 0, nil, fmt.Errorf("reviewed delta %s does not follow the chain at %s", delta.ID, path)
				}
				state[path] = change.After
			}
			if maps.Equal(state, current) {
				stage = index + 1
			}
		}
		if stage < 0 {
			for _, path := range touched {
				if state[path] != current[path] {
					unreviewed = append(unreviewed, path)
				}
			}
		}
	}
	slices.Sort(unreviewed)
	return stage, slices.Compact(unreviewed), nil
}

// mediaPathInScope reports whether path lies under one of the scope paths.
func mediaPathInScope(path string, scope []string) bool {
	return slices.ContainsFunc(scope, func(prefix string) bool {
		return prefix == "." || path == prefix || strings.HasPrefix(path, strings.TrimSuffix(prefix, "/")+"/")
	})
}

// mediaSourceDigests returns each path's content sha256 at revision, "" where
// the path is absent. An empty revision reads the working tree with CRLF
// folded to LF; a commit is read through one git process for every path.
func mediaSourceDigests(root, revision string, paths []string) (map[string]string, error) {
	digests := make(map[string]string, len(paths))
	if revision == "" {
		for _, path := range paths {
			data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
			if errors.Is(err, fs.ErrNotExist) {
				digests[path] = ""
				continue
			}
			if err != nil {
				return nil, err
			}
			digests[path] = fmt.Sprintf("%x", sha256.Sum256(bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))))
		}
		return digests, nil
	}
	var request strings.Builder
	for _, path := range paths {
		fmt.Fprintf(&request, "%s:%s\n", revision, path)
	}
	command := exec.Command("git", "cat-file", "--batch")
	command.Dir = root
	command.Stdin = strings.NewReader(request.String())
	out, err := command.Output()
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReader(bytes.NewReader(out))
	for _, path := range paths {
		header, err := reader.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("git cat-file: %s: %w", path, err)
		}
		header = strings.TrimSuffix(header, "\n")
		if strings.HasSuffix(header, " missing") {
			digests[path] = ""
			continue
		}
		fields := strings.Fields(header)
		if len(fields) != catFileHeaderFields || fields[1] != "blob" {
			return nil, fmt.Errorf("git cat-file: %s: %s", path, header)
		}
		size, err := strconv.Atoi(fields[2])
		if err != nil {
			return nil, err
		}
		data := make([]byte, size+len("\n"))
		if _, err := io.ReadFull(reader, data); err != nil {
			return nil, fmt.Errorf("git cat-file: %s: %w", path, err)
		}
		digests[path] = fmt.Sprintf("%x", sha256.Sum256(data[:size]))
	}
	return digests, nil
}

// Empty revision includes worktree and untracked inputs; test-only edits do not
// turn historical numerical outputs into fresh measurements.
func mediaRuntimeChanges(root, before, revision string, paths []string) ([]string, error) {
	git := func(args ...string) ([]byte, error) {
		c := exec.Command("git", args...)
		c.Dir = root
		return c.Output()
	}
	// Inspect both sides: a rename from source to a non-source extension is a
	// deletion from the execution scope, even when Git detects identical bytes.
	args := []string{"diff", "--no-renames", "--name-only", "-z", before}
	if revision != "" {
		args = append(args, revision)
	}
	changed, err := git(append(append(args, "--"), paths...)...)
	if err != nil {
		return nil, err
	}
	if revision == "" {
		untracked, err := git(append([]string{"ls-files", "--others", "--exclude-standard", "-z", "--"}, paths...)...)
		if err != nil {
			return nil, err
		}
		changed = append(changed, untracked...)
	}
	return slices.DeleteFunc(strings.Split(string(changed), "\x00"), func(path string) bool { return !mediaLifecycleProductionPath(path) }), nil
}

// TestReviewedDeltaRegistryIdentity binds the registry field by field: dropping
// any top-level or delta field breaks its canonical identity.
func TestReviewedDeltaRegistryIdentity(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, mediaReviewedDeltasPath))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaProtocolIdentity(raw, mediaReviewedDeltasSHA256); err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(raw, &document); err != nil {
		t.Fatal(err)
	}
	accepts := func(altered map[string]any) bool {
		encoded, err := json.Marshal(altered)
		if err != nil {
			t.Fatal(err)
		}
		return checkMediaProtocolIdentity(encoded, mediaReviewedDeltasSHA256) == nil
	}
	for field := range document {
		var altered map[string]any
		if err := json.Unmarshal(raw, &altered); err != nil {
			t.Fatal(err)
		}
		delete(altered, field)
		if accepts(altered) {
			t.Fatalf("missing %s accepted", field)
		}
	}
	for index, entry := range document["deltas"].([]any) {
		for field := range entry.(map[string]any) {
			var altered map[string]any
			if err := json.Unmarshal(raw, &altered); err != nil {
				t.Fatal(err)
			}
			delete(altered["deltas"].([]any)[index].(map[string]any), field)
			if accepts(altered) {
				t.Fatalf("delta %d: missing %s accepted", index, field)
			}
		}
	}
}

// TestReviewedDeltaRegistryBase holds the chain's base to the identities the
// media evidence pins, and the registry's scope to every pinned path, with no
// scope entry that neither the pin nor a delta needs.
func TestReviewedDeltaRegistryBase(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	registry, err := readMediaDeltaRegistry(root)
	if err != nil {
		t.Fatal(err)
	}
	var merged mediaMergedEvidence
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_merged.json"), &merged); err != nil {
		t.Fatal(err)
	}
	var routed struct {
		StablePaths []string `json:"stable_paths"`
		StableSHA   string   `json:"stable_sha256"`
	}
	if err := jsonfile.Decode(filepath.Join(root, "docs/image_video_routed_images.json"), &routed); err != nil {
		t.Fatal(err)
	}
	for _, pin := range []struct {
		paths    []string
		expected string
	}{{merged.RuntimePaths, merged.RuntimeSHA256}, {routed.StablePaths, routed.StableSHA}} {
		identity, err := mediaRuntimeIdentity(root, registry.Base, pin.paths)
		if err != nil {
			t.Fatal(err)
		}
		if err := compareMediaRuntimeIdentity(pin.expected, identity); err != nil {
			t.Fatalf("registry base is not the pinned identity: %v", err)
		}
		for _, path := range pin.paths {
			if !mediaPathInScope(path, registry.Paths) {
				t.Fatalf("pinned path outside the registry scope: %s", path)
			}
		}
	}
	for _, scope := range registry.Paths {
		used := slices.Contains(merged.RuntimePaths, scope)
		for _, delta := range registry.Deltas {
			for path := range delta.Changes {
				used = used || mediaPathInScope(path, []string{scope})
			}
		}
		if !used {
			t.Fatalf("registry scope entry no pin or delta needs: %s", scope)
		}
	}
}

// TestReviewedDeltaRegistryExplainsTheTree proves, at HEAD and in the working
// tree, that the pinned runtime has moved and that nothing outside the chain
// moved. The working tree has landed the whole chain; HEAD may stand at a
// landed prefix of it, since a delta arrives in the commit that makes its
// change.
func TestReviewedDeltaRegistryExplainsTheTree(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	registry, err := loadMediaDeltaRegistry(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{"HEAD", ""} {
		raw, err := mediaRuntimeChanges(root, registry.Base, revision, registry.Paths)
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) == 0 {
			t.Fatalf("%q: the registry answers for no move", revision)
		}
		stage, unreviewed, err := mediaDeltaStage(root, registry, revision, registry.Paths)
		if err != nil {
			t.Fatal(err)
		}
		whole := revision == "" && stage != len(registry.Deltas)
		if len(unreviewed) != 0 || whole {
			t.Fatalf("%q: stage %d of %d, unreviewed %v", revision, stage, len(registry.Deltas), unreviewed)
		}
	}
}

// TestReviewedDeltaRegistryEvidence re-reads every receipt and holds each kind
// to refusing a delta its evidence does not answer for.
func TestReviewedDeltaRegistryEvidence(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	registry, err := loadMediaDeltaRegistry(root)
	if err != nil {
		t.Fatal(err)
	}
	receipts := &mediaReceiptStore{root: root}
	t.Cleanup(func() {
		if err := receipts.close(); err != nil {
			t.Error(err)
		}
	})
	kinds := map[string]bool{}
	for _, delta := range registry.Deltas {
		kinds[delta.Receipt] = true
		broken := delta
		broken.Changes = maps.Clone(delta.Changes)
		switch delta.Receipt {
		case mediaReceiptGoTest, mediaReceiptSourceBound:
			broken.Required = map[string][]string{"overgo/cmd/compatibility": {"TestAbsentFromReceipt"}}
		case mediaReceiptRotary:
			path := slices.Sorted(maps.Keys(delta.Changes))[0]
			change := broken.Changes[path]
			change.Before = change.After
			broken.Changes[path] = change
		case mediaReceiptDocument:
			continue // bound by its document's identity when read
		}
		if checkMediaDeltaReceipt(receipts, registry.Paths, broken) == nil {
			t.Fatalf("%s: evidence accepted for a delta it does not answer for", delta.ID)
		}
		if delta.Source != "" {
			moved := delta
			moved.Source = registry.Base
			if checkMediaDeltaReceipt(receipts, registry.Paths, moved) == nil {
				t.Fatalf("%s: a source commit without the after states accepted", delta.ID)
			}
		}
	}
	for _, kind := range []string{mediaReceiptGoTest, mediaReceiptSourceBound, mediaReceiptRotary, mediaReceiptDocument} {
		if !kinds[kind] {
			t.Fatalf("receipt kind %s is unused", kind)
		}
	}
	if _, err := mediaDocumentChanges([]byte(`{"scope":"no changes"}`)); err == nil {
		t.Fatal("an evidence document without changes accepted")
	}
}

// TestReviewedDeltaRegistrySubtractsReviewedChanges drives the chain over a
// throwaway repository: reviewed changes subtract, everything else is named.
func TestReviewedDeltaRegistrySubtractsReviewedChanges(t *testing.T) {
	t.Parallel()
	fixture := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		c := exec.Command("git", args...)
		c.Dir = fixture
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git: %v %s", err, out)
		}
		return strings.TrimSpace(string(out))
	}
	write := func(name, text string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(fixture, name), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	digest := func(text string) string { return fmt.Sprintf("%x", sha256.Sum256([]byte(text))) }
	git("init", "-q")
	git("config", "user.email", "fixture@example.invalid")
	git("config", "user.name", "Fixture")
	git("config", "core.autocrlf", "false")
	const kernel, reviewed, helper = "package fixture\n", "package fixture\n// reviewed\n", "package fixture\n// helper\n"
	write("kernel.go", kernel)
	write("helper.go", helper)
	git("add", ".")
	git("commit", "-qm", "baseline")
	base := git("rev-parse", "HEAD")
	scope := []string{"."}
	registry := mediaDeltaRegistry{Base: base, Deltas: []mediaReviewedDelta{{ID: "edit", Changes: map[string]mediaSourceChange{
		"kernel.go": {Before: digest(kernel), After: digest(reviewed), Reason: "fixture"},
	}}}}
	expect := func(registry mediaDeltaRegistry, revision string, want ...string) {
		t.Helper()
		unreviewed, err := mediaUnreviewedChanges(fixture, registry, revision, scope)
		if err != nil || !slices.Equal(unreviewed, want) {
			t.Fatalf("unreviewed %v (%v), want %v", unreviewed, err, want)
		}
	}
	expect(registry, "")
	// A reviewed change written with CRLF endings is its reviewed content.
	write("kernel.go", strings.ReplaceAll(reviewed, "\n", "\r\n"))
	expect(registry, "")
	// A test-only change never moves the runtime.
	write("kernel_test.go", "package fixture\n")
	expect(registry, "")
	// An unreviewed change is named.
	write("helper.go", helper+"// drift\n")
	expect(registry, "", "helper.go")
	write("helper.go", helper)
	// A delta whose before is not the chain's content breaks the chain even
	// though its after matches.
	wrong := registry
	wrong.Deltas = []mediaReviewedDelta{{ID: "wrong", Changes: map[string]mediaSourceChange{
		"kernel.go": {Before: digest("package other\n"), After: digest(reviewed), Reason: "fixture"},
	}}}
	if _, err := mediaUnreviewedChanges(fixture, wrong, "", scope); err == nil || !strings.Contains(err.Error(), "kernel.go") {
		t.Fatalf("broken chain accepted: %v", err)
	}
	// A half-landed delta is named where it has not landed.
	whole := registry
	whole.Deltas = []mediaReviewedDelta{{ID: "pair", Changes: map[string]mediaSourceChange{
		"kernel.go": {Before: digest(kernel), After: digest(reviewed), Reason: "fixture"},
		"helper.go": {Before: digest(helper), After: digest(helper + "// paired\n"), Reason: "fixture"},
	}}}
	expect(whole, "", "helper.go")
	// A deletion is named unless a delta reviews the absence.
	if err := os.Remove(filepath.Join(fixture, "helper.go")); err != nil {
		t.Fatal(err)
	}
	expect(registry, "", "helper.go")
	removal := registry
	removal.Deltas = append(slices.Clone(registry.Deltas), mediaReviewedDelta{ID: "removal", Changes: map[string]mediaSourceChange{
		"helper.go": {Before: digest(helper), After: "", Reason: "fixture"},
	}})
	expect(removal, "")
	// Committed revisions read the same chain; a commit stores bytes as
	// written, so the reviewed content is committed with LF endings.
	write("kernel.go", reviewed)
	git("add", "-A")
	git("commit", "-qm", "reviewed")
	expect(removal, "HEAD")
	expect(removal, base)
	expect(registry, "HEAD", "helper.go")
	invalid := registry
	invalid.Base = "missing-reference"
	if _, err := mediaUnreviewedChanges(fixture, invalid, "", scope); err == nil {
		t.Fatal("invalid base accepted")
	}
	// Without a registry, nothing is reviewed.
	empty, err := loadMediaDeltaRegistry(fixture)
	if err != nil || empty.Base != "" {
		t.Fatalf("absent registry: %+v %v", empty, err)
	}
	identity, err := mediaRuntimeIdentity(fixture, base, scope)
	if err != nil {
		t.Fatal(err)
	}
	if checkMediaRuntimeAtRevision(fixture, "HEAD", scope, identity) == nil {
		t.Fatal("a moved runtime accepted without a registry")
	}
	if checkMediaRuntimeAtRevision(fixture, base, scope, identity) != nil {
		t.Fatal("the pinned runtime refused")
	}
}

// TestReviewedDeltaRegistryRuntimeChanges holds the production-source
// selection: test files are ignored, additions, removals and renames to a
// non-source name are changes, and a bad reference is an error.
func TestReviewedDeltaRegistryRuntimeChanges(t *testing.T) {
	t.Parallel()
	fixture := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		c := exec.Command("git", args...)
		c.Dir = fixture
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git: %v %s", err, out)
		}
	}
	write := func(name, text string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(fixture, name), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	git("config", "user.email", "fixture@example.invalid")
	git("config", "user.name", "Fixture")
	write("kernel.go", "package fixture\n")
	git("add", ".")
	git("commit", "-qm", "baseline")
	baseline, err := mediaRuntimeIdentity(fixture, "HEAD", []string{"."})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"kernel_test.go", "kernel.go", "added.go", "added file.go", "café.go"} {
		write(name, "package fixture\n// changed\n")
		changed, err := mediaRuntimeChanges(fixture, "HEAD", "", []string{"."})
		if err != nil || slices.Contains(changed, name) == strings.HasSuffix(name, "_test.go") {
			t.Fatalf("source selection %s: %v %v", name, changed, err)
		}
	}
	if _, err := mediaRuntimeChanges(fixture, "missing-reference", "", []string{"."}); err == nil {
		t.Fatal("invalid reference accepted")
	}
	git("add", ".")
	git("commit", "-qm", "source additions")
	added, err := mediaRuntimeIdentity(fixture, "HEAD", []string{"."})
	if err != nil || added == baseline {
		t.Fatal("source additions did not change identity", err)
	}
	digests, err := mediaSourceDigests(fixture, "HEAD", []string{"added file.go", "café.go", "absent.go"})
	if err != nil || digests["absent.go"] != "" || digests["café.go"] != fmt.Sprintf("%x", sha256.Sum256([]byte("package fixture\n// changed\n"))) || digests["added file.go"] != digests["café.go"] {
		t.Fatalf("committed digests: %v %v", digests, err)
	}
	for _, name := range []string{"kernel.go", "added file.go", "café.go"} {
		git("mv", name, name+".txt")
	}
	if err := os.Remove(filepath.Join(fixture, "added.go")); err != nil {
		t.Fatal(err)
	}
	changed, err := mediaRuntimeChanges(fixture, "HEAD", "", []string{"."})
	slices.Sort(changed)
	if err != nil || !slices.Equal(changed, []string{"added file.go", "added.go", "café.go", "kernel.go"}) {
		t.Fatalf("source removals/renames lost: %v, %v", changed, err)
	}
	git("add", ".")
	git("commit", "-qm", "source removals")
	if _, err := mediaRuntimeIdentity(fixture, "HEAD", []string{"."}); err == nil {
		t.Fatal("test-only tree accepted as a production source identity")
	}
}
