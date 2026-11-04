package overgodb

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// TestLineageRestoresFromItsCompactBody holds lineage's checkpoint to its compact body: a facet of several children,
// parents and relations checkpoints each identity once and each edge as three indexes, restores to the same lists, and
// files a later edge as the facet does; and a set that holds lineage's version-1 body, as a store does until its next
// snapshot, still loads to the same facet.
func TestLineageRestoresFromItsCompactBody(t *testing.T) {
	t.Parallel()
	ids := make([]artifact.ID, 6)
	for index := range ids {
		var err error
		if ids[index], err = artifact.IdentifyBytes(artifact.KindEvidence, []byte{byte(index)}); err != nil {
			t.Fatal(err)
		}
	}
	facet := newLineageFacet()
	for _, edge := range [][3]int{{0, 1, 1}, {0, 2, 2}, {3, 1, 1}, {4, 0, 2}, {4, 3, 1}, {2, 1, 2}, {0, 1, 2}} {
		facet.add(relationKey{child: ids[edge[0]], parent: ids[edge[1]], relation: artifact.Relation(edge[2])})
	}
	payload, err := facet.checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	var body lineageCheckpointBody
	if err := json.Unmarshal(payload, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.IDs) != 5 || len(body.Relations) != 2 || len(body.Edges) != lineageEdgeIndexes*facet.edges {
		t.Fatalf("the body holds %d identities, %d relations and %d indexes for %d edges", len(body.IDs), len(body.Relations), len(body.Edges), facet.edges)
	}
	same := func(left, right lineageFacet) bool {
		equal := func(a, b map[artifact.ID][]relationKey) bool { return maps.EqualFunc(a, b, slices.Equal) }
		return left.edges == right.edges && equal(left.parents, right.parents) && equal(left.children, right.children)
	}
	var restored lineageFacet
	if err := restored.restore(payload); err != nil {
		t.Fatal(err)
	}
	if !same(restored, facet) {
		t.Fatal("the compact body restores other lists than the facet's")
	}
	for _, key := range []relationKey{{child: ids[0], parent: ids[5], relation: 1}, {child: ids[3], parent: ids[0], relation: 2}} {
		facet.add(key)
		restored.add(key)
	}
	if !same(restored, facet) {
		t.Fatal("a restored facet files a later edge otherwise than the facet")
	}

	state := newCatalogState()
	state.lineage = facet
	root := t.TempDir()
	if err := writeProjectionCheckpoints(root, &state, 1, artifact.CommitID{}, storeHeaderBytes, hex.EncodeToString(make([]byte, sha256.Size))); err != nil {
		t.Fatal(err)
	}
	member := filepath.Join(root, checkpointDirectory, checkpointGeneration(1), "lineage"+checkpointExtension)
	document, err := os.ReadFile(member)
	if err != nil {
		t.Fatal(err)
	}
	line, _, _ := bytes.Cut(document, []byte{'\n'})
	var header checkpointHeader
	if err := strictjson.DecodeBytes(line, &header); err != nil {
		t.Fatal(err)
	}
	var edges []artifact.Lineage
	for _, child := range slices.SortedFunc(maps.Keys(facet.parents), artifact.CompareID) {
		for _, key := range facet.parents[child] {
			edges = append(edges, artifact.Lineage{Child: key.child, Parent: key.parent, Relation: key.relation})
		}
	}
	earlier, err := json.Marshal(edges)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(earlier)
	header.Version, header.PayloadDigest = initialProjectionVersion, hex.EncodeToString(digest[:])
	line, err = json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(member, append(append(line, '\n'), earlier...), storeFileMode); err != nil {
		t.Fatal(err)
	}
	loaded, _, ok, reason := loadProjectionCheckpoints(root)
	if !ok {
		t.Fatalf("a set holding lineage's version-1 body is refused: %s", reason)
	}
	if !same(loaded.lineage, facet) {
		t.Fatal("lineage's version-1 body loads other lists than the facet's")
	}
}
