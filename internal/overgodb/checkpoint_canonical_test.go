package overgodb

import (
	"crypto/sha256"
	"fmt"
	"io"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestCheckpointCanonicalAtTheWriter holds the proof of canonical form to
// the one place a set is written: a state whose checkpoint would not restore
// is refused before any member of it is published, and a state that does is
// published as a set an open loads.
func TestCheckpointCanonicalAtTheWriter(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	copyScaleCorpus(t, root)
	sound, anchor, loaded, reason := loadProjectionCheckpoints(root)
	if !loaded {
		t.Fatalf("corpus set refused: %s", reason)
	}
	publish := func(state *catalogState) (string, error) {
		target := t.TempDir()
		digest := fmt.Sprintf("%x", anchor.digest)
		return target, writeProjectionCheckpoints(target, state, anchor.sequence, anchor.head, anchor.offset, digest)
	}

	target, err := publish(&sound)
	if err != nil {
		t.Fatal(err)
	}
	if republished, _, loaded, reason := loadProjectionCheckpoints(target); !loaded || catalogDigest(t, republished) != catalogDigest(t, sound) {
		t.Fatalf("a proved set does not load as the state it was written from: %s", reason)
	}

	id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("never durable"))
	if err != nil {
		t.Fatal(err)
	}
	sound.contents.set(id, contentLocator{size: 1, blob: true}, 0)
	target, err = publish(&sound)
	if err == nil || !strings.Contains(err.Error(), "contents") {
		t.Fatalf("publish = %v, want the contents member refused", err)
	}
	if published, _ := filepath.Glob(filepath.Join(target, checkpointDirectory, "*", "contents"+checkpointExtension)); len(published) != 0 {
		t.Fatalf("the refused member was published: %v", published)
	}
}

// checkpointShapes records, per projection, the version its checkpoint is
// written under and a digest of the shape of the entries in it. An open
// trusts a member whose header names the version this reader speaks, so a
// changed shape has to advance that version: when this table disagrees,
// advance the projection's version constant and record the new pair.
var checkpointShapes = map[string]struct {
	entry reflect.Type
	shape string
}{
	"artifacts": {reflect.TypeFor[artifactCheckpointEntry](), "v1 bf6f310aa20dc28e1c29ad0d327594b4"},
	"contents":  {reflect.TypeFor[contentCheckpointEntry](), "v3 04c2b1e6800c213840721b55884bca71"},
	"lineage":   {reflect.TypeFor[artifact.Lineage](), "v1 cb0b7aab83b9d910e2fc42021d1bb769"},
	"causality": {reflect.TypeFor[artifact.CausalLink](), "v1 d3718a8aa40b7c4dff90a8d637a24dce"},
	"locations": {reflect.TypeFor[artifact.Location](), "v1 bbd2efbaa82a9c37bcd510cd0952a375"},
	"aliases":   {reflect.TypeFor[snapshotAlias](), "v1 bf35d137cdaaaaf5d6306bd114ce4881"},
	"commits":   {reflect.TypeFor[commitCheckpointEntry](), "v2 cec808afb26ddee44c2517732f405dc8"},
}

func TestCheckpointShapeIsVersioned(t *testing.T) {
	t.Parallel()
	for _, registered := range projections(new(catalogState)) {
		recorded, ok := checkpointShapes[registered.name]
		if !ok {
			t.Errorf("projection %s records no checkpoint shape", registered.name)
			continue
		}
		hash := sha256.New()
		describeShape(hash, recorded.entry, map[reflect.Type]bool{})
		if shape := fmt.Sprintf("v%d %.16x", registered.version, hash.Sum(nil)); shape != recorded.shape {
			t.Errorf("projection %s checkpoints as %q, recorded %q: a changed shape advances the projection's version", registered.name, shape, recorded.shape)
		}
	}
}

// describeShape writes what decides how a value of the type is encoded: its
// kind, and for a struct every field's name, tag and type in turn.
func describeShape(out io.Writer, shape reflect.Type, seen map[reflect.Type]bool) {
	fmt.Fprintf(out, "%s %s;", shape.Kind(), shape.String())
	switch shape.Kind() {
	case reflect.Struct:
		if seen[shape] {
			return
		}
		seen[shape] = true
		for field := range shape.Fields() {
			fmt.Fprintf(out, "%s `%s` ", field.Name, field.Tag)
			describeShape(out, field.Type, seen)
		}
	case reflect.Map:
		describeShape(out, shape.Key(), seen)
		describeShape(out, shape.Elem(), seen)
	case reflect.Slice, reflect.Array, reflect.Pointer:
		describeShape(out, shape.Elem(), seen)
	}
}
