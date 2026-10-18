package artifact

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
)

const (
	fixtureModelPayload = "fixture-model"
	fixtureMediaType    = "application/vnd.overgo.model"
)

// TestIDRoundTrip holds an identity to its text and its JSON. A store open
// decodes every identity it holds, so a plain quoted identity decodes in place
// without a nested JSON decode, an escaped one still decodes as JSON does, and
// a malformed one is refused.
func TestIDRoundTrip(t *testing.T) {
	id, err := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseID(id.String())
	if err != nil {
		t.Fatal(err)
	}
	if parsed != id || parsed.Kind() != KindModel {
		t.Fatalf("round trip = %v, want %v", parsed, id)
	}
	encoded, err := json.Marshal(id)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ID
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded != id {
		t.Fatalf("JSON round trip = %v, %v, want %v", decoded, err, id)
	}
	if allocations := testing.AllocsPerRun(100, func() {
		if err := decoded.UnmarshalJSON(encoded); err != nil {
			t.Fatal(err)
		}
	}); allocations > 2 || decoded != id {
		t.Fatalf("decoding an identity allocated %.0f times, decoded %v", allocations, decoded)
	}
	escaped := bytes.Replace(encoded, []byte(":"), []byte("\x5cu003a"), 1)
	if bytes.Equal(escaped, encoded) {
		t.Fatal("the escaped identity is the plain one")
	}
	decoded = ID{}
	if err := decoded.UnmarshalJSON(escaped); err != nil || decoded != id {
		t.Fatalf("escaped identity = %v, %v", decoded, err)
	}
	for _, malformed := range []string{`"model:sha256:"`, `"model:sha256:` + id.DigestHex() + `:x"`, `"` + id.String()} {
		if err := decoded.UnmarshalJSON([]byte(malformed)); err == nil {
			t.Fatalf("%s decoded", malformed)
		}
	}
}

func TestIdentityIncludesKind(t *testing.T) {
	model, err := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := IdentifyBytes(KindCheckpoint, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	if model == checkpoint || model.digest != checkpoint.digest {
		t.Fatalf("kind qualification failed: model=%s checkpoint=%s", model, checkpoint)
	}
}

func TestIdentifyReaderReportsSize(t *testing.T) {
	payload := []byte(fixtureModelPayload)
	id, size, err := Identify(KindModel, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := IdentifyBytes(KindModel, payload)
	if id != want || size != uint64(len(payload)) {
		t.Fatalf("identify = (%s, %d), want (%s, %d)", id, size, want, len(payload))
	}
}

// writerToSource is a source that can write itself out, as a file can, and
// records the largest read it was asked for.
type writerToSource struct {
	*bytes.Reader
	wroteItself bool
	largestRead int
}

func (source *writerToSource) WriteTo(writer io.Writer) (int64, error) {
	source.wroteItself = true
	return source.Reader.WriteTo(writer)
}

func (source *writerToSource) Read(buffer []byte) (int, error) {
	source.largestRead = max(source.largestRead, len(buffer))
	return source.Reader.Read(buffer)
}

// TestIdentifyReadsItsOwnBuffer holds the hash to reads of its own buffer's
// size from a source that could write itself out, as a file can through a
// 32 KiB buffer, with the identity unchanged.
func TestIdentifyReadsItsOwnBuffer(t *testing.T) {
	payload := bytes.Repeat([]byte(fixtureModelPayload), identifyBufferBytes/len(fixtureModelPayload)*3)
	source := &writerToSource{Reader: bytes.NewReader(payload)}
	id, size, err := Identify(KindModel, source)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := IdentifyBytes(KindModel, payload)
	if id != want || size != uint64(len(payload)) {
		t.Fatalf("identify = (%s, %d), want (%s, %d)", id, size, want, len(payload))
	}
	if source.wroteItself || source.largestRead != identifyBufferBytes {
		t.Fatalf("the hash read through the source's own copy (%t) with reads of at most %d bytes, want reads of %d", source.wroteItself, source.largestRead, identifyBufferBytes)
	}
}

func TestDescriptorValidation(t *testing.T) {
	id, _ := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	valid := Descriptor{ID: id, Size: uint64(len(fixtureModelPayload)), MediaType: fixtureMediaType}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	invalid := valid
	invalid.MediaType = " text/plain "
	if err := invalid.Validate(); err == nil {
		t.Fatal("accepted padded media type")
	}
}

func TestRelationJSONRoundTrip(t *testing.T) {
	encoded, err := json.Marshal(RelationQuantizedFrom)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Relation
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != RelationQuantizedFrom {
		t.Fatalf("relation = %v, want %v", decoded, RelationQuantizedFrom)
	}
}

// TestHasherMatchesIdentifyForAnyChunking holds an incremental identity to the
// one Identify gives the same bytes, however they are cut into writes.
func TestHasherMatchesIdentifyForAnyChunking(t *testing.T) {
	payload := bytes.Repeat([]byte(fixtureModelPayload), 997)
	want, wantSize, err := Identify(KindTensorSet, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{1, 3, 64, 65, 4096, len(payload), len(payload) + 1} {
		hasher, err := NewHasher(KindTensorSet)
		if err != nil {
			t.Fatal(err)
		}
		for rest := payload; len(rest) > 0; {
			size := min(cut, len(rest))
			if _, err := hasher.Write(rest[:size]); err != nil {
				t.Fatal(err)
			}
			rest = rest[size:]
		}
		if got, size := hasher.ID(); got != want || size != wantSize {
			t.Fatalf("cut %d: identity (%s, %d), want (%s, %d)", cut, got, size, want, wantSize)
		}
	}
	empty, err := NewHasher(KindTensorSet)
	if err != nil {
		t.Fatal(err)
	}
	wantEmpty, _ := IdentifyBytes(KindTensorSet, nil)
	if got, size := empty.ID(); got != wantEmpty || size != 0 {
		t.Fatalf("empty identity = (%s, %d), want (%s, 0)", got, size, wantEmpty)
	}
	if _, err := NewHasher(KindInvalid); err == nil {
		t.Fatal("a hasher accepted the invalid kind")
	}
}
