package adaptiveparity

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestSnapshotImportRoundTrip(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("caller path unavailable")
	}
	path := filepath.Join(filepath.Dir(file), "..", "..", "docs", "adaptive_snapshot.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	first, err := Import(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := Export(&encoded, first); err != nil {
		t.Fatal(err)
	}
	second, err := Import(bytes.NewReader(encoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("snapshot changed across canonical export/import")
	}
	if len(first.Capabilities) < 9 {
		t.Fatalf("capabilities = %d, want representative breadth", len(first.Capabilities))
	}
	for _, capability := range first.Capabilities {
		if strings.Contains(capability.EntryPoints[0].Path, "adaptive_new") || strings.Contains(capability.EntryPoints[0].Path, `C:\`) {
			t.Fatalf("%s retains cross-repo runtime path", capability.ID)
		}
	}

	broken := bytes.Replace(raw, []byte(`"schema": 1`), []byte(`"schema": 1, "unknown": true`), 1)
	if _, err := Import(bytes.NewReader(broken)); err == nil {
		t.Fatal("unknown snapshot field accepted")
	}
}
