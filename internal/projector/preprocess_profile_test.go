package projector

import "testing"

func TestMediaPreprocessProfileCatalog(t *testing.T) {
	first, found, err := catalogMediaPreprocessProfile(qwen2VLProjectorType)
	if err != nil || !found {
		t.Fatalf("first profile = (%+v, %t, %v)", first, found, err)
	}
	second, found, err := catalogMediaPreprocessProfile(qwen3VLProjectorType)
	if err != nil || !found || second.ID != first.ID {
		t.Fatalf("second profile = (%+v, %t, %v), want %s", second, found, err, first.ID)
	}
	if _, err := first.Content(); err != nil {
		t.Fatal(err)
	}
	if _, found, err := catalogMediaPreprocessProfile("unbound"); err != nil || found {
		t.Fatalf("unbound profile = (%t, %v)", found, err)
	}
}
