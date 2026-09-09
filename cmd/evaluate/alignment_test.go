package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestAlignmentManifestRefusesBeforeStoreMutation(t *testing.T) {
	t.Run("alignment", func(t *testing.T) { verifySpeechManifestAdmission(t, evaluateAlignmentManifest) })
	t.Run("diarization", func(t *testing.T) { verifySpeechManifestAdmission(t, evaluateDiarizationManifest) })
}

func verifySpeechManifestAdmission(t *testing.T, evaluate func(context.Context, string, string) error) {
	for _, data := range []string{`{}`, `{"unexpected":true}`, `{"inputs":[{}]}`} {
		t.Run(data, func(t *testing.T) {
			root := t.TempDir()
			manifest := filepath.Join(root, "manifest.json")
			if err := os.WriteFile(manifest, []byte(data), 0o600); err != nil {
				t.Fatal(err)
			}
			store := filepath.Join(root, "absent-store")
			if err := evaluate(t.Context(), store, manifest); err == nil {
				t.Fatal("invalid manifest admitted")
			}
			if _, err := os.Stat(store); !os.IsNotExist(err) {
				t.Fatalf("invalid manifest touched store: %v", err)
			}
		})
	}
}
