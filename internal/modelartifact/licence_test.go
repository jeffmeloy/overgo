package modelartifact

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/testcheck"
)

// TestAModelWithoutALicenceIsServed holds registration to the owner's rule of 2026-11-08: no model is ever
// distributed, so a licence binds only a model's local use, recorded where the model ships one. Registration refused
// a declaration with no licence identifier. A model directory that ships no licence registers: its batch holds no
// licence content and its manifest depends on the declaration alone.
func TestAModelWithoutALicenceIsServed(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	directory := filepath.Join(root, "unlicensed")
	testcheck.NoError(t, os.Mkdir(directory, 0o700))
	header := `{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	weights := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	weights = append(append(weights, header...), 0, 0, 128, 63)
	for name, data := range map[string][]byte{"config.json": []byte(`{"model_type":"fixture"}`), "tokenizer.json": []byte(`{"version":"1.0"}`), "model.safetensors": weights} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), data, 0o600))
	}
	git := func(arguments ...string) string {
		command := exec.CommandContext(t.Context(), "git", append([]string{"-C", directory}, arguments...)...)
		command.Env = gitauthority.ReaderEnvironment()
		return strings.TrimSpace(string(testcheck.Must(command.CombinedOutput())(t)))
	}
	git("init", "-q")
	git("add", ".")
	git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--no-gpg-sign", "-qm", "fixture")
	git("remote", "add", "origin", "https://example.invalid/unlicensed")
	inventory := testcheck.Must(FromHFPath(directory))(t)
	declaration := []map[string]any{{
		"directory": "unlicensed", "model": inventory.Manifest.ID, "tensor_inventory": inventory.TensorInventory.ID,
		"source_repository": "https://example.invalid/unlicensed", "source_commit": git("rev-parse", "HEAD"),
	}}
	path := filepath.Join(t.TempDir(), "registration.json")
	testcheck.NoError(t, os.WriteFile(path, testcheck.Must(json.Marshal(declaration))(t), 0o600))
	batch := testcheck.Must(PrepareRegistration(t.Context(), root, path))(t)
	specification := batch.Contents[0].Descriptor.ID
	for _, content := range batch.Contents[1:] {
		if content.Descriptor.MediaType == "text/plain" || content.Descriptor.MediaType == "text/markdown" {
			t.Fatalf("a model with no licence registered licence content %s", content.Descriptor.ID)
		}
	}
	var parents []artifact.ID
	for _, lineage := range batch.Lineage {
		if lineage.Child == inventory.Manifest.ID && lineage.Relation == artifact.RelationDependsOn {
			parents = append(parents, lineage.Parent)
		}
	}
	if !slices.Contains(parents, specification) || len(parents) != 1 {
		t.Fatalf("the unlicensed model depends on %v, want the declaration %s alone", parents, specification)
	}
}
