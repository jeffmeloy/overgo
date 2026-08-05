package hfrepo

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenReadsIdentityCompanionsAndTensors(t *testing.T) {
	directory := t.TempDir()
	writeFile(t, filepath.Join(directory, "config.json"), `{
  "model_type":"unified",
  "architectures":["UnifiedForGeneration"],
  "text_config":{"model_type":"decoder"},
  "unknown_field":true
}`)
	writeFile(t, filepath.Join(directory, "tokenizer.json"), `{}`)
	writeFile(t, filepath.Join(directory, "chat_template.jinja"), `{{ messages }}`)
	writeSingleTensor(t, filepath.Join(directory, "model.safetensors"))

	repository, err := Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	if repository.Identity.ModelType != "unified" || repository.Identity.TextModelType != "decoder" ||
		len(repository.Identity.Architectures) != 1 || repository.Identity.Architectures[0] != "UnifiedForGeneration" {
		t.Fatalf("identity = %+v", repository.Identity)
	}
	if len(repository.Companions) != 2 || len(repository.Tensors.Tensors) != 1 {
		t.Fatalf("companions = %v, tensors = %d", repository.Companions, len(repository.Tensors.Tensors))
	}
}

func TestOpenRejectsTrailingConfig(t *testing.T) {
	directory := t.TempDir()
	writeFile(t, filepath.Join(directory, "config.json"), `{} {}`)
	writeSingleTensor(t, filepath.Join(directory, "model.safetensors"))
	if _, err := Open(directory); err == nil {
		t.Fatal("trailing config accepted")
	}
}

func writeSingleTensor(t *testing.T, path string) {
	t.Helper()
	header := `{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`
	data := make([]byte, 8, 9+len(header))
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 1)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeFile(t *testing.T, path string, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
}
