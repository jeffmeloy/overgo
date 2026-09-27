package export

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gguf"
	"overgo/internal/quant"
)

// GGUFTypes are the quantized types a GGUF export writes without an
// importance matrix.
func GGUFTypes() []string {
	var names []string
	for _, name := range quant.TypeNames() {
		if target, ok := quant.ParseType(name); ok && quant.CanQuantize(target) && !quant.RequiresImportance(target) {
			names = append(names, name)
		}
	}
	return names
}

// GGUF writes the GGUF at source into a new directory named name under root,
// quantized to typeName, with an Ollama Modelfile beside it, and returns the
// written file. Ollama reads the chat template from the GGUF's own metadata,
// so the Modelfile names the file alone. An existing directory is refused;
// a failed export leaves no directory.
func GGUF(source, root, name, typeName string) (string, error) {
	target, ok := quant.ParseType(typeName)
	if !ok || !quant.CanQuantize(target) || quant.RequiresImportance(target) {
		return "", fmt.Errorf("export: %q is not a type an export writes without an importance matrix", typeName)
	}
	name = strings.TrimSpace(name)
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name {
		return "", errors.New("export: the output must be one path element")
	}
	directory := filepath.Join(root, name)
	if _, err := os.Stat(directory); err == nil {
		return "", fmt.Errorf("export: %s already exists; export under another name", directory)
	}
	if err := os.MkdirAll(directory, clioptions.OutputDirectoryMode); err != nil {
		return "", err
	}
	file := filepath.Join(directory, strings.TrimSuffix(filepath.Base(source), filepath.Ext(source))+"."+strings.ToLower(typeName)+".gguf")
	if err := quantizeGGUF(source, file, target); err != nil {
		_ = os.RemoveAll(directory)
		return "", err
	}
	modelfile := "FROM ./" + filepath.Base(file) + "\n"
	if err := os.WriteFile(filepath.Join(directory, "Modelfile"), []byte(modelfile), clioptions.OutputFileMode); err != nil {
		_ = os.RemoveAll(directory)
		return "", err
	}
	return file, nil
}

// quantizeGGUF writes source quantized to target at file.
func quantizeGGUF(source, file string, target gguf.DType) error {
	input, err := gguf.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	destination, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.OutputFileMode)
	if err != nil {
		return err
	}
	if _, err := input.QuantizeTo(destination, target, gguf.QuantizeOptions{}); err != nil {
		return errors.Join(err, destination.Close())
	}
	return errors.Join(destination.Sync(), destination.Close())
}
