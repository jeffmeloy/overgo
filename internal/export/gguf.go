package export

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gguf"
	"overgo/internal/modelmerge"
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
	return writeInto(root, name, source, strings.ToLower(typeName), func(file string) error {
		if err := quantizeGGUF(source, file, target); err != nil {
			return err
		}
		modelfile := "FROM ./" + filepath.Base(file) + "\n"
		return os.WriteFile(filepath.Join(filepath.Dir(file), "Modelfile"), []byte(modelfile), clioptions.OutputFileMode)
	})
}

// MergeGGUF writes base + sum(scale * (fine-tuned - base)) of GGUF models
// sharing the base's tensor inventory, as an F32 GGUF, into a new directory
// named name under root, and returns the written file.
func MergeGGUF(base string, tuned []string, scales []float32, root, name string) (string, error) {
	if len(tuned) != len(scales) {
		return "", errors.New("export: every fine-tuned model needs its scale")
	}
	tasks := make([]modelmerge.GGUFTask, len(tuned))
	for index := range tuned {
		tasks[index] = modelmerge.GGUFTask{Path: tuned[index], Scale: scales[index]}
	}
	return writeInto(root, name, base, "merged", func(file string) error {
		destination, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.OutputFileMode)
		if err != nil {
			return err
		}
		if err := modelmerge.MergeGGUF(base, tasks, destination); err != nil {
			return errors.Join(err, destination.Close())
		}
		return errors.Join(destination.Sync(), destination.Close())
	})
}

// writeInto makes the new directory name under root, writes the file named
// for source and suffix into it, and leaves no directory when writing fails.
func writeInto(root, name, source, suffix string, write func(file string) error) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name {
		return "", errors.New("export: the output must be one path element")
	}
	directory := filepath.Join(root, name)
	if _, err := os.Stat(directory); err == nil {
		return "", fmt.Errorf("export: %s already exists; write under another name", directory)
	}
	if err := os.MkdirAll(directory, clioptions.OutputDirectoryMode); err != nil {
		return "", err
	}
	file := filepath.Join(directory, strings.TrimSuffix(filepath.Base(source), filepath.Ext(source))+"."+suffix+".gguf")
	if err := write(file); err != nil {
		return "", errors.Join(err, os.RemoveAll(directory))
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
