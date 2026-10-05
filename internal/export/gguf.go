package export

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/fsatomic"
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
func GGUF(ctx context.Context, source, root, name, typeName string) (string, error) {
	target, ok := quant.ParseType(typeName)
	if !ok || !quant.CanQuantize(target) || quant.RequiresImportance(target) {
		return "", fmt.Errorf("export: %q is not a type an export writes without an importance matrix", typeName)
	}
	return writeInto(ctx, root, name, source, strings.ToLower(typeName), func(file string) error {
		if err := quantizeGGUF(ctx, source, file, target); err != nil {
			return err
		}
		modelfile := "FROM ./" + filepath.Base(file) + "\n"
		return os.WriteFile(filepath.Join(filepath.Dir(file), "Modelfile"), []byte(modelfile), clioptions.OutputFileMode)
	})
}

// MergeGGUF writes base + sum(scale * (fine-tuned - base)) of GGUF models
// sharing the base's tensor inventory, as an F32 GGUF, into a new directory
// named name under root, and returns the written file.
func MergeGGUF(ctx context.Context, base string, tuned []string, scales []float32, root, name string) (string, error) {
	if len(tuned) != len(scales) {
		return "", errors.New("export: every fine-tuned model needs its scale")
	}
	tasks := make([]modelmerge.GGUFTask, len(tuned))
	for index := range tuned {
		tasks[index] = modelmerge.GGUFTask{Path: tuned[index], Scale: scales[index]}
	}
	return writeInto(ctx, root, name, base, "merged", func(file string) error {
		destination, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.OutputFileMode)
		if err != nil {
			return err
		}
		if err := modelmerge.MergeGGUF(base, tasks, contextWriter{ctx: ctx, writer: destination}); err != nil {
			return errors.Join(err, destination.Close())
		}
		return errors.Join(destination.Sync(), destination.Close())
	})
}

// writeInto makes the new directory name under root, writes the file named
// for source and suffix into it, and leaves no directory when writing fails:
// the directory is staged and moved into place whole, so two writers of one
// name cannot remove each other's output.
func writeInto(ctx context.Context, root, name, source, suffix string, write func(file string) error) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	name = strings.TrimSpace(name)
	base := strings.TrimSuffix(filepath.Base(source), filepath.Ext(source)) + "." + suffix + ".gguf"
	directory, err := fsatomic.PublishDirectory(root, name, clioptions.OutputDirectoryMode, func(stage string) error {
		return write(filepath.Join(stage, base))
	})
	if errors.Is(err, fs.ErrExist) {
		return "", fmt.Errorf("export: %s already exists; write under another name", filepath.Join(root, name))
	}
	if err != nil {
		return "", err
	}
	return filepath.Join(directory, base), nil
}

// contextWriter stops a write the context has cancelled: a writer reaches each
// tensor in turn, so a cancelled export or merge ends at the next write, and its
// staged directory is removed.
type contextWriter struct {
	ctx    context.Context
	writer io.Writer
}

// Write passes the bytes on unless the context is done, and then writes none.
func (w contextWriter) Write(data []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	return w.writer.Write(data)
}

// quantizeGGUF writes source quantized to target at file.
func quantizeGGUF(ctx context.Context, source, file string, target gguf.DType) error {
	input, err := gguf.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	destination, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.OutputFileMode)
	if err != nil {
		return err
	}
	if _, err := input.QuantizeTo(contextWriter{ctx: ctx, writer: destination}, target, gguf.QuantizeOptions{}); err != nil {
		return errors.Join(err, destination.Close())
	}
	return errors.Join(destination.Sync(), destination.Close())
}
