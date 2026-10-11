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
	write, err := quantized(source, typeName)
	if err != nil {
		return "", err
	}
	return writeInto(ctx, root, name, source, strings.ToLower(typeName), func(file string) error {
		if err := writeFile(ctx, file, write); err != nil {
			return err
		}
		modelfile := "FROM ./" + filepath.Base(file) + "\n"
		return os.WriteFile(filepath.Join(filepath.Dir(file), "Modelfile"), []byte(modelfile), clioptions.OutputFileMode)
	})
}

// GGUFSize is the size in bytes of the model file GGUF would write for source
// and typeName, from the source's header alone: nothing is written and no
// tensor is read. The Modelfile beside the model is not counted.
func GGUFSize(source, typeName string) (uint64, error) {
	return sizeOf(quantized(source, typeName))
}

// MergeGGUF writes base + sum(scale * (fine-tuned - base)) of GGUF models
// sharing the base's tensor inventory, as an F32 GGUF, into a new directory
// named name under root, and returns the written file.
func MergeGGUF(ctx context.Context, base string, tuned []string, scales []float32, root, name string) (string, error) {
	write, err := merged(base, tuned, scales)
	if err != nil {
		return "", err
	}
	return writeInto(ctx, root, name, base, "merged", func(file string) error { return writeFile(ctx, file, write) })
}

// MergeGGUFSize is the size in bytes of the file MergeGGUF would write, from
// the models' headers alone: a merge refused on those headers is refused
// here, nothing is written and no tensor is read.
func MergeGGUFSize(base string, tuned []string, scales []float32) (uint64, error) {
	return sizeOf(merged(base, tuned, scales))
}

// quantized is the writer of the GGUF at source quantized to typeName, or the
// refusal of a type an export does not write.
func quantized(source, typeName string) (func(io.Writer) error, error) {
	target, ok := quant.ParseType(typeName)
	if !ok || !quant.CanQuantize(target) || quant.RequiresImportance(target) {
		return nil, fmt.Errorf("export: %q is not a type an export writes without an importance matrix", typeName)
	}
	return func(destination io.Writer) error {
		input, err := gguf.Open(source)
		if err != nil {
			return err
		}
		defer input.Close()
		_, err = input.QuantizeTo(destination, target, gguf.QuantizeOptions{})
		return err
	}, nil
}

// merged is the writer of the merge of the fine-tuned models into base, or
// the refusal of a fine-tuned model without its scale.
func merged(base string, tuned []string, scales []float32) (func(io.Writer) error, error) {
	if len(tuned) != len(scales) {
		return nil, errors.New("export: every fine-tuned model needs its scale")
	}
	tasks := make([]modelmerge.GGUFTask, len(tuned))
	for index := range tuned {
		tasks[index] = modelmerge.GGUFTask{Path: tuned[index], Scale: scales[index]}
	}
	return func(destination io.Writer) error { return modelmerge.MergeGGUF(base, tasks, destination) }, nil
}

// writeFile creates file, which must not exist, writes it through write
// under ctx, and syncs it.
func writeFile(ctx context.Context, file string, write func(io.Writer) error) error {
	destination, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.OutputFileMode)
	if err != nil {
		return err
	}
	if err := write(contextWriter{ctx: ctx, writer: destination}); err != nil {
		return errors.Join(err, destination.Close())
	}
	return errors.Join(destination.Sync(), destination.Close())
}

// sizeOf is the bytes write writes, counted and kept nowhere; the refusal
// that came in place of a writer is passed on.
func sizeOf(write func(io.Writer) error, err error) (uint64, error) {
	if err != nil {
		return 0, err
	}
	var count sizer
	err = write(&count)
	return uint64(count), err
}

// sizer is a destination that counts what a GGUF writer gives it. The writer
// copies each tensor with io.CopyN, which hands a destination that reads for
// itself the tensor's reader bounded to the tensor's size, so a tensor is
// counted from that bound and never read.
type sizer uint64

// Write counts the header and the padding.
func (count *sizer) Write(data []byte) (int, error) {
	*count += sizer(len(data))
	return len(data), nil
}

// ReadFrom counts a bounded source by its bound, unread, and any other by
// reading it to its end.
func (count *sizer) ReadFrom(source io.Reader) (read int64, err error) {
	if bounded, ok := source.(*io.LimitedReader); ok {
		read = bounded.N
	} else {
		read, err = io.Copy(io.Discard, source)
	}
	*count += sizer(read)
	return read, err
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
