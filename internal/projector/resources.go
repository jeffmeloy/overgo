package projector

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor/reference"
)

type projectorResource interface {
	Close() error
}

func openProjectorResource[R any](
	path string,
	build func(*gguf.File) (R, error),
) (R, error) {
	var zero R
	file, err := gguf.Open(path)
	if err != nil {
		return zero, err
	}
	result, err := build(file)
	if err != nil {
		_ = file.Close()
		return zero, err
	}
	return result, nil
}

func openCatalogProjector[S, R any](
	path string,
	options OpenOptions,
	label string,
	excluded []string,
	readSpec func(*gguf.File) (S, error),
	validateCatalog func(*gguf.File, S) ([]string, error),
	build func(*gguf.File, S, *projectorCUDA) R,
) (R, error) {
	return openProjectorResource(path, func(file *gguf.File) (R, error) {
		var zero R
		spec, err := readSpec(file)
		if err != nil {
			return zero, err
		}
		catalog, err := validateCatalog(file, spec)
		if err != nil {
			return zero, err
		}
		var cuda *projectorCUDA
		if options.CUDA {
			cuda, err = openProjectorCUDA(context.Background(), file, catalog, excluded, options.DeviceOrdinal)
			if err != nil {
				return zero, fmt.Errorf("projector: initialize %s CUDA: %w", label, err)
			}
		}
		return build(file, spec, cuda), nil
	})
}

func loadProjectorHostTensor(ctx context.Context, file *gguf.File, name string) (reference.Value, error) {
	info, ok := file.Tensor(name)
	if !ok {
		return reference.Value{}, fmt.Errorf("tensor %q is unavailable", name)
	}
	return model.LoadHostTensor(ctx, file, info)
}

func loadProjectorHostTensorPair(
	ctx context.Context,
	file *gguf.File,
	first, second string,
) (reference.Value, reference.Value, error) {
	a, err := loadProjectorHostTensor(ctx, file, first)
	if err != nil {
		return reference.Value{}, reference.Value{}, err
	}
	b, err := loadProjectorHostTensor(ctx, file, second)
	return a, b, err
}

func closeProjectorResources[C projectorResource](file **gguf.File, cuda *C) error {
	var closeErrors []error
	if cuda != nil {
		closeErrors = append(closeErrors, (*cuda).Close())
		var zero C
		*cuda = zero
	}
	if file != nil && *file != nil {
		closeErrors = append(closeErrors, (*file).Close())
		*file = nil
	}
	return errors.Join(closeErrors...)
}
