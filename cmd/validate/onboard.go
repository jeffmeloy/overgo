package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/libraryintake"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
)

// onboardCommand is the one command that takes a model directory to
// validated; the inventory names it for a directory nothing has registered.
const onboardCommand = "go run ./cmd/validate -onboard "

// conversionDirectory holds the GGUF conversions of Hugging Face directories,
// beneath the data root's checkpoints.
const conversionDirectory = "overgo-hfconvert"

// ggufExtension marks the one weights format the library validation serves.
const ggufExtension = ".gguf"

// onboarding is the one command's steps: registered names the models the
// store already ties to a directory, run executes an existing command at the
// repository root, and validate is the library's own validation of a GGUF --
// register, record and replay an exact suite, publish, activate.
type onboarding struct {
	store, checkpoints string
	registered         func(ctx context.Context, directory string) ([]InventoryModel, error)
	run                func(ctx context.Context, name string, args []string) error
	validate           func(ctx context.Context, model, projector string) (operation.Completion, error)
}

// newOnboarding binds the steps to the data roots, the repository root and
// the output the steps print to, with the store resolved once and absolute
// for every step.
func newOnboarding(root, repository string, output io.Writer) (onboarding, error) {
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return onboarding{}, err
	}
	store, err := filepath.Abs(cmp.Or(strings.TrimSpace(repository), roots.Store))
	if err != nil {
		return onboarding{}, err
	}
	return onboarding{
		store: store, checkpoints: roots.Checkpoints,
		registered: func(ctx context.Context, directory string) (tied []InventoryModel, err error) {
			repository, err := overgodb.OpenReadOnly(store)
			if err != nil {
				return nil, err
			}
			defer func() { err = errors.Join(err, repository.Close()) }()
			cells, err := registeredCells(ctx, repository)
			if err != nil {
				return nil, err
			}
			converted, err := convertedSources(ctx, repository)
			if err != nil {
				return nil, err
			}
			read, err := readModelDirectory(directory)
			if err != nil {
				return nil, err
			}
			return buildInventory([]modelDirectory{read}, cells, converted, store)[0].Models, nil
		},
		run: func(ctx context.Context, name string, args []string) error {
			return runCell(ctx, output, root, name, args)
		},
		validate: func(ctx context.Context, model, projector string) (completion operation.Completion, err error) {
			repository, err := overgodb.Open(store)
			if err != nil {
				return completion, err
			}
			defer func() { err = errors.Join(err, repository.Close()) }()
			_, execute, err := libraryintake.Validate(ctx, repository, model, projector)
			if err != nil {
				return completion, err
			}
			return execute(ctx, stageReporter{output})
		},
	}, nil
}

// onboard takes one model from its directory to validated: a Hugging Face
// directory converts to GGUF, recording the config digests that tie the
// conversion back to it; the model registers and activates through the
// library's recorded and replayed exact suite, so no golden is copied by hand;
// and the long-form guard publishes the quality record its standing is judged
// by. The one GGUF the store already ties to the directory -- by location or
// by the config its conversion recorded -- is onboarded instead of converting
// again, so a run a failed step stopped is simply run again; several are
// refused. A conversion is written only where no file is, and an untied file
// there is neither validated nor overwritten. A directory with an activated
// model is refused, since validating again would re-record its goldens: the
// inventory states its standing and the command that advances it. The first
// step that fails stops the command and is named.
func (steps onboarding) onboard(ctx context.Context, output io.Writer, directory string) error {
	model, projector, convert, err := onboardingModel(directory, filepath.Join(steps.checkpoints, conversionDirectory))
	if err != nil {
		return err
	}
	if info, err := os.Stat(directory); err == nil && info.IsDir() {
		tied, err := steps.registered(ctx, directory)
		if err != nil {
			return err
		}
		var bound []string
		for _, registered := range tied {
			if slices.ContainsFunc(registered.Cells, func(cell ModelValidation) bool { return cell.Kind != "" }) {
				return fmt.Errorf("onboard: %s is already activated as %s; validate -inventory states its standing and the command that advances it",
					directory, registered.Location)
			}
			if convert && strings.EqualFold(filepath.Ext(registered.Location), ggufExtension) && !slices.Contains(bound, registered.Location) {
				bound = append(bound, registered.Location)
			}
		}
		if len(bound) > 1 {
			return fmt.Errorf("onboard: the store ties %s to %d conversions (%s); onboard the one meant by its file",
				directory, len(bound), strings.Join(bound, ", "))
		}
		if len(bound) == 1 {
			model, convert = bound[0], false
		}
	}
	if convert {
		if _, err := os.Lstat(model); !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("onboard: the store ties %s to no conversion of %s: it may be another directory's or a partial one; move it aside and run again (%v)",
				model, directory, cmp.Or(err, fs.ErrExist))
		}
		fmt.Fprintf(output, "onboard: convert %s to %s\n", directory, model)
		if err := steps.run(ctx, "go", []string{"run", "./cmd/gguf", "convert", "-source", directory, "-output", model, "-record", steps.store}); err != nil {
			return fmt.Errorf("onboard: convert %s: %w", directory, err)
		}
	}
	fmt.Fprintf(output, "onboard: register, record, replay and activate %s\n", model)
	completion, err := steps.validate(ctx, model, projector)
	if err != nil {
		return fmt.Errorf("onboard: validate %s: %w", model, err)
	}
	name, args, err := reacquireCommand(ModelValidation{Validation: guardValidation, Location: model}, steps.store)
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "onboard: activated under run %s; long-form guard\n", completion.Run)
	if err := steps.run(ctx, name, args); err != nil {
		return fmt.Errorf("onboard: guard %s: %w", model, err)
	}
	fmt.Fprintf(output, "onboard: %s validated\n", directory)
	return nil
}

// onboardingModel resolves the GGUF a path onboards as: the file itself, the
// one model GGUF and any projector a directory holds, or the conversion a
// Hugging Face directory is written to, convert reporting that the directory
// onboards through one.
func onboardingModel(path, conversions string) (model, projector string, convert bool, err error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", "", false, err
	}
	converted, err := filepath.Glob(filepath.Join(path, "*"+ggufExtension))
	if err != nil {
		return "", "", false, err
	}
	if !info.IsDir() || len(converted) != 0 {
		model, projector, err = libraryintake.ModelFiles(path, "")
		return model, projector, false, err
	}
	if _, err := os.Stat(filepath.Join(path, modelConfigFile)); err != nil {
		return "", "", false, fmt.Errorf("onboard: %s holds neither a GGUF nor a Hugging Face %s", path, modelConfigFile)
	}
	return filepath.Join(conversions, filepath.Base(path)+ggufExtension), "", true, nil
}

// stageReporter prints the library validation's progress as the one command
// reaches each stage.
type stageReporter struct{ output io.Writer }

// OperationID is empty: the one command runs the validation outside the
// operation runtime.
func (stageReporter) OperationID() artifact.ID { return artifact.ID{} }

// Metric is not printed; the validation's evidence records what it measured.
func (stageReporter) Metric(operation.Metric) {}

// Attempt is not printed; the published gate evidence names it.
func (stageReporter) Attempt(artifact.ID) {}

// Publishing is not printed; the stage count already reaches it.
func (stageReporter) Publishing() {}

// Progress prints the stage the validation reached out of its total.
func (reporter stageReporter) Progress(completed uint64, total *uint64) {
	if total != nil {
		fmt.Fprintf(reporter.output, "onboard: validation %d/%d\n", completed, *total)
	}
}
