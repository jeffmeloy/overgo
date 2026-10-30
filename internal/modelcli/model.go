// Package modelcli binds command flags and recipe storage to model opening.
package modelcli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync/atomic"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/inference"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/tensor"
)

// ModelFlags holds common model-loading flags.
type ModelFlags struct {
	DeviceOrdinal *int
	Repository    *string
	loraPaths     stringList
}

type stringList []string

// String joins adapter paths for flag help.
func (values *stringList) String() string {
	return strings.Join(*values, ",")
}

// Set appends a nonempty adapter path.
func (values *stringList) Set(value string) error {
	if strings.TrimSpace(value) == "" {
		return errors.New("LoRA path is empty")
	}
	*values = append(*values, value)
	return nil
}

// AddModelFlags registers common model-loading flags.
func AddModelFlags(flags *flag.FlagSet, loraHelp string) *ModelFlags {
	result := &ModelFlags{}
	result.DeviceOrdinal = flags.Int("device", 0, "CUDA device ordinal")
	result.Repository = flags.String("repo", "", "OvergoDB containing the active model recipe; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	flags.Var(&result.loraPaths, "lora", loraHelp)
	return result
}

// OpenRunner resolves the active identity-bound recipe before inference.
func (flags *ModelFlags) OpenRunner(
	ctx context.Context,
	path string,
) (*inference.Runner, error) {
	return flags.OpenRunnerWithOptions(ctx, path, flags.OpenOptions(tensor.UnitScale))
}

// OpenRunnerWithOptions resolves the repository and model from flags.
func (flags *ModelFlags) OpenRunnerWithOptions(
	ctx context.Context,
	path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	if flags == nil || flags.Repository == nil {
		return nil, errors.New("model recipe repository is required")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return nil, err
	}
	repository, err := flags.repositoryPath(roots)
	if err != nil {
		return nil, err
	}
	return OpenRunner(ctx, repository, roots.ResolveModelPath(path), options)
}

// RepositoryPath returns the exact store selected by model-loading flags.
func (flags *ModelFlags) RepositoryPath() (string, error) {
	if flags == nil || flags.Repository == nil {
		return "", errors.New("model recipe repository is required")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return "", err
	}
	return flags.repositoryPath(roots)
}

func (flags *ModelFlags) repositoryPath(roots dataroot.Roots) (string, error) {
	repository := strings.TrimSpace(*flags.Repository)
	if repository != "" {
		return repository, nil
	}
	if _, err := os.Stat(roots.Store); err != nil {
		return "", fmt.Errorf("model recipe repository is required: no -repo flag and no store at %s (%s)", roots.Store, roots.Source)
	}
	return roots.Store, nil
}

// OpenRunner resolves storage inputs and opens the compiled inference program.
// A weights file the store's identity memo remembers is not hashed before the
// open: its remembered identity finds the recipe, and the open proves that
// identity against the bytes it uploads, in the same pass.
func OpenRunner(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	return open(ctx, repository, path, options, false)
}

// OpenServedRunner opens the model at path as it is served: with the draft its active speculation recipe binds where
// options names none, none when the store declares none, resolved over the one open of the store the recipe is
// resolved from, so a measured request decodes through the model's draft as a server started without -draft does.
// The store opened apart for the draft took as long again as the recipe's open (0.7 s of Carbon-500M's 1.9 s load).
func OpenServedRunner(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	return open(ctx, repository, path, options, true)
}

func open(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
	served bool,
) (*inference.Runner, error) {
	if strings.TrimSpace(repository) == "" {
		return nil, errors.New("model recipe repository is required")
	}
	runner, err := openRunner(ctx, repository, path, options, true, served)
	if errors.Is(err, artifact.ErrContentMismatch) {
		// The memo was stale: the file changed without a new size or modification
		// time, and nothing was served. Resolve it again by hashing, under the
		// identity its bytes have now.
		fmt.Fprintf(os.Stderr, "model %s: its remembered identity does not match its bytes; hashing the file\n", path)
		return openRunner(ctx, repository, path, options, false, served)
	}
	return runner, err
}

func openRunner(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
	remembered, served bool,
) (*inference.Runner, error) {
	// Serving only READS the store (recipe resolution); a writer open here
	// starved concurrent serves and self-deadlocked the smoke lane against
	// its own child process.
	// The parts of a load name themselves to a request that keeps a ledger of them (processmeasure).
	opened := processmeasure.Stage(ctx, "store open")
	store, err := overgodb.OpenReadOnly(repository)
	opened()
	if err != nil {
		return nil, fmt.Errorf("open model recipe repository: %w", err)
	}
	resolved := processmeasure.Stage(ctx, "recipe resolve")
	var memo *discovery.Memo
	var claimer modelartifact.Claimer
	if remembered || served {
		memo = discovery.LoadMemo(ctx, store)
	}
	if remembered {
		claimer = memo.Claim
	}
	loaded, resolveErr := modelrecipe.ResolveClaimedGGUF(ctx, store, path, claimer)
	if resolveErr == nil && served && options.Draft == "" {
		var identity modelrecipe.ProgramIdentity
		if identity, resolveErr = loaded.Identity(); resolveErr == nil {
			options.Draft, _, resolveErr = discovery.ActiveDraft(ctx, store, identity.Model, memo)
		}
	}
	// A served model decodes through Strata's round where its facts are the round's (inference.OpenOptions.StrataRound),
	// and drafts through its own next-token layer by default only where drafting is faster (JudgeOwnDrafts).
	options.StrataRound, options.JudgeOwnDrafts = options.StrataRound || served, options.JudgeOwnDrafts || served
	closeErr := store.Close()
	resolved()
	if resolveErr != nil || closeErr != nil {
		_ = loaded.Close()
		return nil, errors.Join(resolveErr, closeErr)
	}
	if profile := ExpertProfilePath(path); options.ExpertProfile == "" {
		if _, err := os.Stat(profile); err == nil {
			options.ExpertProfile = profile
		}
	}
	return inference.OpenWithProgram(ctx, &loaded, options)
}

// ExpertProfilePath is where the expert profile of the model at path lies, beside it: a load of the model that finds
// one there fills the card from it (cmd/expert-profile writes it).
func ExpertProfilePath(path string) string { return path + ".experts.strp" }

// ModelReserve answers the memory reserve of a projector opened before its model: reserve asks the runner bind binds,
// once it opens, for room as its own allocations do, however much context it holds (a screenshot's 663 MB beside a
// 240,000-token cache was refused); nothing before then.
func ModelReserve(ctx context.Context) (reserve func(bytes uint64) error, bind func(*inference.Runner)) {
	var served atomic.Pointer[inference.Runner]
	return func(bytes uint64) error { return served.Load().ReserveDeviceMemory(ctx, bytes) }, served.Store
}

// ResolveModel names the model at path as the store resolves it, through its identity memo and without loading its
// weights, with its inference recipe's runtime description: the model a projector opened before it binds to, and the
// placement the projector follows, so a hybrid model's expert cache takes the device memory the projector leaves, as
// the reference engine keeps room for its image encoder.
func ResolveModel(ctx context.Context, store *overgodb.Store, path string) (artifact.ID, modelrecipe.RuntimeDescription, error) {
	loaded, err := modelrecipe.ResolveClaimedGGUF(ctx, store, path, discovery.LoadMemo(ctx, store).Claim)
	if err != nil {
		return artifact.ID{}, modelrecipe.RuntimeDescription{}, err
	}
	identity, err := loaded.Identity()
	description, describeErr := loaded.Describe()
	return identity.Model, description, errors.Join(err, describeErr, loaded.Close())
}

// OpenOptions returns inference model-loading options.
func (flags *ModelFlags) OpenOptions(loraScale float32) inference.OpenOptions {
	return BuildOpenOptions(*flags.DeviceOrdinal, flags.loraPaths, loraScale)
}

// LoRAPaths returns a copy of the adapter paths.
func (flags *ModelFlags) LoRAPaths() []string {
	return slices.Clone(flags.loraPaths)
}

// BuildOpenOptions builds common inference open options.
func BuildOpenOptions(device int, loraPaths []string, loraScale float32) inference.OpenOptions {
	adapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		adapters[index] = inference.LoRAConfig{Path: path, Scale: loraScale}
	}
	return inference.OpenOptions{
		DeviceOrdinal: device, LoRAAdapters: adapters,
	}
}
