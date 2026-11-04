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

// OpenRunnerFromStore is OpenRunnerWithOptions over a store its caller opened read-only from the flags' repository, so
// a start that resolved other things from the store opens it once. It closes the store once the recipe is resolved,
// before the weights load, as an open of its own does.
func (flags *ModelFlags) OpenRunnerFromStore(
	ctx context.Context,
	store *overgodb.Store,
	path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	roots, err := dataroot.ResolveCurrent()
	if err == nil && (flags == nil || flags.Repository == nil) {
		err = errors.New("model recipe repository is required")
	}
	var repository string
	if err == nil {
		repository, err = flags.repositoryPath(roots)
	}
	if err != nil {
		return nil, errors.Join(err, store.Close())
	}
	return open(ctx, repository, store, roots.ResolveModelPath(path), options, false)
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
	return open(ctx, repository, nil, path, options, false)
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
	return open(ctx, repository, nil, path, options, true)
}

// open opens the runner over store, or over a read-only open of repository of its own when store is nil.
func open(
	ctx context.Context,
	repository string,
	store *overgodb.Store,
	path string,
	options inference.OpenOptions,
	served bool,
) (*inference.Runner, error) {
	if strings.TrimSpace(repository) == "" {
		return nil, errors.Join(errors.New("model recipe repository is required"), store.Close())
	}
	runner, err := openRunner(ctx, repository, store, path, options, true, served)
	if errors.Is(err, artifact.ErrContentMismatch) {
		// The memo was stale: the file changed without a new size or modification
		// time, and nothing was served. Resolve it again by hashing, under the
		// identity its bytes have now.
		fmt.Fprintf(os.Stderr, "model %s: its remembered identity does not match its bytes; hashing the file\n", path)
		return openRunner(ctx, repository, nil, path, options, false, served)
	}
	return runner, err
}

// openRunner opens the runner over store, closing it once the recipe is resolved; a nil store is opened here.
func openRunner(
	ctx context.Context,
	repository string,
	store *overgodb.Store,
	path string,
	options inference.OpenOptions,
	remembered, served bool,
) (*inference.Runner, error) {
	// Serving only READS the store (recipe resolution); a writer open here
	// starved concurrent serves and self-deadlocked the smoke lane against
	// its own child process.
	// The parts of a load name themselves to a request that keeps a ledger of them (processmeasure).
	if store == nil {
		opened := processmeasure.Stage(ctx, "store open")
		var err error
		store, err = overgodb.OpenReadOnly(repository)
		opened()
		if err != nil {
			return nil, fmt.Errorf("open model recipe repository: %w", err)
		}
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
	// A served model decodes through Strata's round where its facts are the round's (inference.OpenOptions.StrataRound),
	// and drafts through its own next-token layer by default only where drafting is faster, by the verdict the store
	// records for its model, build and card where one is (judgeServedDrafts), never at the runner's own open.
	judge := options.JudgeOwnDrafts || served
	options.StrataRound, options.JudgeOwnDrafts = options.StrataRound || served, false
	var identity modelrecipe.ProgramIdentity
	var verdicts draftVerdicts
	if resolveErr == nil && (judge || served && options.Draft == "") {
		identity, resolveErr = loaded.Identity()
	}
	if resolveErr == nil && served && options.Draft == "" {
		options.Draft, _, resolveErr = discovery.ActiveDraft(ctx, store, identity.Model, memo)
	}
	if judge {
		verdicts = readDraftVerdicts(ctx, store)
	}
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
	derivation, derived := loaded.Derived()
	runner, err := inference.OpenWithProgram(ctx, &loaded, options)
	if err == nil {
		noteModelState(path, runner.ModelState())
	}
	if err == nil && derived {
		// What the store cannot take now, another writer holding it among the causes, it takes at the next open.
		_ = registerDerived(ctx, repository, derivation)
	}
	if err != nil || !judge {
		return runner, err
	}
	if err := judgeServedDrafts(ctx, repository, runner, identity.Model, verdicts); err != nil {
		return nil, errors.Join(err, runner.Close())
	}
	return runner, nil
}

// registerDerived publishes what the records of a derived run name, the model's facts and its recipe, as a candidate and
// no more, and the identities the open already proved, so the next open claims them. The open has proved them by now.
func registerDerived(ctx context.Context, repository string, derivation modelrecipe.Derivation) error {
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	if err := modelrecipe.PublishDerivation(ctx, store, derivation); err != nil {
		return err
	}
	return discovery.RememberInventory(ctx, store, derivation.Inventory)
}

// noteModelState says on the standard error what stands in the way of a model's verification; it runs all the same.
func noteModelState(path string, state modelrecipe.ModelState) {
	switch {
	case state.Verification == modelrecipe.StateVerified:
	case state.Reason == "":
		fmt.Fprintf(os.Stderr, "model %s: %s\n", path, state.Verification)
	default:
		fmt.Fprintf(os.Stderr, "model %s: %s: %s\n", path, state.Verification, state.Reason)
	}
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
