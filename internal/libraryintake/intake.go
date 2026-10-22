// Package libraryintake is the model intake the library page drives: a
// downloaded model registers with its facts, its candidate recipe and any
// projector beside it, and validates by recording and replaying an exact
// suite whose evidence activates the recipe. The HTTP package receives
// these entry points from the assembly root and names no intake package.
package libraryintake

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
)

// draftAcceptanceFloor is the least share of its drafts a speculative replay of the validation prompts keeps for the
// draft beside a model to activate: the trunk keeps every token it would have decoded whatever the draft proposes,
// so the share measures whether the draft is the model's; Flash-Next's MTP layer keeps about 0.6, another model's
// draft almost none.
const draftAcceptanceFloor = 0.3

// refusal marks an intake error the request itself caused (an artifact the
// store cannot admit), which a route answers as a refusal rather than a
// failure; the HTTP package recognises it by the LibraryRefused method
// alone, and the embedded cause speaks for itself.
type refusal struct{ error }

// LibraryRefused reports the refusal; one without a cause is no refusal.
func (r refusal) LibraryRefused() bool { return r.error != nil }

// ModelFiles resolves the model file a register or validate names and the
// projector beside it: a GGUF file, or a downloaded directory holding
// exactly one model GGUF and at most one projector GGUF, each sorted by
// what its own metadata declares. A projector named explicitly takes
// precedence over one found beside the model.
func ModelFiles(path, explicitProjector string) (model, projectorPath string, err error) {
	path = strings.TrimSpace(path)
	info, err := os.Stat(path)
	if err != nil {
		return "", "", err
	}
	candidates := []string{path}
	if info.IsDir() {
		if candidates, err = filepath.Glob(filepath.Join(path, "*.gguf")); err != nil {
			return "", "", err
		}
	}
	var projectors []string
	var models []string
	for _, candidate := range candidates {
		// A draft beside the model is the model's draft (draftBeside), not a model of its own.
		if draftFile(candidate) {
			continue
		}
		describes, err := projector.DescribesProjection(candidate)
		if err != nil {
			return "", "", fmt.Errorf("library: %s is not a GGUF the store can read: %w", candidate, err)
		}
		if describes {
			projectors = append(projectors, candidate)
		} else {
			models = append(models, candidate)
		}
	}
	if len(models) != 1 {
		return "", "", fmt.Errorf("library: %s holds %d model GGUF files; name the file to register", path, len(models))
	}
	if len(projectors) > 1 {
		return "", "", fmt.Errorf("library: %s holds %d projector GGUF files; name the projector to register", path, len(projectors))
	}
	if explicitProjector = strings.TrimSpace(explicitProjector); explicitProjector != "" {
		if describes, err := projector.DescribesProjection(explicitProjector); err != nil || !describes {
			return "", "", fmt.Errorf("library: %s does not declare a projector", explicitProjector)
		}
		return models[0], explicitProjector, nil
	}
	if len(projectors) == 1 {
		return models[0], projectors[0], nil
	}
	return models[0], "", nil
}

// draftFile reports whether a GGUF declares a draft layer; a file the store cannot read declares none.
func draftFile(path string) bool {
	file, err := gguf.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	return model.DescribesHyperDraft(file)
}

// draftBeside finds the draft layer GGUF in a model file's directory, the draft its decoding speculates through, or
// none; two drafts are refused, since the model's draft would be a guess.
func draftBeside(modelPath string) (string, error) {
	candidates, err := filepath.Glob(filepath.Join(filepath.Dir(modelPath), "*.gguf"))
	if err != nil {
		return "", err
	}
	var drafts []string
	for _, candidate := range candidates {
		if !strings.EqualFold(filepath.Clean(candidate), filepath.Clean(modelPath)) && draftFile(candidate) {
			drafts = append(drafts, candidate)
		}
	}
	if len(drafts) > 1 {
		return "", fmt.Errorf("library: %s holds %d draft GGUF files beside the model", filepath.Dir(modelPath), len(drafts))
	}
	if len(drafts) == 0 {
		return "", nil
	}
	return drafts[0], nil
}

// Register publishes the model's facts and candidate recipe (known to the
// store, not yet servable) and a projector with it as the model's
// projection candidate; the receipt names what the store now holds.
func Register(ctx context.Context, repository *overgodb.Store, path, projectorPath string) (map[string]any, error) {
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, repository, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		return nil, refusal{err}
	}
	if err := modelintake.RegisterCandidate(context.WithoutCancel(ctx), repository, candidate); err != nil {
		return nil, err
	}
	receipt := map[string]any{
		"path": path, "model": candidate.Inventory.Manifest.ID,
		"definition": candidate.Resolved.Document.ID, "recipe": candidate.Definition.ID,
	}
	// A projector beside the model registers as the model's projection
	// candidate: the model's image, audio or video input, servable once
	// validated with the model.
	if projectorPath != "" {
		projection, err := modelintake.PrepareProjectionCandidate(ctx, repository, path, projectorPath)
		if err != nil {
			return nil, refusal{err}
		}
		if err := modelintake.RegisterProjectionCandidate(context.WithoutCancel(ctx), repository, projection); err != nil {
			return nil, err
		}
		receipt["projector"], receipt["projection"], receipt["media"] = projectorPath, projection.Definition.ID, projection.Media
	}
	// A draft beside the model registers as the model's speculation candidate, servable once validated with the model.
	draftPath, err := draftBeside(path)
	if err != nil {
		return nil, refusal{err}
	}
	if draftPath != "" {
		draft, err := modelintake.PrepareDraftCandidate(candidate.Inventory.Manifest.ID, draftPath)
		if err != nil {
			return nil, refusal{err}
		}
		if err := modelintake.RegisterDraftCandidate(context.WithoutCancel(ctx), repository, draft); err != nil {
			return nil, err
		}
		receipt["draft"], receipt["speculation"] = draftPath, draft.Definition.ID
	}
	return receipt, nil
}

// validationDocument holds the prompts a validation records from a freshly
// registered model's greedy output and replays exactly before activating it.
//
//go:embed validation.json
var validationDocument []byte

type validationPolicy struct {
	Schema    int      `json:"schema"`
	Doc       string   `json:"doc,omitzero"`
	Prompts   []string `json:"prompts"`
	MaxTokens int      `json:"max_tokens"`
}

// Validate prepares text validation of a model under
// the clean source revision and returns the candidate recipe it runs under
// with the operation body that registers, records, replays, publishes and
// activates, under the embedded prompts and token bound. A
// projector beside the model registers as its projection candidate; the
// text validation runs on the model alone, and the projection activates
// once a media suite executes.
func Validate(ctx context.Context, repository *overgodb.Store, path, projectorPath string) (artifact.ID, operation.Executor, error) {
	var policy validationPolicy
	if err := strictjson.DecodeBytes(validationDocument, &policy); err != nil {
		return artifact.ID{}, nil, err
	}
	if len(policy.Prompts) == 0 || policy.MaxTokens <= 0 {
		return artifact.ID{}, nil, errors.New("library: the validation policy is incomplete")
	}
	projectorPath = strings.TrimSpace(projectorPath)
	if projectorPath != "" {
		if describes, err := projector.DescribesProjection(projectorPath); err != nil || !describes {
			return artifact.ID{}, nil, refusal{fmt.Errorf("library: %s does not declare a projector", projectorPath)}
		}
	}
	revision, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return artifact.ID{}, nil, refusal{err}
	}
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, repository, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		return artifact.ID{}, nil, refusal{err}
	}
	execute := func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return validate(ctx, repository, reporter, path, projectorPath, candidate, revision, policy.Prompts, policy.MaxTokens)
	}
	return candidate.Definition.ID, execute, nil
}

// ValidateDraft validates the draft beside a model the library has activated, as its validation validates a newly
// registered model's draft: it reads the model through its active recipe and records nothing of the model's own, so
// adding a draft re-records no golden. It returns the draft's verification gate and whether the draft activated.
func ValidateDraft(ctx context.Context, repository *overgodb.Store, path string) (artifact.ID, bool, error) {
	var policy validationPolicy
	if err := strictjson.DecodeBytes(validationDocument, &policy); err != nil {
		return artifact.ID{}, false, err
	}
	revision, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return artifact.ID{}, false, err
	}
	draftPath, err := draftBeside(path)
	if err == nil && draftPath == "" {
		err = fmt.Errorf("library: %s has no draft GGUF beside it", path)
	}
	if err != nil {
		return artifact.ID{}, false, err
	}
	return validateDraft(ctx, repository, path, draftPath, revision, policy.Prompts, policy.MaxTokens)
}

// validateDraft registers a draft beside an activated model, replays the validation prompts speculatively through it,
// and activates the speculation recipe when the trunk keeps at least draftAcceptanceFloor of the drafts; it returns
// the draft's verification gate. A draft that keeps fewer records its failure and stays a candidate: the model
// serves, decoding a token a step.
func validateDraft(ctx context.Context, repository *overgodb.Store, path, draftPath, revision string, prompts []string, maxTokens int) (artifact.ID, bool, error) {
	loaded, err := modelrecipe.ResolveClaimedGGUF(ctx, repository, path, discovery.LoadMemo(ctx, repository).Claim)
	if err != nil {
		return artifact.ID{}, false, err
	}
	identity, err := loaded.Identity()
	if err != nil {
		return artifact.ID{}, false, errors.Join(err, loaded.Close())
	}
	draft, err := modelintake.PrepareDraftCandidate(identity.Model, draftPath)
	if err == nil {
		err = modelintake.RegisterDraftCandidate(ctx, repository, draft)
	}
	if err != nil {
		return artifact.ID{}, false, errors.Join(err, loaded.Close())
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{Draft: draftPath})
	if err != nil {
		return artifact.ID{}, false, err
	}
	defer runner.Close()
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		return artifact.ID{}, false, err
	}
	started := time.Now()
	drafted, kept := 0, 0
	for _, prompt := range prompts {
		if _, _, err := runner.Generate(ctx, prompt, inference.GenerateOptions{
			MaxNewTokens: maxTokens, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true,
			OnDraftSpan: func(spanDrafted, spanKept int) { drafted, kept = drafted+spanDrafted, kept+spanKept },
		}); err != nil {
			return artifact.ID{}, false, err
		}
	}
	evidence := fmt.Sprintf("contract=draft-acceptance;floor=%g;drafted=%d;kept=%d;prompts=%d", draftAcceptanceFloor, drafted, kept, len(prompts))
	if drafted == 0 || float64(kept) < draftAcceptanceFloor*float64(drafted) {
		failure, err := modelintake.PublishFailure(ctx, repository, draft.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence, "draft-acceptance")
		return failure.Gate, false, err
	}
	verification, err := modelintake.PublishVerification(ctx, repository, draft.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence)
	if err != nil {
		return artifact.ID{}, false, err
	}
	err = modelrecipe.ActivateCapability(ctx, repository, draft.Definition, verification, recipe.EvidenceVerified,
		"validated from the library: the trunk kept its share of the draft's tokens on the validation prompts")
	return verification.Gate, err == nil, err
}

// validate is the validation operation: register, record, replay, publish,
// activate, each a progress step the strip shows.
func validate(ctx context.Context, repository *overgodb.Store, reporter operation.Reporter, path, projectorPath string, candidate modelintake.Candidate, revision string, prompts []string, maxTokens int) (operation.Completion, error) {
	stages := []string{"register", "record", "replay", "publish", "activate"}
	total := uint64(len(stages))
	progress := func(stage string) { reporter.Progress(uint64(slices.Index(stages, stage)), &total) }
	progress("register")
	if err := modelintake.RegisterCandidate(ctx, repository, candidate); err != nil {
		return operation.Completion{}, err
	}
	if projectorPath != "" {
		projection, err := modelintake.PrepareProjectionCandidate(ctx, repository, path, projectorPath)
		if err != nil {
			return operation.Completion{}, err
		}
		if err := modelintake.RegisterProjectionCandidate(ctx, repository, projection); err != nil {
			return operation.Completion{}, err
		}
	}
	loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved, candidate.Inventory.Claim)
	if err != nil {
		return operation.Completion{}, err
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return operation.Completion{}, err
	}
	defer runner.Close()
	progress("record")
	started := time.Now()
	suite, err := modelintake.RecordExactSuite(ctx, runner, "library/"+filepath.Base(path), prompts, maxTokens)
	if err != nil {
		return operation.Completion{}, err
	}
	exactPlan, err := evaluation.CompileExact(suite)
	if err != nil {
		return operation.Completion{}, err
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		return operation.Completion{}, err
	}
	evaluationPlan, err := evaluation.BindExact(exactPlan, evaluation.ExactAuthorities{
		ModelDefinition: candidate.Resolved.Document.ID, RuntimeRecipe: candidate.Definition.ID,
		CodeCommit: revision, Environment: environment.ID,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return operation.Completion{}, err
	}
	progress("replay")
	reportID, replayErr := evaluation.EvaluateExactSharded(ctx, repository, runner, exactPlan, evaluationPlan, nil)
	progress("publish")
	if replayErr != nil {
		evidence := "contract=exact; plan=" + evaluationPlan.Identity().String() + "; " + strings.ReplaceAll(replayErr.Error(), "\n", " ")
		verification, publishErr := modelintake.PublishFailure(ctx, repository, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence, "exact-mismatch")
		if publishErr != nil {
			return operation.Completion{}, errors.Join(replayErr, publishErr)
		}
		return operation.Completion{Run: verification.Run, Outputs: []artifact.ID{verification.Gate}}, replayErr
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", evaluationPlan.Identity(), reportID, len(suite.Cases))
	verification, err := modelintake.PublishVerification(ctx, repository, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence)
	if err != nil {
		return operation.Completion{}, err
	}
	progress("activate")
	if err := modelrecipe.ActivateCapability(ctx, repository, candidate.Definition, verification, recipe.EvidenceVerified,
		"validated from the library: recorded and replayed an exact suite"); err != nil {
		return operation.Completion{}, err
	}
	outputs := []artifact.ID{verification.Gate, reportID}
	if err := runner.Close(); err != nil {
		return operation.Completion{}, err
	}
	// A draft beside the model validates once the model is active, through its active recipe.
	draftPath, err := draftBeside(path)
	if err != nil {
		return operation.Completion{}, err
	}
	if draftPath != "" {
		draftGate, _, err := validateDraft(ctx, repository, path, draftPath, revision, prompts, maxTokens)
		if err != nil {
			return operation.Completion{}, err
		}
		outputs = append(outputs, draftGate)
	}
	reporter.Publishing()
	reporter.Progress(total, &total)
	return operation.Completion{Run: verification.Run, Outputs: outputs}, nil
}
