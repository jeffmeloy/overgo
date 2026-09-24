package gate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/gosource"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

const modernCensusCheckName = "modern-census"

// One immutable census payload; retry evidence holds references, never copies.
var modernCensusContract = artifact.JSONContract(artifact.KindProfile, repoanalysis.ModernGoComputationSchema)

type modernGoInput struct {
	ID           artifact.ID
	CandidateKey artifact.ID
	BaseKey      artifact.ID
	Source       repoanalysis.SourceSnapshot
	Base         repoanalysis.SourceSnapshot
	Selection    gosource.BuildSelection
	TargetGo     string
	candidate    func() (repoanalysis.ModernGoCensus, error)
}

type modernGoReferences struct {
	Candidate    artifact.ID `json:"candidate"`
	Base         artifact.ID `json:"base"`
	CandidateKey artifact.ID `json:"candidate_key,omitzero"`
	Prior        artifact.ID `json:"prior,omitzero"`
}

// Preparation binds computation inputs. Policy, paths and current authority
// remain inputs of the live admission, not permission carried by a census.
func (g *gateContext) prepareModernGoInput() (*modernGoInput, error) {
	if g.modernInput != nil {
		return g.modernInput, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return nil, err
	}
	base, err := g.baseSnapshot(snapshot)
	if err != nil {
		return nil, err
	}
	baseline, err := repoanalysis.LoadModernGoBaseline(filepath.Join(g.sourceRoot(), repoanalysis.ModernGoBaselineFile))
	if err != nil {
		return nil, err
	}
	selection, err := gosource.HostBuildSelection(g.sourceRoot(), "./cmd/...", "./internal/...")
	if err != nil {
		return nil, err
	}
	module, err := fingerprintPhaseInputs(g.sourceRoot(), modernCensusCheckName, []string{"go.mod", "go.sum"})
	if err != nil {
		return nil, err
	}
	compiler, err := command(g.sourceRoot(), "go", "tool", "compile", "-V=full")
	if err != nil {
		return nil, err
	}
	build, err := command(g.sourceRoot(), "go", "env", "GOROOT", "GOPATH", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN")
	if err != nil {
		return nil, err
	}
	input := &modernGoInput{Source: snapshot, Base: base, Selection: selection, TargetGo: baseline.TargetGo}
	producer, err := artifact.JSONID(artifact.KindProfile, struct {
		Schema, Source, Context, TargetGo, Catalog, Compiler, Build string
		Module                                                      artifact.ID
		Environment                                                 artifact.ID `json:"environment,omitzero"`
		Files                                                       map[string]bool
		Packages                                                    map[string]string
	}{modernCensusContract.Schema, snapshot.Identity(), selection.Context, baseline.TargetGo,
		repoanalysis.ModernGoCatalogSHA256, compiler, build, module, g.environment.ID, selection.Files, selection.Packages})
	if err != nil {
		return nil, err
	}
	keys := []*artifact.ID{&input.CandidateKey, &input.BaseKey}
	for index, source := range []repoanalysis.SourceSnapshot{snapshot, base} {
		*keys[index], err = artifact.JSONID(artifact.KindProfile, struct {
			Producer artifact.ID
			Files    []repoanalysis.GoFile
		}{producer, source.Files})
		if err != nil {
			return nil, err
		}
	}
	input.ID, err = artifact.JSONID(artifact.KindProfile, []artifact.ID{input.CandidateKey, input.BaseKey})
	if err != nil {
		return nil, err
	}
	g.modernInput = input
	input.candidate = sync.OnceValues(func() (repoanalysis.ModernGoCensus, error) {
		return repoanalysis.ModernGoCensusSnapshot(input.Source, input.Selection, input.TargetGo)
	})
	return input, nil
}

// Rebind against the actual source root, retaining only an exactly matching
// candidate computation. Generated policy is still read at each admission.
func (g *gateContext) refreshModernGoInput() (*modernGoInput, error) {
	current := gateContext{repo: g.repo, candidateRoot: g.candidateRoot, environment: g.environment}
	input, err := current.prepareModernGoInput()
	if err != nil {
		return nil, err
	}
	if g.modernInput != nil && g.modernInput.CandidateKey == input.CandidateKey {
		input.candidate = g.modernInput.candidate
	}
	g.modernInput = input
	return input, nil
}

func (g *gateContext) computeModernGo(ctx context.Context, _ automationcheck.Invocation) (bool, string, error) {
	input, err := g.prepareModernGoInput()
	if err != nil {
		return false, "", err
	}
	candidate, previous, prior, err := g.computeModernGoOutputs(input)
	if err != nil {
		return false, "", err
	}
	store, err := g.openStore()
	if err != nil {
		return false, "", err
	}
	batch := artifact.Batch{Key: modernCensusCheckName + "/" + input.ID.String()}
	candidateID, err := stageModernGoCensus(ctx, store, &batch, candidate)
	if err != nil {
		return false, "", err
	}
	references := modernGoReferences{Candidate: candidateID, Base: candidateID, CandidateKey: input.CandidateKey, Prior: prior}
	if input.Base.Identity() != input.Source.Identity() {
		if references.Base, err = stageModernGoCensus(ctx, store, &batch, previous); err != nil {
			return false, "", err
		}
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return false, "", err
	}
	encoded, err := json.Marshal(references)
	return false, string(encoded), err
}

// storedModernGoCensus is a census as the store holds it: the header of its
// split by package and the parts, each its own content. Parts of unchanged
// packages are the bytes the store already holds, so a landing adds only the
// parts of the packages it changed. A census stored whole has no parts.
type storedModernGoCensus struct {
	repoanalysis.ModernGoCensus
	Parts []artifact.ID `json:"parts,omitempty"`
}

var modernCensusPartContract = artifact.JSONContract(artifact.KindProfile, repoanalysis.ModernGoCensusPartSchema)

// stageModernGoCensus adds a census's header, and the parts the store does not
// yet hold, to batch, and returns the header's identity.
func stageModernGoCensus(ctx context.Context, store *overgodb.Store, batch *artifact.Batch, census repoanalysis.ModernGoCensus) (artifact.ID, error) {
	header, parts := census.SplitByPackage()
	stored := storedModernGoCensus{ModernGoCensus: header}
	for _, part := range parts {
		content, err := artifact.JSONContent(modernCensusPartContract, part)
		if err != nil {
			return artifact.ID{}, err
		}
		stored.Parts = append(stored.Parts, content.Descriptor.ID)
		held, err := store.HasContent(ctx, content.Descriptor.ID)
		if err != nil {
			return artifact.ID{}, err
		}
		if !held && !slices.ContainsFunc(batch.Contents, func(staged artifact.Content) bool { return staged.Descriptor.ID == content.Descriptor.ID }) {
			batch.Contents = append(batch.Contents, content)
		}
	}
	content, err := artifact.JSONContent(modernCensusContract, stored)
	if err != nil {
		return artifact.ID{}, err
	}
	batch.Contents = append(batch.Contents, content)
	return content.Descriptor.ID, nil
}

func (input *modernGoInput) compute() (repoanalysis.ModernGoCensus, repoanalysis.ModernGoCensus, error) {
	candidate, err := input.candidate()
	if err != nil {
		return candidate, repoanalysis.ModernGoCensus{}, err
	}
	previous := candidate
	if input.Base.Identity() != input.Source.Identity() {
		previous, err = repoanalysis.ModernGoCensusSnapshot(input.Base, input.Selection, input.TargetGo)
	}
	return candidate, previous, err
}

// Baseline rollover changes the pair, not the candidate computation. The
// prior execution is captured before parallel cache writes begin.
func (g *gateContext) computeModernGoOutputs(input *modernGoInput) (repoanalysis.ModernGoCensus, repoanalysis.ModernGoCensus, artifact.ID, error) {
	if g.modernPrior != nil && input.CandidateKey == input.BaseKey {
		var references modernGoReferences
		if json.Unmarshal([]byte(g.modernPrior.Detail), &references) == nil && references.CandidateKey == input.CandidateKey {
			references.Base = references.Candidate
			detail, err := json.Marshal(references)
			if err != nil {
				return repoanalysis.ModernGoCensus{}, repoanalysis.ModernGoCensus{}, artifact.ID{}, err
			}
			candidate, base, found, err := g.readModernGoOutput(automationcheck.Evidence{Detail: string(detail)})
			if err != nil || found {
				return candidate, base, g.modernPrior.Evidence, err
			}
		}
	}
	candidate, base, err := input.compute()
	return candidate, base, artifact.ID{}, err
}

// A proven result with an absent output runs again through the same DAG node.
func (g *gateContext) reusableOutput(check automationcheck.Invocation, evidence automationcheck.Evidence) (bool, error) {
	if check.Check.Name != modernCensusCheckName {
		return true, nil
	}
	started := time.Now()
	_, _, found, err := g.readModernGoOutput(evidence)
	g.note(fmt.Sprintf("modern-Go computation reuse: usable=%t lookup=%s", found, time.Since(started)))
	return found, err
}

func (g *gateContext) readModernGoOutput(evidence automationcheck.Evidence) (repoanalysis.ModernGoCensus, repoanalysis.ModernGoCensus, bool, error) {
	var empty repoanalysis.ModernGoCensus
	detail := evidence.Detail
	if evidence.Reused && evidence.Source != nil {
		detail = evidence.Source.Detail
	}
	var references modernGoReferences
	if json.Unmarshal([]byte(detail), &references) != nil || !references.Candidate.Valid() || !references.Base.Valid() {
		return empty, empty, false, nil
	}
	input, err := g.prepareModernGoInput()
	if err != nil {
		return empty, empty, false, err
	}
	store, err := g.openStore()
	if err != nil {
		return empty, empty, false, err
	}
	read := func(id artifact.ID, source string) (repoanalysis.ModernGoCensus, bool, error) {
		content, found, err := artifact.ReadContent(context.Background(), store, id)
		if err != nil || !found {
			return empty, false, err
		}
		var stored storedModernGoCensus
		if content.Descriptor.Schema != modernCensusContract.Schema || json.Unmarshal(content.Data, &stored) != nil ||
			stored.SourceIdentity != source || stored.TargetGo != input.TargetGo ||
			stored.BuildContext != input.Selection.Context || stored.CatalogSHA256 != repoanalysis.ModernGoCatalogSHA256 {
			return empty, false, nil
		}
		// A released part is a miss, as a released census is.
		parts := make([]repoanalysis.ModernGoCensus, len(stored.Parts))
		for index, id := range stored.Parts {
			part, found, err := artifact.ReadContent(context.Background(), store, id)
			if err != nil || !found {
				return empty, false, err
			}
			if part.Descriptor.Schema != modernCensusPartContract.Schema || json.Unmarshal(part.Data, &parts[index]) != nil {
				return empty, false, nil
			}
		}
		return repoanalysis.JoinModernGoCensus(stored.ModernGoCensus, parts), true, nil
	}
	candidate, found, err := read(references.Candidate, input.Source.Identity())
	if err != nil || !found {
		return empty, empty, false, err
	}
	if references.Candidate == references.Base && input.Source.Identity() == input.Base.Identity() {
		return candidate, candidate, true, nil
	}
	base, found, err := read(references.Base, input.Base.Identity())
	return candidate, base, found, err
}

func (g *gateContext) modernGoResults() (repoanalysis.ModernGoCensus, repoanalysis.ModernGoCensus, error) {
	g.terminalMutex.Lock()
	evidence, found := g.terminal[modernCensusCheckName]
	g.terminalMutex.Unlock()
	// Standalone ratchet callers still compute; pipeline admission requires
	// the completed predecessor and its retained output.
	if !found && g.manifestPlan == nil {
		input, err := g.prepareModernGoInput()
		if err != nil {
			return repoanalysis.ModernGoCensus{}, repoanalysis.ModernGoCensus{}, err
		}
		return input.compute()
	}
	if !found {
		return repoanalysis.ModernGoCensus{}, repoanalysis.ModernGoCensus{}, errors.New("modern-Go admission: computation has not completed")
	}
	candidate, base, found, err := g.readModernGoOutput(evidence)
	if err == nil && !found {
		err = errors.New("modern-Go admission: computation output is unavailable")
	}
	return candidate, base, err
}
