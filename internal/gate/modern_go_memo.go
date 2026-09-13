package gate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
)

const modernCensusCheckName = "modern-census"

// One immutable census payload; retry evidence holds references, never copies.
var modernCensusContract = artifact.JSONContract(artifact.KindProfile, "overgo/modern-go-computation/v1")

type modernGoInput struct {
	ID        artifact.ID
	Source    repoanalysis.SourceSnapshot
	Base      repoanalysis.SourceSnapshot
	Selection repoanalysis.BuildSelection
	TargetGo  string
}

type modernGoReferences struct {
	Candidate artifact.ID `json:"candidate"`
	Base      artifact.ID `json:"base"`
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
	selection, err := repoanalysis.HostBuildSelection(g.sourceRoot(), "./cmd/...", "./internal/...")
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
	input.ID, err = artifact.JSONID(artifact.KindProfile, struct {
		Schema, Source, Base, Context, TargetGo, Catalog, Compiler, Build string
		Module                                                            artifact.ID
		Environment                                                       artifact.ID `json:"environment,omitzero"`
		Files                                                             map[string]bool
		Packages                                                          map[string]string
	}{modernCensusContract.Schema, snapshot.Identity(), base.Identity(), selection.Context, baseline.TargetGo,
		repoanalysis.ModernGoCatalogSHA256, compiler, build, module, g.environment.ID, selection.Files, selection.Packages})
	if err != nil {
		return nil, err
	}
	g.modernInput = input
	return input, nil
}

func (g *gateContext) computeModernGo(ctx context.Context, _ automationcheck.Invocation) (bool, string, error) {
	input, err := g.prepareModernGoInput()
	if err != nil {
		return false, "", err
	}
	candidate, previous, err := input.compute()
	if err != nil {
		return false, "", err
	}
	store, err := g.openStore()
	if err != nil {
		return false, "", err
	}
	batch := artifact.Batch{Key: modernCensusCheckName + "/" + input.ID.String()}
	candidateContent, err := artifact.JSONContent(modernCensusContract, candidate)
	if err != nil {
		return false, "", err
	}
	batch.Contents = append(batch.Contents, candidateContent)
	references := modernGoReferences{Candidate: candidateContent.Descriptor.ID, Base: candidateContent.Descriptor.ID}
	if input.Base.Identity() != input.Source.Identity() {
		baseContent, err := artifact.JSONContent(modernCensusContract, previous)
		if err != nil {
			return false, "", err
		}
		batch.Contents = append(batch.Contents, baseContent)
		references.Base = baseContent.Descriptor.ID
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return false, "", err
	}
	encoded, err := json.Marshal(references)
	return false, string(encoded), err
}

func (input *modernGoInput) compute() (repoanalysis.ModernGoCensus, repoanalysis.ModernGoCensus, error) {
	candidate, err := repoanalysis.ModernGoCensusSnapshot(input.Source, input.Selection, input.TargetGo)
	if err != nil {
		return candidate, repoanalysis.ModernGoCensus{}, err
	}
	previous := candidate
	if input.Base.Identity() != input.Source.Identity() {
		previous, err = repoanalysis.ModernGoCensusSnapshot(input.Base, input.Selection, input.TargetGo)
	}
	return candidate, previous, err
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
		var output repoanalysis.ModernGoCensus
		if content.Descriptor.Schema != modernCensusContract.Schema || json.Unmarshal(content.Data, &output) != nil ||
			output.SourceIdentity != source || output.TargetGo != input.TargetGo ||
			output.BuildContext != input.Selection.Context || output.CatalogSHA256 != repoanalysis.ModernGoCatalogSHA256 {
			return empty, false, nil
		}
		return output, true, nil
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
