package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// A projection proof is the verification its activation accepted. Its
// re-acquisition runs again, at the running commit, a verification its recipe
// passed, and re-verifies the active recipe on what that run records, as a
// training run's Observer.Verify does. Two kinds ground one: an exact
// verification names the suite it ran through its plan's dataset, which the
// committed suite's cases identify, and recipe verify -reverify runs that
// suite; the E4B serving producer's verification names the native oracle it
// matched by sha256, and that producer publishes and re-verifies when its
// publication is switched on.
const (
	projectionProof = "projection-proof"
	exactPlan       = "contract=exact;plan="
	protocolOracle  = "fixtures/e4b_multimodal_golden.json"
	protocolNames   = "contract=e4b-http-modalities;oracle_sha256="
	// The serving producer is a real-device test: it publishes only when
	// asked, and into the store it is told.
	protocolDevice  = "OVERGO_CUDA_TEST=1"
	protocolPublish = "OVERGO_E4B_PUBLISH_PROTOCOL=1"
	protocolStore   = "OVERGO_E4B_PUBLISH_STORE="
)

// projectionSuites are the committed exact projection suites, each byte for
// byte the suite an exact verification of its recipe ran, with the media its
// cases name beside it.
var projectionSuites = []string{"docs/verification/fixtures/qwen27/suite.json"}

// protocolProducer runs the E4B serving producer alone.
var protocolProducer = []string{"test", "./internal/server", "-run", "^TestE4BHTTPModalities$", "-count=1"}

// groundProjection gives each projection cell, in place, the committed suite
// or oracle that grounds a succeeded verification of its recipe, an exact
// suite before the oracle since any projection recipe runs one, and for an
// exact suite the projector its active recipe binds. A projector whose bytes
// are absent stays unnamed, and the cell's command says so.
func groundProjection(ctx context.Context, root string, store *overgodb.Store, cells []ModelValidation) error {
	grounds := append(slices.Clone(projectionSuites), protocolOracle)
	names := make([]string, len(grounds))
	for index, path := range grounds {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			return err
		}
		names[index] = fmt.Sprintf("%x", sha256.Sum256(data))
		if path == protocolOracle {
			continue
		}
		// CompileExact identifies an exact suite's dataset by its cases.
		var suite evaluation.ExactSuite
		dataset := artifact.ID{}
		if err = json.Unmarshal(data, &suite); err == nil {
			dataset, err = artifact.JSONID(artifact.KindDataset, suite.Cases)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		names[index] = dataset.String()
	}
	var memo *discovery.Memo
	for index, cell := range cells {
		if cell.Validation != projectionProof || !cell.Recipe.Valid() {
			continue
		}
		passed, err := passedVerifications(ctx, store, cell.Recipe)
		if err != nil {
			return err
		}
		if at := slices.IndexFunc(names, func(name string) bool { return passed[name] }); at >= 0 {
			cells[index].Grounding = grounds[at]
		}
		if cells[index].Grounding == "" || cells[index].Grounding == protocolOracle {
			continue
		}
		if memo == nil {
			memo = discovery.LoadMemo(ctx, store)
		}
		if projector, found, err := discovery.ActiveProjector(ctx, store, cell.Model, memo); err == nil && found {
			cells[index].Projector = projector
		}
	}
	return nil
}

// passedVerifications names what each succeeded verification of a recipe
// ran: an exact verification its plan's dataset, the E4B serving producer's
// its oracle's sha256.
func passedVerifications(ctx context.Context, store *overgodb.Store, recipeID artifact.ID) (map[string]bool, error) {
	edges, err := store.Children(ctx, recipeID)
	if err != nil {
		return nil, err
	}
	passed := map[string]bool{}
	for _, edge := range edges {
		if edge.Relation != artifact.RelationDependsOn || edge.Child.Kind() != artifact.KindEvidence {
			continue
		}
		content, found, err := artifact.ReadContent(ctx, store, edge.Child)
		if err != nil {
			return nil, err
		}
		if !found || content.Descriptor.MediaType != runrecord.GateMediaType || content.Descriptor.Schema != runrecord.GateSchema {
			continue
		}
		gate, err := runrecord.ParseGateResult(content.Data)
		if err != nil {
			return nil, err
		}
		if gate.Recipe != recipeID || gate.Outcome != runrecord.OutcomeSucceeded || len(gate.Steps) != 1 {
			continue
		}
		evidence := gate.Steps[0].Evidence
		if oracle, protocol := strings.CutPrefix(evidence, protocolNames); protocol {
			oracle, _, _ = strings.Cut(oracle, ";")
			passed[oracle] = true
		}
		named, exact := strings.CutPrefix(evidence, exactPlan)
		if !exact {
			continue
		}
		named, _, _ = strings.Cut(named, ";")
		planID, err := artifact.ParseID(named)
		if err != nil {
			return nil, err
		}
		planContent, found, err := artifact.ReadContent(ctx, store, planID)
		if err != nil || !found {
			return nil, errors.Join(err, fmt.Errorf("validate: exact verification %s names absent plan %s", gate.ID, planID))
		}
		plan, err := evaluation.ParsePlan(planContent.Data)
		if err != nil {
			return nil, err
		}
		passed[plan.Dataset().String()] = true
	}
	return passed, nil
}

// projectionRun is the command that re-verifies a projection cell's active
// recipe into store and the environment it runs under: recipe verify
// -reverify over its exact suite against the projector the recipe binds, or
// the E4B serving producer publishing into store. It says why when nothing
// grounds the cell or an input is missing.
func projectionRun(cell ModelValidation, store string) (args, environment []string, err error) {
	switch {
	case cell.Grounding == protocolOracle:
		return slices.Clone(protocolProducer), []string{protocolDevice, protocolPublish, protocolStore + store}, nil
	case cell.Grounding == "":
		return nil, nil, errors.New("no committed exact suite or oracle grounds a verification its recipe passed")
	case cell.Location == "" || cell.Projector == "":
		return nil, nil, errors.New("an exact projection suite runs on the model's weights and the projector bytes its active recipe binds")
	}
	return []string{
		"run", "./cmd/recipe", "verify", "-repo", store, "-task", string(recipe.TaskProjection),
		"-projector", cell.Projector, "-input-file", cell.Grounding, "-reverify", cell.Location,
	}, nil, nil
}
