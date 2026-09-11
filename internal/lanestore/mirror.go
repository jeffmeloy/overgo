// Package lanestore mirrors a source store's active inference activations
// into a lane store: the record closure each activation validates
// (definition, lifecycle events, decisions, gate and run evidence, model
// manifest, descriptors and file locations) is copied to a bounded depth,
// the source's aliases are bound over the copied records, and lane
// activations the source no longer holds are released, so a lane gate
// reads the same activation truth as its source.
package lanestore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// MirrorDepth is the record depth the closure of one activation is followed
// to: the definition, its lifecycle and evidence, and the records those name.
const MirrorDepth = 3

const (
	// closureBound refuses a closure that stopped being one activation's.
	closureBound = 5000
	// batchSize bounds one committed batch of copied records.
	batchSize = 200
)

var idPattern = regexp.MustCompile(`[a-z][a-z-]*:sha256:[0-9a-f]{64}`)

// allMatches asks the pattern for every match, the sentinel regexp takes.
const allMatches = -1

// Report is the mirror's outcome: what the source held, what the closure
// reached, what was copied, and the activations bound and released.
type Report struct {
	SourceActive   int
	ClosureRecords int
	Copied         map[string]int
	Bindings       int
	Released       []string
	Commits        []artifact.CommitID
}

type pending struct {
	id    artifact.ID
	depth int
}

// MirrorActivations copies the source's active inference activations into
// the target and returns the report; a mirror after which some source
// activation does not resolve in the target is an error.
func MirrorActivations(ctx context.Context, source, target *overgodb.Store, depth int) (Report, error) {
	if ctx == nil || source == nil || target == nil {
		return Report{}, errors.New("lanestore: mirror requires a context, a source and a target")
	}
	prefix := modelrecipe.ActiveAliasPrefix(recipe.TaskInference)
	sourceActive := map[artifact.ID]artifact.ID{}
	var queue []pending
	if err := source.VisitAliases(ctx, prefix, func(view overgodb.AliasView) error {
		model, _, ok := modelrecipe.ParseActiveAlias(view.Name)
		if !ok {
			return fmt.Errorf("lanestore: alias %q is not an activation", view.Name)
		}
		sourceActive[model] = view.Target
		queue = append(queue, pending{id: model}, pending{id: view.Target})
		return nil
	}); err != nil {
		return Report{}, err
	}
	report := Report{SourceActive: len(sourceActive), Copied: map[string]int{}}
	walker := closureWalker{ctx: ctx, source: source, target: target, depth: map[artifact.ID]int{}, maxDepth: depth, report: &report}
	batches, err := walker.walk(queue)
	if err != nil {
		return Report{}, err
	}
	aliases, err := walker.bindings(prefix, sourceActive)
	if err != nil {
		return Report{}, err
	}
	if !aliases.Empty() {
		batches = append(batches, aliases)
	}
	for _, batch := range batches {
		commit, err := artifact.CommitBatch(ctx, target, batch)
		if err != nil {
			return Report{}, fmt.Errorf("lanestore: %s: %w", batch.Key, err)
		}
		report.Commits = append(report.Commits, commit)
	}
	for model, want := range sourceActive {
		current, bound, err := target.ResolveAlias(ctx, modelrecipe.ActiveAlias(model, recipe.TaskInference))
		if err != nil {
			return Report{}, err
		}
		if !bound || current != want {
			return Report{}, fmt.Errorf("lanestore: activation of %s does not resolve to the source's %s after the mirror", model, want)
		}
	}
	return report, nil
}

// closureWalker follows the records an activation names, copying the ones
// the target lacks.
type closureWalker struct {
	ctx      context.Context
	source   *overgodb.Store
	target   *overgodb.Store
	depth    map[artifact.ID]int
	maxDepth int
	report   *Report
	current  artifact.Batch
	batches  []artifact.Batch
	items    int
}

func (w *closureWalker) walk(queue []pending) ([]artifact.Batch, error) {
	// An alias whose name embeds a closure id (the lifecycle status alias of
	// a recipe) reaches records the contents never name; follow those too.
	seedAliases := func() error {
		return w.source.VisitAliases(w.ctx, "", func(view overgodb.AliasView) error {
			if _, seen := w.depth[view.Target]; seen {
				return nil
			}
			for _, match := range idPattern.FindAllString(view.Name, allMatches) {
				if id, err := artifact.ParseID(match); err == nil {
					if level, seen := w.depth[id]; seen && level < w.maxDepth {
						queue = append(queue, pending{view.Target, level + 1})
						return nil
					}
				}
			}
			return nil
		})
	}
	for len(queue) > 0 {
		next := queue[0]
		queue = queue[1:]
		if level, seen := w.depth[next.id]; seen && level <= next.depth {
			if len(queue) == 0 {
				if err := seedAliases(); err != nil {
					return nil, err
				}
			}
			continue
		}
		if len(w.depth) >= closureBound {
			return nil, fmt.Errorf("lanestore: closure exceeds %d records", closureBound)
		}
		_, revisited := w.depth[next.id]
		w.depth[next.id] = next.depth
		texts, err := w.visit(next.id, !revisited)
		if err != nil {
			return nil, err
		}
		if next.depth < w.maxDepth {
			for _, text := range texts {
				for _, match := range idPattern.FindAll(text, allMatches) {
					child, err := artifact.ParseID(string(match))
					if err != nil {
						continue
					}
					if level, seen := w.depth[child]; !seen || level > next.depth+1 {
						queue = append(queue, pending{child, next.depth + 1})
					}
				}
			}
		}
		if len(queue) == 0 {
			if err := seedAliases(); err != nil {
				return nil, err
			}
		}
	}
	w.flush()
	w.report.ClosureRecords = len(w.depth)
	return w.batches, nil
}

// visit copies one record's descriptor, content, manifest and locations
// the target lacks when first is set, and returns the texts that name
// further records.
func (w *closureWalker) visit(id artifact.ID, first bool) ([][]byte, error) {
	var texts [][]byte
	if !first {
		// Re-walked at a shallower depth: re-read the texts to descend.
		if content, found, err := artifact.ReadContent(w.ctx, w.source, id); err == nil && found {
			texts = append(texts, content.Data)
		}
		if manifest, found, err := w.source.Manifest(w.ctx, id); err == nil && found {
			raw, _ := json.Marshal(manifest)
			texts = append(texts, raw)
		}
		return texts, nil
	}
	descriptor, found, err := w.source.Artifact(w.ctx, id)
	if err != nil {
		return nil, err
	}
	if found {
		if _, known, err := w.target.Artifact(w.ctx, id); err != nil {
			return nil, err
		} else if !known {
			w.current.Artifacts = append(w.current.Artifacts, descriptor)
			w.add("descriptor")
		}
		raw, _ := json.Marshal(descriptor)
		texts = append(texts, raw)
	}
	if content, stored, err := artifact.ReadContent(w.ctx, w.source, id); err == nil && stored {
		if present, err := w.target.HasContent(w.ctx, id); err != nil {
			return nil, err
		} else if !present {
			// A content without its own descriptor row carries the one it embeds.
			if !found {
				w.current.Artifacts = append(w.current.Artifacts, content.Descriptor)
			}
			w.current.Contents = append(w.current.Contents, content)
			w.add("content")
		}
		texts = append(texts, content.Data)
	}
	if manifest, found, err := w.source.Manifest(w.ctx, id); err != nil {
		return nil, err
	} else if found {
		if _, known, err := w.target.Manifest(w.ctx, id); err != nil {
			return nil, err
		} else if !known {
			w.current.Manifests = append(w.current.Manifests, manifest)
			w.add("manifest")
		}
		raw, _ := json.Marshal(manifest)
		texts = append(texts, raw)
	}
	sourceLocations, err := w.source.Locations(w.ctx, id)
	if err != nil {
		return nil, err
	}
	if len(sourceLocations) != 0 {
		targetLocations, err := w.target.Locations(w.ctx, id)
		if err != nil {
			return nil, err
		}
		for _, location := range sourceLocations {
			if !slices.Contains(targetLocations, location) {
				w.current.Locations = append(w.current.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
				w.add("location")
			}
		}
	}
	return texts, nil
}

func (w *closureWalker) add(kind string) {
	w.report.Copied[kind]++
	w.items++
	if w.items%batchSize == 0 {
		w.flush()
	}
}

// flush closes the current batch under a key derived from its own records,
// so a repeated mirror names the same content and never conflicts.
func (w *closureWalker) flush() {
	if w.current.Empty() {
		return
	}
	w.current.Key = "lane-mirror/records/" + batchDigest(w.current)
	w.batches = append(w.batches, w.current)
	w.current = artifact.Batch{}
}

// bindings binds the source's aliases over the copied closure and releases
// the target's activations the source no longer holds.
func (w *closureWalker) bindings(prefix string, sourceActive map[artifact.ID]artifact.ID) (artifact.Batch, error) {
	closure := func(id artifact.ID) bool { _, ok := w.depth[id]; return ok }
	var aliases artifact.Batch
	if err := w.source.VisitAliases(w.ctx, "", func(view overgodb.AliasView) error {
		if !closure(view.Target) {
			return nil
		}
		// An alias naming a record outside the closure (a policy alias of a
		// recipe the lane never received) stays with its recipe.
		for _, match := range idPattern.FindAllString(view.Name, allMatches) {
			if id, err := artifact.ParseID(match); err == nil && !closure(id) {
				return nil
			}
		}
		current, bound, err := w.target.ResolveAlias(w.ctx, view.Name)
		if err != nil {
			return err
		}
		if bound && current == view.Target {
			return nil
		}
		binding := artifact.AliasBinding{Name: view.Name, Target: view.Target}
		if bound {
			binding.Previous = &current
		}
		aliases.Aliases = append(aliases.Aliases, binding)
		return nil
	}); err != nil {
		return artifact.Batch{}, err
	}
	if err := w.target.VisitAliases(w.ctx, prefix, func(view overgodb.AliasView) error {
		model, _, ok := modelrecipe.ParseActiveAlias(view.Name)
		if !ok {
			return fmt.Errorf("lanestore: alias %q is not an activation", view.Name)
		}
		if _, held := sourceActive[model]; held {
			return nil
		}
		previous := view.Target
		aliases.Aliases = append(aliases.Aliases, artifact.AliasBinding{Name: view.Name, Target: view.Target, Previous: &previous, Remove: true})
		w.report.Released = append(w.report.Released, view.Name)
		return nil
	}); err != nil {
		return artifact.Batch{}, err
	}
	w.report.Bindings = len(aliases.Aliases)
	if !aliases.Empty() {
		aliases.Key = "lane-mirror/aliases/" + batchDigest(aliases)
	}
	return aliases, nil
}

// batchDigest names a batch by its own records.
func batchDigest(batch artifact.Batch) string {
	var names []string
	for _, descriptor := range batch.Artifacts {
		names = append(names, "artifact:"+descriptor.ID.String())
	}
	for _, content := range batch.Contents {
		names = append(names, "content:"+content.Descriptor.ID.String())
	}
	for _, manifest := range batch.Manifests {
		names = append(names, "manifest:"+manifest.ID.String())
	}
	for _, location := range batch.Locations {
		raw, _ := json.Marshal(location.Location)
		names = append(names, "location:"+string(raw))
	}
	for _, alias := range batch.Aliases {
		names = append(names, fmt.Sprintf("alias:%s>%s>%t", alias.Name, alias.Target, alias.Remove))
	}
	slices.Sort(names)
	digest := sha256.Sum256([]byte(strings.Join(names, "\n")))
	return hex.EncodeToString(digest[:])
}
