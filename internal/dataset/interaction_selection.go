package dataset

import (
	"errors"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

// InteractionSelectionBounds is one aggregate context budget. Every dimension
// is mandatory so callers cannot accidentally replace one unbounded scan with
// another.
type InteractionSelectionBounds struct {
	MaxTokens    uint64 `json:"max_tokens"`
	MaxBytes     uint64 `json:"max_bytes"`
	MaxDocuments uint64 `json:"max_documents"`
	MaxDepth     uint32 `json:"max_depth"`
	MaxResults   uint64 `json:"max_results"`
}

// Identify validates and identifies the selection contract.
func (bounds InteractionSelectionBounds) Identify() (artifact.ID, error) {
	if bounds.MaxTokens == 0 || bounds.MaxBytes == 0 || bounds.MaxDocuments == 0 || bounds.MaxDepth == 0 || bounds.MaxResults == 0 {
		return artifact.ID{}, errors.New("dataset: interaction selection requires every aggregate bound")
	}
	return artifact.JSONID(artifact.KindProfile, bounds)
}

// InteractionSelectionSource is one complete recent interaction arc with its
// exact source citation, causal root, and pre-measured aggregate cost.
type InteractionSelectionSource struct {
	Source     artifact.ID `json:"source"`
	CausalRoot artifact.ID `json:"causal_root"`
	Tokens     uint64      `json:"tokens"`
	Bytes      uint64      `json:"bytes"`
	Documents  uint64      `json:"documents"`
	Depth      uint32      `json:"depth"`
}

// InteractionSelectionUsage reports the aggregate cost actually admitted.
type InteractionSelectionUsage struct {
	Tokens    uint64 `json:"tokens"`
	Bytes     uint64 `json:"bytes"`
	Documents uint64 `json:"documents"`
	Results   uint64 `json:"results"`
	Depth     uint32 `json:"depth"`
}

// InteractionSelectionCursor binds continuation to the exact journal head,
// selection contract, and cited source set.
type InteractionSelectionCursor struct {
	Head       artifact.CommitID `json:"head"`
	Contract   artifact.ID       `json:"contract"`
	SourceSet  artifact.ID       `json:"source_set"`
	AfterDepth uint32            `json:"after_depth"`
	After      artifact.ID       `json:"after"`
	ID         artifact.ID       `json:"id"`
}

type interactionSelectionCursorIdentity struct {
	Head       artifact.CommitID `json:"head"`
	Contract   artifact.ID       `json:"contract"`
	SourceSet  artifact.ID       `json:"source_set"`
	AfterDepth uint32            `json:"after_depth"`
	After      artifact.ID       `json:"after"`
}

func newInteractionSelectionCursor(head artifact.CommitID, contract, sourceSet artifact.ID, afterDepth uint32, after artifact.ID) (InteractionSelectionCursor, error) {
	identity := interactionSelectionCursorIdentity{
		Head: head, Contract: contract, SourceSet: sourceSet, AfterDepth: afterDepth, After: after,
	}
	id, err := artifact.JSONID(artifact.KindEvidence, identity)
	return InteractionSelectionCursor{
		Head: head, Contract: contract, SourceSet: sourceSet, AfterDepth: afterDepth, After: after, ID: id,
	}, err
}

func (cursor InteractionSelectionCursor) valid() bool {
	expected, err := newInteractionSelectionCursor(cursor.Head, cursor.Contract, cursor.SourceSet, cursor.AfterDepth, cursor.After)
	return err == nil && cursor.ID == expected.ID
}

// InteractionSelection is one bounded page of complete cited arcs.
type InteractionSelection struct {
	Head      artifact.CommitID            `json:"head"`
	Bounds    InteractionSelectionBounds   `json:"bounds"`
	SourceSet artifact.ID                  `json:"source_set"`
	Sources   []InteractionSelectionSource `json:"sources"`
	Usage     InteractionSelectionUsage    `json:"usage"`
	Cursor    *InteractionSelectionCursor  `json:"cursor,omitempty"`
	Truncated bool                         `json:"truncated"`
}

// Validate verifies a selection after it crosses a durable boundary.
func (selection InteractionSelection) Validate() error {
	contract, err := selection.Bounds.Identify()
	if err != nil || !selection.Head.Valid() || selection.SourceSet.Kind() != artifact.KindEvidence || len(selection.Sources) == 0 {
		return errors.Join(errors.New("dataset: invalid interaction selection boundary"), err)
	}
	if selection.Truncated {
		if selection.Cursor == nil || !selection.Cursor.valid() || selection.Cursor.Head != selection.Head ||
			selection.Cursor.Contract != contract || selection.Cursor.SourceSet != selection.SourceSet {
			return errors.New("dataset: invalid truncated interaction selection boundary")
		}
		last := selection.Sources[len(selection.Sources)-1]
		if selection.Cursor.AfterDepth != last.Depth || selection.Cursor.After != last.Source {
			return errors.New("dataset: interaction selection cursor does not follow its admitted sources")
		}
	} else {
		if selection.Cursor != nil {
			return errors.New("dataset: complete interaction selection carries a cursor")
		}
		rebuilt, rebuildErr := SelectInteractions(selection.Bounds, selection.Head, selection.Sources, nil)
		if rebuildErr != nil || rebuilt.SourceSet != selection.SourceSet || rebuilt.Usage != selection.Usage || rebuilt.Truncated {
			return errors.Join(errors.New("dataset: interaction selection boundary differs from its sources"), rebuildErr)
		}
	}
	return validateInteractionSelectionPage(selection)
}

// SelectInteractions returns recent arcs in stable depth/source order. A
// continuation whose head, contract, or source set moved is refused.
func SelectInteractions(bounds InteractionSelectionBounds, head artifact.CommitID, sources []InteractionSelectionSource, cursor *InteractionSelectionCursor) (InteractionSelection, error) {
	contract, err := bounds.Identify()
	if err != nil || !head.Valid() || len(sources) == 0 {
		return InteractionSelection{}, errors.Join(errors.New("dataset: invalid interaction selection"), err)
	}
	ordered := slices.Clone(sources)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Depth != ordered[j].Depth {
			return ordered[i].Depth < ordered[j].Depth
		}
		return artifact.CompareID(ordered[i].Source, ordered[j].Source) < 0
	})
	for index, source := range ordered {
		if source.Source.Kind() != artifact.KindEvidence || source.CausalRoot.Kind() != artifact.KindEvidence ||
			source.Tokens == 0 || source.Bytes == 0 || source.Documents == 0 || source.Depth == 0 ||
			index > 0 && source.Source == ordered[index-1].Source {
			return InteractionSelection{}, errors.New("dataset: invalid cited interaction source")
		}
	}
	sourceSet, err := artifact.JSONID(artifact.KindEvidence, ordered)
	if err != nil {
		return InteractionSelection{}, err
	}
	start := 0
	if cursor != nil {
		if !cursor.valid() || cursor.Head != head || cursor.Contract != contract || cursor.SourceSet != sourceSet {
			return InteractionSelection{}, errors.New("dataset: interaction selection cursor is stale")
		}
		start = slices.IndexFunc(ordered, func(source InteractionSelectionSource) bool {
			return source.Depth == cursor.AfterDepth && source.Source == cursor.After
		}) + 1
		if start == 0 {
			return InteractionSelection{}, errors.New("dataset: interaction selection cursor source is absent")
		}
	}
	selection := InteractionSelection{Head: head, Bounds: bounds, SourceSet: sourceSet}
	for index := start; index < len(ordered); index++ {
		source := ordered[index]
		if source.Depth > bounds.MaxDepth || selection.Usage.Results == bounds.MaxResults ||
			!withinSelectionBound(selection.Usage.Tokens, source.Tokens, bounds.MaxTokens) ||
			!withinSelectionBound(selection.Usage.Bytes, source.Bytes, bounds.MaxBytes) ||
			!withinSelectionBound(selection.Usage.Documents, source.Documents, bounds.MaxDocuments) {
			selection.Truncated = true
			break
		}
		selection.Sources = append(selection.Sources, source)
		selection.Usage.Tokens += source.Tokens
		selection.Usage.Bytes += source.Bytes
		selection.Usage.Documents += source.Documents
		selection.Usage.Results++
		selection.Usage.Depth = max(selection.Usage.Depth, source.Depth)
	}
	if len(selection.Sources) == 0 {
		return InteractionSelection{}, errors.New("dataset: one interaction arc exceeds the aggregate selection bounds")
	}
	if selection.Truncated {
		last := selection.Sources[len(selection.Sources)-1]
		continuation, cursorErr := newInteractionSelectionCursor(head, contract, sourceSet, last.Depth, last.Source)
		if cursorErr != nil {
			return InteractionSelection{}, cursorErr
		}
		selection.Cursor = &continuation
	}
	return selection, nil
}

func validateInteractionSelectionPage(selection InteractionSelection) error {
	var usage InteractionSelectionUsage
	for index, source := range selection.Sources {
		if source.Source.Kind() != artifact.KindEvidence || source.CausalRoot.Kind() != artifact.KindEvidence ||
			source.Tokens == 0 || source.Bytes == 0 || source.Documents == 0 || source.Depth == 0 ||
			index > 0 && (source.Depth < selection.Sources[index-1].Depth ||
				source.Depth == selection.Sources[index-1].Depth && artifact.CompareID(selection.Sources[index-1].Source, source.Source) >= 0) ||
			source.Depth > selection.Bounds.MaxDepth || usage.Results == selection.Bounds.MaxResults ||
			!withinSelectionBound(usage.Tokens, source.Tokens, selection.Bounds.MaxTokens) ||
			!withinSelectionBound(usage.Bytes, source.Bytes, selection.Bounds.MaxBytes) ||
			!withinSelectionBound(usage.Documents, source.Documents, selection.Bounds.MaxDocuments) {
			return errors.New("dataset: invalid admitted interaction source")
		}
		usage.Tokens += source.Tokens
		usage.Bytes += source.Bytes
		usage.Documents += source.Documents
		usage.Results++
		usage.Depth = max(usage.Depth, source.Depth)
	}
	if usage != selection.Usage {
		return errors.New("dataset: interaction selection usage differs")
	}
	return nil
}

func withinSelectionBound(current, addition, limit uint64) bool {
	return addition <= limit && current <= math.MaxUint64-addition && current+addition <= limit
}
