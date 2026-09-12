package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// A census freezes the denominator and its observed evidence, not a claim that
// every registered identity has been verified. Later evidence may improve while
// the expected model/task/recipe set remains fixed.
type capabilityCensus struct {
	Version        uint16            `json:"version"`
	ProducerCommit string            `json:"producer_commit"`
	StoreCommit    artifact.CommitID `json:"store_commit"`
	StoreSequence  uint64            `json:"store_sequence"`
	Models         []censusModel     `json:"models"`
	Verifications  []artifact.ID     `json:"verifications"`
	ID             artifact.ID       `json:"-"`
}

// These fields freeze the stored census contract independently of discovery's UI view.
type censusModel struct {
	Model        artifact.ID
	Location     string
	Present      bool
	Capabilities []censusCapability
	// Preserve both historical absence and the later explicit empty field.
	KeyEnvironment *string `json:",omitzero"`
}

type censusCapability struct {
	Task   recipe.Task
	Recipe artifact.ID
	Tier   recipe.EvidenceTier
	Stale  string
}

func censusModels(entries []discovery.CatalogEntry) []censusModel {
	result := make([]censusModel, len(entries))
	for i, entry := range entries {
		result[i] = censusModel{Model: entry.Model, Location: entry.Location, Present: entry.Present, KeyEnvironment: new(entry.KeyEnvironment)}
		if entry.Capabilities != nil {
			result[i].Capabilities = make([]censusCapability, len(entry.Capabilities))
			for j, capability := range entry.Capabilities {
				result[i].Capabilities[j] = censusCapability(capability)
			}
		}
	}
	return result
}

var capabilityCensusCodec = artifact.JSONDocumentCodec(
	"capability census", artifact.KindEvidence,
	"application/vnd.overgo.capability-census+json", "overgo/capability-census/v1",
	validateCapabilityCensus,
	func(value capabilityCensus) artifact.ID { return value.ID },
	func(value *capabilityCensus, id artifact.ID) { value.ID = id },
	func(value capabilityCensus) capabilityCensus {
		value.Models = slices.Clone(value.Models)
		for index := range value.Models {
			value.Models[index].Capabilities = slices.Clone(value.Models[index].Capabilities)
			value.Models[index].KeyEnvironment = artifact.ClonePointer(value.Models[index].KeyEnvironment)
		}
		value.Verifications = slices.Clone(value.Verifications)
		return value
	},
)

func validateCapabilityCensus(value *capabilityCensus) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.ProducerCommit == "" || strings.TrimSpace(value.ProducerCommit) != value.ProducerCommit || len(value.Models) == 0 ||
		value.StoreSequence == 0 || value.StoreCommit == (artifact.CommitID{}) {
		return errors.New("capability census: version, producer or registered denominator is absent")
	}
	var previous artifact.ID
	for _, model := range value.Models {
		if model.Model.Kind() != artifact.KindModel || previous.Valid() && artifact.CompareID(previous, model.Model) >= 0 {
			return errors.New("capability census: model identities must be unique and ordered")
		}
		previous = model.Model
		var previousTask recipe.Task
		for _, activation := range model.Capabilities {
			if !activation.Task.Valid() || activation.Task <= previousTask || activation.Recipe.Kind() != artifact.KindRecipe ||
				activation.Stale == "" && !activation.Tier.Valid() {
				return errors.New("capability census: activation identity, order or evidence tier differs")
			}
			previousTask = activation.Task
		}
	}
	previous = artifact.ID{}
	for _, id := range value.Verifications {
		if id.Kind() != artifact.KindEvidence || previous.Valid() && artifact.CompareID(previous, id) >= 0 {
			return errors.New("capability census: verification identities must be unique and ordered")
		}
		previous = id
	}
	return nil
}

func buildCapabilityCensus(ctx context.Context, store *overgodb.Store, commit string, limit int, memo *discovery.Memo) (capabilityCensus, error) {
	models, truncated, err := discovery.RegisteredCatalog(ctx, store, limit, memo)
	if err != nil {
		return capabilityCensus{}, err
	}
	if truncated {
		return capabilityCensus{}, errors.New("capability census: registered catalog is truncated")
	}
	value := capabilityCensus{ProducerCommit: commit, Models: censusModels(models)}
	page, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, runrecord.ParseModelVerification, func(_ overgodb.DocumentView, record runrecord.ModelVerification) error {
		value.Verifications = append(value.Verifications, record.ID)
		return nil
	})
	if err != nil {
		return capabilityCensus{}, err
	}
	value.StoreCommit, value.StoreSequence = page.Head, page.Sequence
	slices.SortFunc(value.Verifications, artifact.CompareID)
	return capabilityCensusCodec.NewInitial(value)
}

func checkCapabilityCensus(ctx context.Context, store *overgodb.Store, value capabilityCensus, limit int) error {
	if err := capabilityCensusCodec.ValidateIdentity(value); err != nil {
		return err
	}
	current, truncated, err := discovery.RegisteredCatalog(ctx, store, limit, discovery.LoadMemo(ctx, store))
	if err != nil {
		return err
	}
	if truncated {
		return errors.New("capability census: live registered catalog is truncated")
	}
	if !slices.EqualFunc(value.Models, censusModels(current), func(expected, actual censusModel) bool {
		return expected.Model == actual.Model && expected.Location == actual.Location && expected.Present == actual.Present &&
			slices.Equal(expected.Capabilities, actual.Capabilities)
	}) {
		return fmt.Errorf("capability census: registered identity, availability or task/recipe set changed (snapshot=%d live=%d); record a new disposition before replacing the denominator", len(value.Models), len(current))
	}
	_, err = readCensusVerifications(ctx, store, value)
	return err
}

type censusVerification struct {
	ID     artifact.ID                 `json:"id"`
	Record runrecord.ModelVerification `json:"record"`
}

func readCensusVerifications(ctx context.Context, store *overgodb.Store, value capabilityCensus) ([]censusVerification, error) {
	checkpoint, found, err := store.CommitAt(ctx, value.StoreSequence)
	if err != nil {
		return nil, err
	}
	if !found || checkpoint.ID != value.StoreCommit {
		return nil, errors.New("capability census: recorded store checkpoint is absent")
	}
	registered := make(map[artifact.ID]bool, len(value.Models))
	for _, model := range value.Models {
		registered[model.Model] = true
	}
	var records []censusVerification
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, runrecord.ParseModelVerification, func(view overgodb.DocumentView, record runrecord.ModelVerification) error {
		// A descriptor can precede its content. Only durable-content
		// introduction establishes that an observation existed at the checkpoint.
		introduced, found, err := store.ArtifactIntroduction(ctx, record.ID)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("capability census: verification %s has no durable introduction", record.ID)
		}
		if introduced.Sequence > value.StoreSequence {
			return nil
		}
		if record.ID != view.Content.Descriptor.ID || !registered[record.Model] {
			return fmt.Errorf("capability census: verification %s does not bind a registered identity", record.ID)
		}
		records = append(records, censusVerification{ID: record.ID, Record: record})
		return nil
	})
	if err != nil {
		return nil, err
	}
	slices.SortFunc(records, func(left, right censusVerification) int { return artifact.CompareID(left.ID, right.ID) })
	if !slices.EqualFunc(value.Verifications, records, func(id artifact.ID, record censusVerification) bool { return id == record.ID }) {
		return nil, errors.New("capability census: verification coverage differs from the recorded store checkpoint")
	}
	return records, nil
}

func capabilityCensusBatch(value capabilityCensus) (artifact.Batch, error) {
	parents := slices.Clone(value.Verifications)
	for _, model := range value.Models {
		parents = append(parents, model.Model)
		for _, activation := range model.Capabilities {
			parents = append(parents, activation.Recipe)
		}
	}
	slices.SortFunc(parents, artifact.CompareID)
	var lineage []artifact.Lineage
	for _, parent := range slices.Compact(parents) {
		lineage = append(lineage, artifact.Lineage{Child: value.ID, Parent: parent, Relation: artifact.RelationDerivedFrom})
	}
	return capabilityCensusCodec.Batch("capability-census/"+value.ID.String(), value, lineage, nil)
}

func runCapabilityCensus(repository string, publish bool, checkID string, output io.Writer) error {
	ctx := context.Background()
	var store *overgodb.Store
	var err error
	if publish {
		store, err = overgodb.Open(repository)
	} else {
		store, err = overgodb.OpenReadOnly(repository)
	}
	if err != nil {
		return err
	}
	defer store.Close()
	var value capabilityCensus
	if publish {
		commit, err := runrecord.ExecutableCodeCommit(".")
		if err != nil {
			return err
		}
		memo := discovery.LoadMemo(ctx, store)
		value, err = buildCapabilityCensus(ctx, store, commit, mediaCatalogLimit, memo)
		if err != nil {
			return err
		}
		// Verify all referenced observations before publishing; evidence records
		// are historical snapshots, not substituted for task-native oracles.
		if _, err := readCensusVerifications(ctx, store, value); err != nil {
			return err
		}
		// Reuse the existing stat-validated digest memo for subsequent read-only
		// checks instead of hashing the same registered weights on every gate.
		if err := discovery.PublishMemo(ctx, store, memo); err != nil {
			return err
		}
		batch, err := capabilityCensusBatch(value)
		if err != nil {
			return err
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			return err
		}
	} else {
		var id artifact.ID
		if err := id.UnmarshalText([]byte(checkID)); err != nil {
			return err
		}
		value, err = capabilityCensusCodec.Require(ctx, store, id)
		if err != nil {
			return err
		}
		if err := checkCapabilityCensus(ctx, store, value, mediaCatalogLimit); err != nil {
			return err
		}
	}
	records, err := readCensusVerifications(ctx, store, value)
	if err != nil {
		return err
	}
	activations, missing, stale := 0, 0, 0
	for _, model := range value.Models {
		if !model.Present {
			missing++
		}
		for _, activation := range model.Capabilities {
			activations++
			if activation.Stale != "" {
				stale++
			}
		}
	}
	return json.NewEncoder(output).Encode(struct {
		ID         artifact.ID          `json:"id"`
		Census     capabilityCensus     `json:"census"`
		Historical []censusVerification `json:"historical_verification_records"`
		Audit      string               `json:"audit"`
	}{value.ID, value, records, fmt.Sprintf("registered identities=%d activated task/recipe pairs=%d unavailable identities=%d stale activations=%d historical verification records=%d; no model execution ran; historical claims retain their original commit and do not establish current task quality or evidence reuse", len(value.Models), activations, missing, stale, len(records))})
}
