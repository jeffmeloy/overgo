package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/finding"
	"overgo/internal/jsonfile"
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
	if !sameCensusModels(value.Models, censusModels(current)) {
		return fmt.Errorf("capability census: registered identity, availability or task/recipe set changed (snapshot=%d live=%d); record a new disposition before replacing the denominator", len(value.Models), len(current))
	}
	_, err = readCensusVerifications(ctx, store, value)
	return err
}

// sameCensusModels reports whether two census entry lists name the same
// registered identities, availability and task/recipe sets, in order.
func sameCensusModels(expected, actual []censusModel) bool {
	return slices.EqualFunc(expected, actual, func(expected, actual censusModel) bool {
		return expected.Model == actual.Model && expected.Location == actual.Location && expected.Present == actual.Present &&
			slices.Equal(expected.Capabilities, actual.Capabilities)
	})
}

type censusVerification struct {
	ID     artifact.ID                 `json:"id"`
	Record runrecord.ModelVerification `json:"record"`
}

// errCensusCheckpointAbsent says the store does not hold, at the sequence a
// census recorded, the commit it recorded there. A store rebuilt since the
// census was taken holds none of the commits the census counted among: a
// rebuild writes what it carries in commits of its own.
var errCensusCheckpointAbsent = errors.New("capability census: recorded store checkpoint is absent")

func readCensusVerifications(ctx context.Context, store *overgodb.Store, value capabilityCensus) ([]censusVerification, error) {
	checkpoint, found, err := store.CommitAt(ctx, value.StoreSequence)
	if err != nil {
		return nil, err
	}
	if !found || checkpoint.ID != value.StoreCommit {
		return nil, errCensusCheckpointAbsent
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

// censusPublication returns the batch that publishes a census just built,
// after the two steps every publication takes first.
func censusPublication(ctx context.Context, store *overgodb.Store, value capabilityCensus, memo *discovery.Memo) (artifact.Batch, error) {
	// Verify all referenced observations before publishing; evidence records
	// are historical snapshots, not substituted for task-native oracles.
	if _, err := readCensusVerifications(ctx, store, value); err != nil {
		return artifact.Batch{}, err
	}
	// Reuse the existing stat-validated digest memo for subsequent read-only
	// checks instead of hashing the same registered weights on every gate.
	if err := discovery.PublishMemo(ctx, store, memo); err != nil {
		return artifact.Batch{}, err
	}
	return capabilityCensusBatch(value)
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
		value, err = buildCapabilityCensus(ctx, store, commit, catalogLimit, memo)
		if err != nil {
			return err
		}
		batch, err := censusPublication(ctx, store, value, memo)
		if err != nil {
			return err
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
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
		if err := checkCapabilityCensus(ctx, store, value, catalogLimit); err != nil {
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

// censusBindingMode is the mode of a binding file: a committed fixture that
// anyone who reads the checkout may read.
const censusBindingMode = 0o644

// acceptedCensus is what a binding file holds: the census an acceptance froze
// (Base), the census accepted in its place (Current), and the finding and its
// closing that admitted the replacement. The file names its own base, so the
// command that accepts a new census is given no identity.
type acceptedCensus struct {
	Base, Current, Finding, Disposition artifact.ID
}

// checkCensusBinding requires the accepted checkpoint and its exact disposition.
//   - the historical base remains readable without its pre-rebuild checkpoint.
//   - capability and media consumers retain their distinct denominator checks.
func checkCensusBinding(ctx context.Context, store *overgodb.Store, path string) error {
	var binding acceptedCensus
	if err := jsonfile.DecodeStrict(path, &binding); err != nil {
		return err
	}
	value, err := capabilityCensusCodec.Require(ctx, store, binding.Base)
	if err != nil {
		return err
	}
	if binding.Current.Valid() || binding.Finding.Valid() || binding.Disposition.Valid() {
		if err := checkCensusDisposition(ctx, store, binding); err != nil {
			return err
		}
		value, err = capabilityCensusCodec.Require(ctx, store, binding.Current)
		if err != nil {
			return err
		}
	}
	_, err = readCensusVerifications(ctx, store, value)
	return err
}

// checkCensusDisposition holds a binding to the store: the current census is
// not the base, the finding is open and names both, and the finding's
// disposition is the bound one, closed, citing the finding.
func checkCensusDisposition(ctx context.Context, store artifact.Reader, binding acceptedCensus) error {
	if binding.Current == binding.Base || binding.Current.Kind() != artifact.KindEvidence {
		return errors.New("census disposition: distinct current census required")
	}
	content, found, err := artifact.ReadContent(ctx, store, binding.Finding)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("census disposition: finding absent")
	}
	original, err := finding.Parse(content.Data)
	if err != nil {
		return err
	}
	if original.Status != finding.StatusOpen || !slices.Contains(original.Evidence, binding.Base) || !slices.Contains(original.Evidence, binding.Current) {
		return errors.New("census disposition: finding does not bind both snapshots")
	}
	resolved, found, err := finding.Disposition(ctx, store, original.ID)
	if err != nil {
		return err
	}
	if !found || resolved.ID != binding.Disposition || resolved.Status != finding.StatusClosed || !slices.Contains(resolved.Evidence, original.ID) {
		return errors.New("census disposition: exact closed authority required")
	}
	return nil
}

// printCensusEntries prints, after sign, each of entries that others holds no
// equal of. Equal is what sameCensusModels says of the two entries, so a line
// is printed whenever that predicate says two censuses differ, and an entry
// that changed is printed from both sides.
func printCensusEntries(output io.Writer, sign string, entries, others []censusModel) {
	for _, entry := range entries {
		if !slices.ContainsFunc(others, func(other censusModel) bool {
			return sameCensusModels([]censusModel{entry}, []censusModel{other})
		}) {
			fmt.Fprintf(output, "%s %s %s present=%t %v\n", sign, entry.Model, entry.Location, entry.Present, entry.Capabilities)
		}
	}
}

// acceptCensus replaces the census the binding file at path accepts with the
// live one, built at commit. It refuses while the registered models are those
// of the census accepted before and the store still holds the checkpoint that
// census recorded, and otherwise prints what differs. With no
// reason it stops there, on a store it opened for reading. With a reason it
// publishes, in one store commit, the live census, a finding that names the
// base, the census accepted before and the new one, and that finding's
// closing with the reason; then it holds the new binding to the check its
// acceptance tests use and writes the file.
func acceptCensus(repository, commit, path, reason string, output io.Writer) error {
	ctx := context.Background()
	var binding acceptedCensus
	if err := jsonfile.DecodeStrict(path, &binding); err != nil {
		return err
	}
	open := overgodb.OpenReadOnly
	if reason != "" {
		open = overgodb.Open
	}
	store, err := open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	// A binding nothing was accepted into yet answers for its base.
	previous, err := capabilityCensusCodec.Require(ctx, store, cmp.Or(binding.Current, binding.Base))
	if err != nil {
		return err
	}
	memo := discovery.LoadMemo(ctx, store)
	value, err := buildCapabilityCensus(ctx, store, commit, catalogLimit, memo)
	if err != nil {
		return err
	}
	// The same models on a store that no longer holds the accepted census's
	// checkpoint are a census to take again: the store was rebuilt, and the
	// accepted census counts its verifications by a commit that is gone.
	if _, err := readCensusVerifications(ctx, store, previous); sameCensusModels(previous.Models, value.Models) && !errors.Is(err, errCensusCheckpointAbsent) {
		return fmt.Errorf("capability census: the registered models are those of the accepted census %s; there is nothing to accept", previous.ID)
	}
	printCensusEntries(output, "-", previous.Models, value.Models)
	printCensusEntries(output, "+", value.Models, previous.Models)
	if reason == "" {
		_, err := fmt.Fprintln(output, "nothing written: give -reason to accept this census")
		return err
	}
	filed, filing, err := finding.NewTextBatch("The registered models differ from the accepted capability census", finding.SeverityLow,
		[]string{path}, []string{binding.Base.String(), previous.ID.String(), value.ID.String()},
		"go run ./cmd/validate -accept-census "+path+" -reason <why the difference is accepted>",
		"go run ./cmd/validate -check-census "+value.ID.String())
	if err != nil {
		return err
	}
	closed, closing, err := finding.NewDispositionBatch(ctx, store, filed, finding.StatusClosed, reason)
	if err != nil {
		return err
	}
	batch, err := censusPublication(ctx, store, value, memo)
	if err != nil {
		return err
	}
	// One commit carries the three documents, so a failure between them cannot
	// leave a census nothing accepts or a finding nothing closes.
	for _, part := range []artifact.Batch{filing, closing} {
		batch.Contents = append(batch.Contents, part.Contents...)
		batch.Lineage = append(batch.Lineage, part.Lineage...)
		batch.Aliases = append(batch.Aliases, part.Aliases...)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return err
	}
	binding.Current, binding.Finding, binding.Disposition = value.ID, filed.ID, closed.ID
	if err := checkCensusDisposition(ctx, store, binding); err != nil {
		return err
	}
	if err := jsonfile.Write(path, binding, censusBindingMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "accepted into %s: %+v\n", path, binding)
	return err
}
