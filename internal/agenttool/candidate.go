package agenttool

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	// CandidateCatalogMediaType identifies encoded candidate-catalog documents.
	CandidateCatalogMediaType = "application/vnd.overgo.agent-tool-candidate+json"
	// CandidateCatalogSchema identifies the exact stored candidate-catalog schema.
	CandidateCatalogSchema = "overgo/agent-tool-candidate/v1"
	// CandidateVerificationMediaType identifies encoded candidate-verification documents.
	CandidateVerificationMediaType = "application/vnd.overgo.agent-tool-candidate-verification+json"
	// CandidateVerificationSchema identifies the exact stored candidate-verification schema.
	CandidateVerificationSchema = "overgo/agent-tool-candidate-verification/v2"
	// ActiveCatalogAlias names the store alias of the active tool catalog.
	ActiveCatalogAlias = "tool.catalog.active"
)

// CandidateCatalog binds an inactive snapshot to its exact discovery source.
type CandidateCatalog struct {
	Version  uint16      `json:"version"`
	Source   artifact.ID `json:"source"`
	Snapshot artifact.ID `json:"snapshot"`
	ID       artifact.ID `json:"-"`
}

// CandidateVerification is the immutable activation grant emitted after every
// snapshot entry is loaded, matched, and checked against the verifier host.
type CandidateVerification struct {
	Version     uint16                      `json:"version"`
	Candidate   artifact.ID                 `json:"candidate"`
	Source      artifact.ID                 `json:"source"`
	Snapshot    artifact.ID                 `json:"snapshot"`
	ToolSet     artifact.ID                 `json:"tool_set"`
	Schemas     artifact.ID                 `json:"schemas"`
	Endpoints   artifact.ID                 `json:"endpoints"`
	Host        artifact.ID                 `json:"host"`
	Manuals     uint32                      `json:"manuals"`
	Authorities []CandidateAuthorityBinding `json:"authorities,omitempty"`
	ID          artifact.ID                 `json:"-"`
}

// CandidateAuthorityBinding records mutable policy state consumed by verification.
type CandidateAuthorityBinding struct {
	Alias    string      `json:"alias"`
	Artifact artifact.ID `json:"artifact"`
}

// CandidateStaging reports one inactive publication.
type CandidateStaging struct {
	Commit      artifact.CommitID `json:"commit"`
	CandidateID artifact.ID       `json:"candidate_id"`
	SnapshotID  artifact.ID       `json:"snapshot_id"`
	Candidate   CandidateCatalog  `json:"candidate"`
	Snapshot    CatalogSnapshot   `json:"snapshot"`
}

// CatalogActivation reports one evidence-gated active transition.
type CatalogActivation struct {
	Commit       artifact.CommitID `json:"commit"`
	Candidate    artifact.ID       `json:"candidate"`
	Verification artifact.ID       `json:"verification"`
	Snapshot     artifact.ID       `json:"snapshot"`
	Changed      bool              `json:"changed"`
	Coverage     CatalogCoverage   `json:"coverage"`
}

var candidateCatalogCodec = artifact.JSONDocumentCodec(
	"agent tool candidate", artifact.KindProfile, CandidateCatalogMediaType, CandidateCatalogSchema,
	canonicalizeCandidateCatalog,
	func(value CandidateCatalog) artifact.ID { return value.ID },
	func(value *CandidateCatalog, id artifact.ID) { value.ID = id },
	func(value CandidateCatalog) CandidateCatalog { return value },
)

var candidateVerificationCodec = artifact.JSONDocumentCodec(
	"agent tool candidate verification", artifact.KindEvidence,
	CandidateVerificationMediaType, CandidateVerificationSchema,
	canonicalizeCandidateVerification,
	func(value CandidateVerification) artifact.ID { return value.ID },
	func(value *CandidateVerification, id artifact.ID) { value.ID = id },
	func(value CandidateVerification) CandidateVerification {
		value.Authorities = slices.Clone(value.Authorities)
		return value
	},
)

// StageManualCandidates commits discovery output and lineage without changing
// a registered or active alias.
func StageManualCandidates(
	ctx context.Context,
	repository artifact.Repository,
	source artifact.Content,
	manuals []Manual,
) (CandidateStaging, error) {
	if ctx == nil || repository == nil {
		return CandidateStaging{}, errors.New("agent tool: candidate staging authority is absent")
	}
	if err := source.Validate(); err != nil || source.Descriptor.ID.Kind() != artifact.KindFile {
		return CandidateStaging{}, errors.Join(errors.New("agent tool: candidate source is invalid"), err)
	}
	ordered := slices.Clone(manuals)
	slices.SortFunc(ordered, func(left, right Manual) int { return artifact.CompareID(left.ID, right.ID) })
	snapshot, err := NewCatalogSnapshot(ordered)
	if err != nil {
		return CandidateStaging{}, err
	}
	candidate, err := candidateCatalogCodec.New(CandidateCatalog{
		Version: artifact.InitialDocumentVersion, Source: source.Descriptor.ID, Snapshot: snapshot.ID,
	})
	if err != nil {
		return CandidateStaging{}, err
	}
	contents := []artifact.Content{source.Clone()}
	lineage := []artifact.Lineage{}
	capabilities := map[artifact.ID]bool{}
	manualIDs := make([]artifact.ID, len(ordered))
	for index, manual := range ordered {
		if err := appendManualDocuments(&contents, &lineage, manual, capabilities); err != nil {
			return CandidateStaging{}, err
		}
		manualIDs[index] = manual.ID
	}
	snapshotContent, err := snapshot.ArtifactContent()
	if err != nil {
		return CandidateStaging{}, err
	}
	candidateContent, err := candidateCatalogCodec.Content(candidate)
	if err != nil {
		return CandidateStaging{}, err
	}
	contents = append(contents, snapshotContent, candidateContent)
	lineage = append(lineage, artifact.DependencyLineage(snapshot.ID, manualIDs...)...)
	lineage = append(lineage, artifact.DependencyLineage(candidate.ID, source.Descriptor.ID, snapshot.ID)...)
	batch, err := artifact.NewDocumentBatch(
		"agent-tool/candidate/"+candidate.ID.String(), contents, lineage, nil,
	)
	if err != nil {
		return CandidateStaging{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return CandidateStaging{}, err
	}
	return CandidateStaging{
		Commit: commit, CandidateID: candidate.ID, SnapshotID: snapshot.ID,
		Candidate: candidate, Snapshot: snapshot,
	}, nil
}

// PublishCandidateVerification reloads and verifies every staged boundary and
// commits success evidence. Failure publishes nothing.
func PublishCandidateVerification(
	ctx context.Context,
	repository artifact.Repository,
	candidateID artifact.ID,
	executor *Executor,
) (CandidateVerification, artifact.CommitID, error) {
	if ctx == nil || repository == nil || executor == nil {
		return CandidateVerification{}, artifact.CommitID{}, errors.New("agent tool: candidate verifier is absent")
	}
	candidate, snapshot, manuals, authorities, err := inspectCandidate(ctx, repository, candidateID, executor)
	if err != nil {
		return CandidateVerification{}, artifact.CommitID{}, err
	}
	binding, err := candidateGrantBinding(ctx, repository, manuals, executor)
	if err != nil {
		return CandidateVerification{}, artifact.CommitID{}, err
	}
	verification, err := candidateVerificationCodec.New(CandidateVerification{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID,
		Source: candidate.Source, Snapshot: candidate.Snapshot, Manuals: uint32(len(manuals)),
		ToolSet: binding.ToolSet, Schemas: binding.Schemas, Endpoints: binding.Endpoints, Host: binding.Host,
		Authorities: authorities,
	})
	if err != nil {
		return CandidateVerification{}, artifact.CommitID{}, err
	}
	content, err := candidateVerificationCodec.Content(verification)
	if err != nil {
		return CandidateVerification{}, artifact.CommitID{}, err
	}
	parents := []artifact.ID{candidate.ID, candidate.Source, snapshot.ID}
	for _, manual := range manuals {
		parents = append(parents, manual.ID)
	}
	for _, authority := range verification.Authorities {
		parents = append(parents, authority.Artifact)
	}
	batch, err := artifact.NewDocumentBatch(
		"agent-tool/candidate-verification/"+verification.ID.String(),
		[]artifact.Content{content}, artifact.DependencyLineage(verification.ID, parents...), nil,
	)
	if err != nil {
		return CandidateVerification{}, artifact.CommitID{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	return verification, commit, err
}

// ActivateCandidateCatalog atomically binds the exact candidate manuals and
// active snapshot after requiring matching committed verification evidence.
func ActivateCandidateCatalog(
	ctx context.Context,
	repository artifact.Repository,
	candidateID, verificationID artifact.ID,
) (CatalogActivation, error) {
	if ctx == nil || repository == nil {
		return CatalogActivation{}, errors.New("agent tool: candidate activation authority is absent")
	}
	candidate, err := candidateCatalogCodec.Require(ctx, repository, candidateID)
	if err != nil {
		return CatalogActivation{}, err
	}
	verification, err := candidateVerificationCodec.Require(ctx, repository, verificationID)
	if err != nil {
		return CatalogActivation{}, err
	}
	if verification.Candidate != candidate.ID || verification.Source != candidate.Source || verification.Snapshot != candidate.Snapshot {
		return CatalogActivation{}, errors.New("agent tool: candidate verification binds different authority")
	}
	for _, authority := range verification.Authorities {
		current, found, err := repository.ResolveAlias(ctx, authority.Alias)
		if err != nil || !found || current != authority.Artifact {
			return CatalogActivation{}, errors.Join(errors.New("agent tool: candidate verification policy authority is stale"), err)
		}
	}
	snapshot, err := RequireCatalogSnapshot(ctx, repository, candidate.Snapshot)
	if err != nil || int(verification.Manuals) != len(snapshot.Entries) {
		return CatalogActivation{}, errors.Join(errors.New("agent tool: candidate verification coverage differs"), err)
	}
	manuals, err := candidateManuals(ctx, repository, snapshot)
	if err != nil {
		return CatalogActivation{}, err
	}
	binding, err := candidateGrantBinding(ctx, repository, manuals, NewOperatorExecutor())
	if err != nil || verification.ToolSet != binding.ToolSet || verification.Schemas != binding.Schemas ||
		verification.Endpoints != binding.Endpoints || verification.Host != binding.Host {
		return CatalogActivation{}, errors.Join(errors.New("agent tool: candidate activation grant is stale"), err)
	}
	coverage, err := InspectManualCatalog(ctx, repository, manuals)
	if err != nil {
		return CatalogActivation{}, err
	}
	active, found, err := repository.ResolveAlias(ctx, ActiveCatalogAlias)
	if err != nil {
		return CatalogActivation{}, err
	}
	if coverage.Complete && found && active == snapshot.ID {
		commit, _ := repository.Head()
		return CatalogActivation{
			Commit: commit, Candidate: candidate.ID, Verification: verification.ID,
			Snapshot: snapshot.ID, Coverage: coverage,
		}, nil
	}
	batch, err := manualCatalogBatch(ctx, repository, manuals)
	if err != nil {
		return CatalogActivation{}, err
	}
	if !found || active != snapshot.ID {
		binding := artifact.AliasBinding{Name: ActiveCatalogAlias, Target: snapshot.ID}
		if found {
			binding.Previous = &active
		}
		batch.Aliases = append(batch.Aliases, binding)
	}
	batch.Key = "agent-tool/activate/" + candidate.ID.String() + "/" + verification.ID.String()
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return CatalogActivation{}, err
	}
	coverage, err = InspectManualCatalog(ctx, repository, manuals)
	if err != nil || !coverage.Complete {
		return CatalogActivation{}, errors.Join(errors.New("agent tool: candidate activation is incomplete"), err)
	}
	return CatalogActivation{
		Commit: commit, Candidate: candidate.ID, Verification: verification.ID,
		Snapshot: snapshot.ID, Changed: true, Coverage: coverage,
	}, nil
}

func inspectCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidateID artifact.ID,
	executor *Executor,
) (CandidateCatalog, CatalogSnapshot, []Manual, []CandidateAuthorityBinding, error) {
	candidate, err := candidateCatalogCodec.Require(ctx, reader, candidateID)
	if err != nil {
		return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, err
	}
	if _, found, err := artifact.ReadContent(ctx, reader, candidate.Source); err != nil || !found {
		return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, errors.Join(errors.New("agent tool: candidate source is absent"), err)
	}
	snapshot, err := RequireCatalogSnapshot(ctx, reader, candidate.Snapshot)
	if err != nil {
		return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, err
	}
	manuals, err := candidateManuals(ctx, reader, snapshot)
	if err != nil {
		return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, err
	}
	authorities := map[string]artifact.ID{}
	for _, manual := range manuals {
		if _, found := executor.adapters[manual.Transport.Kind]; !found {
			return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, fmt.Errorf("agent tool: candidate transport %q has no adapter", manual.Transport.Kind)
		}
		if err := CheckArgvAuthority(ctx, reader, manual); err != nil {
			return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, err
		}
		if manual.Transport.Kind == TransportArgv {
			policy, found, err := ResolveArgvPolicy(ctx, reader)
			if err != nil || !found {
				return CandidateCatalog{}, CatalogSnapshot{}, nil, nil, errors.Join(errors.New("agent tool: candidate argv policy is absent"), err)
			}
			authorities[ArgvPolicyAlias] = policy.ID
		}
	}
	bindings := make([]CandidateAuthorityBinding, 0, len(authorities))
	for alias, id := range authorities {
		bindings = append(bindings, CandidateAuthorityBinding{Alias: alias, Artifact: id})
	}
	sort.Slice(bindings, func(left, right int) bool { return bindings[left].Alias < bindings[right].Alias })
	return candidate, snapshot, manuals, bindings, nil
}

func candidateManuals(ctx context.Context, reader artifact.Reader, snapshot CatalogSnapshot) ([]Manual, error) {
	manuals := make([]Manual, len(snapshot.Entries))
	for index, entry := range snapshot.Entries {
		manual, err := RequireManual(ctx, reader, entry.Manual)
		if err != nil {
			return nil, err
		}
		if manual.Name != entry.Name || manual.Description != entry.Description ||
			manual.Effect != entry.Effect || !slices.Equal(manual.Arguments, entry.Arguments) {
			return nil, errors.New("agent tool: candidate snapshot projection differs from its manual")
		}
		manuals[index] = manual
	}
	return manuals, nil
}

func canonicalizeCandidateCatalog(candidate *CandidateCatalog) error {
	if candidate == nil || candidate.Version != artifact.InitialDocumentVersion ||
		candidate.Source.Kind() != artifact.KindFile || candidate.Snapshot.Kind() != artifact.KindProfile {
		return errors.New("agent tool: invalid candidate catalog")
	}
	return nil
}

func canonicalizeCandidateVerification(verification *CandidateVerification) error {
	if verification == nil || verification.Version != artifact.InitialDocumentVersion ||
		verification.Candidate.Kind() != artifact.KindProfile || verification.Source.Kind() != artifact.KindFile ||
		verification.Snapshot.Kind() != artifact.KindProfile || verification.ToolSet.Kind() != artifact.KindProfile ||
		verification.Schemas.Kind() != artifact.KindProfile || verification.Endpoints.Kind() != artifact.KindProfile ||
		verification.Host.Kind() != artifact.KindProfile || verification.Manuals == 0 {
		return errors.New("agent tool: invalid candidate verification")
	}
	sort.Slice(verification.Authorities, func(left, right int) bool {
		return verification.Authorities[left].Alias < verification.Authorities[right].Alias
	})
	for index, authority := range verification.Authorities {
		if authority.Alias == "" || !authority.Artifact.Valid() ||
			index > 0 && verification.Authorities[index-1].Alias == authority.Alias {
			return errors.New("agent tool: invalid candidate verification authority")
		}
	}
	return nil
}

type candidateGrantDigests struct {
	ToolSet   artifact.ID
	Schemas   artifact.ID
	Endpoints artifact.ID
	Host      artifact.ID
}

func candidateGrantBinding(
	ctx context.Context,
	reader artifact.Reader,
	manuals []Manual,
	executor *Executor,
) (candidateGrantDigests, error) {
	if ctx == nil || reader == nil || executor == nil {
		return candidateGrantDigests{}, errors.New("agent tool: candidate grant authority is absent")
	}
	type manualSchema struct {
		Manual     artifact.ID `json:"manual"`
		Arguments  []Field     `json:"arguments,omitempty"`
		Capability artifact.ID `json:"capability_schema,omitzero"`
	}
	type endpointScope struct {
		Manual    artifact.ID `json:"manual"`
		Transport Transport   `json:"transport"`
	}
	manualIDs := make([]artifact.ID, len(manuals))
	schemas := make([]manualSchema, len(manuals))
	endpoints := make([]endpointScope, len(manuals))
	for index, manual := range manuals {
		manualIDs[index] = manual.ID
		schemas[index] = manualSchema{Manual: manual.ID, Arguments: slices.Clone(manual.Arguments)}
		if manual.Capability.Valid() {
			identity, err := runrecord.RequireCapabilityIdentity(ctx, reader, manual.Capability)
			if err != nil {
				return candidateGrantDigests{}, err
			}
			schemas[index].Capability = identity.Schema
		}
		transport := manual.Transport
		transport.Args = slices.Clone(transport.Args)
		endpoints[index] = endpointScope{Manual: manual.ID, Transport: transport}
	}
	transports := make([]TransportKind, 0, len(executor.adapters))
	for kind := range executor.adapters {
		transports = append(transports, kind)
	}
	slices.Sort(transports)
	toolSet, err := artifact.JSONID(artifact.KindProfile, struct {
		Version uint16        `json:"version"`
		Manuals []artifact.ID `json:"manuals"`
	}{Version: artifact.InitialDocumentVersion, Manuals: manualIDs})
	if err != nil {
		return candidateGrantDigests{}, err
	}
	schemaDigest, err := artifact.JSONID(artifact.KindProfile, schemas)
	if err != nil {
		return candidateGrantDigests{}, err
	}
	endpointDigest, err := artifact.JSONID(artifact.KindProfile, endpoints)
	if err != nil {
		return candidateGrantDigests{}, err
	}
	hostDigest, err := artifact.JSONID(artifact.KindProfile, struct {
		OS         string          `json:"os"`
		Arch       string          `json:"arch"`
		Transports []TransportKind `json:"transports"`
	}{OS: runtime.GOOS, Arch: runtime.GOARCH, Transports: transports})
	if err != nil {
		return candidateGrantDigests{}, err
	}
	return candidateGrantDigests{ToolSet: toolSet, Schemas: schemaDigest, Endpoints: endpointDigest, Host: hostDigest}, nil
}
