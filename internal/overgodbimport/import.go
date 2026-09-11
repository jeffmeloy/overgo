//overgo:runtime-inputs caller

// Package overgodbimport ingests wire-format import streams into the
// catalog, verifying digests and upgrading legacy profile schemas; the
// wire labels keep their original repodb names because published
// history is immutable.
package overgodbimport

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/finding"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/objectstore"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
	"overgo/internal/textcheck"
)

const (
	// Version is the import stream wire version this reader accepts.
	Version uint16 = 1
	// SourceMediaType is the immutable wire label committed documents carry.
	SourceMediaType = "application/vnd.overgo.repodb-import+json"
	// SourceSchema is the immutable wire schema committed documents carry.
	SourceSchema         = "overgo/repodb-import/v1"
	maxLineBytes         = artifact.MaxContentBytes
	maxImportBytes       = 512 << 20
	maxRecords           = 1_000_000
	legacyProfileSchema  = "overgo/model-profile/v1"
	renamedProfileSchema = "overgo/model-profile/v2"
)

var sourceContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: SourceMediaType, Schema: SourceSchema,
}

// Header is the first stream line: type, version, provenance, and
// declared record counts the body must reconcile against.
type Header struct {
	Type         string            `json:"type"`
	Version      uint16            `json:"version"`
	SourceCommit string            `json:"source_commit"`
	Counts       map[string]uint64 `json:"counts"`
}

type componentRecord struct {
	Role     string `json:"role"`
	Ordinal  uint32 `json:"ordinal"`
	Name     string `json:"name"`
	Artifact string `json:"artifact"`
}

type wireRecord struct {
	Type       string            `json:"type"`
	Name       string            `json:"name,omitzero"`
	Kind       string            `json:"kind,omitzero"`
	Path       string            `json:"path,omitzero"`
	Document   json.RawMessage   `json:"document,omitempty"`
	MediaType  string            `json:"media_type,omitzero"`
	Schema     string            `json:"schema,omitzero"`
	Components []componentRecord `json:"components,omitempty"`
	Child      string            `json:"child,omitzero"`
	Parent     string            `json:"parent,omitzero"`
	Relation   string            `json:"relation,omitzero"`
	Target     string            `json:"target,omitzero"`
	Previous   string            `json:"previous,omitzero"`
}

type sourceBody struct {
	Version      uint16            `json:"version"`
	SourceCommit string            `json:"source_commit"`
	ExportSHA256 string            `json:"export_sha256"`
	Counts       map[string]uint64 `json:"counts"`
}

// Result reports the committed import: the commit, the provenance
// document, and the imported artifacts by name.
type Result struct {
	Commit artifact.CommitID
	Source artifact.ID
	Names  map[string]artifact.ID
}

type pendingNode struct {
	record wireRecord
	kind   artifact.Kind
}

// ImportStored ingests file records through streaming object storage before
// committing the import's manifests, lineage, aliases, and source evidence.
func ImportStored(
	ctx context.Context,
	repository artifact.Repository,
	root string,
	objects *objectstore.Store,
	input io.Reader,
) (Result, error) {
	if objects == nil {
		return Result{}, errors.New("repodb import: object store is required")
	}
	return importRecords(ctx, repository, root, objects, input)
}

func importRecords(
	ctx context.Context,
	repository artifact.Repository,
	root string,
	objects *objectstore.Store,
	input io.Reader,
) (Result, error) {
	if ctx == nil || repository == nil || input == nil {
		return Result{}, errors.New("repodb import: nil input")
	}
	if strings.TrimSpace(root) == "" {
		return Result{}, errors.New("repodb import: artifact root is empty")
	}
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		return Result{}, err
	}
	absoluteRoot, err = filepath.EvalSymlinks(absoluteRoot)
	if err != nil {
		return Result{}, fmt.Errorf("repodb import: resolve artifact root: %w", err)
	}
	raw, err := io.ReadAll(io.LimitReader(input, maxImportBytes+1))
	if err != nil {
		return Result{}, err
	}
	if len(raw) > maxImportBytes {
		return Result{}, errors.New("repodb import: stream exceeds size limit")
	}
	lines, err := canonicalLines(raw)
	if err != nil || len(lines) < 2 {
		return Result{}, errors.New("repodb import: invalid JSONL stream")
	}
	var header Header
	if err := strictjson.DecodeBytes(lines[0], &header); err != nil || header.Type != "manifest" ||
		header.Version != Version || !validCommit(header.SourceCommit) || len(header.Counts) == 0 {
		return Result{}, errors.New("repodb import: invalid manifest")
	}
	if canonical, _ := json.Marshal(header); !bytes.Equal(canonical, lines[0]) {
		return Result{}, errors.New("repodb import: non-canonical manifest")
	}
	records := make([]wireRecord, len(lines)-1)
	actualCounts := make(map[string]uint64)
	for index, line := range lines[1:] {
		if err := strictjson.DecodeBytes(line, &records[index]); err != nil {
			return Result{}, fmt.Errorf("repodb import: record %d: %w", index+1, err)
		}
		if canonical, _ := json.Marshal(records[index]); !bytes.Equal(canonical, line) {
			return Result{}, fmt.Errorf("repodb import: record %d is not canonical", index+1)
		}
		actualCounts[recordCountKey(records[index])]++
	}
	if !equalCounts(header.Counts, actualCounts) {
		return Result{}, errors.New("repodb import: manifest counts differ")
	}
	digest := exportDigest(raw)
	batch := artifact.Batch{Key: "import/" + header.SourceCommit + "/" + digest}
	names := make(map[string]artifact.ID)
	pending := make([]pendingNode, 0)
	for _, record := range records {
		switch record.Type {
		case "artifact":
			kind, err := artifact.ParseKind(record.Kind)
			if err != nil || !textcheck.Bounded(record.Name, 512, "\x00\r\n") {
				return Result{}, errors.New("repodb import: invalid artifact record")
			}
			if _, duplicate := names[record.Name]; duplicate || containsPending(pending, record.Name) {
				return Result{}, errors.New("repodb import: duplicate logical name")
			}
			if record.Path != "" {
				if objects != nil {
					publication, err := importStoredFile(ctx, absoluteRoot, objects, kind, record)
					if err != nil {
						return Result{}, err
					}
					names[record.Name] = publication.Descriptor.ID
					continue
				}
				id, descriptor, location, err := importFile(absoluteRoot, kind, record.Path)
				if err != nil {
					return Result{}, err
				}
				names[record.Name] = id
				batch.Artifacts = append(batch.Artifacts, descriptor)
				batch.Locations = append(batch.Locations, location)
			} else if len(record.Document) != 0 && record.MediaType != "" && record.Schema != "" {
				pending = append(pending, pendingNode{record: record, kind: kind})
			} else {
				return Result{}, errors.New("repodb import: artifact has no valid source")
			}
		case "manifest":
			kind, err := artifact.ParseKind(record.Kind)
			if err != nil || !textcheck.Bounded(record.Name, 512, "\x00\r\n") || len(record.Components) == 0 ||
				containsPending(pending, record.Name) {
				return Result{}, errors.New("repodb import: invalid manifest record")
			}
			if _, duplicate := names[record.Name]; duplicate {
				return Result{}, errors.New("repodb import: duplicate logical name")
			}
			pending = append(pending, pendingNode{record: record, kind: kind})
		case "lineage", "alias":
		default:
			return Result{}, fmt.Errorf("repodb import: unknown record type %q", record.Type)
		}
	}
	for len(pending) > 0 {
		progress := false
		rest := pending[:0]
		for _, node := range pending {
			resolved, id, content, manifest, err := resolveNode(node, names)
			if err != nil {
				return Result{}, err
			}
			if !resolved {
				rest = append(rest, node)
				continue
			}
			names[node.record.Name] = id
			if content != nil {
				batch.Contents = append(batch.Contents, *content)
				lineage, lineageErr := nativeDocumentLineage(*content)
				if lineageErr != nil {
					return Result{}, lineageErr
				}
				batch.Lineage = append(batch.Lineage, lineage...)
			} else {
				batch.Manifests = append(batch.Manifests, *manifest)
				batch.Lineage = append(batch.Lineage, manifest.Lineage()...)
			}
			progress = true
		}
		pending = rest
		if !progress {
			return Result{}, errors.New("repodb import: unresolved or cyclic artifact references")
		}
	}
	for _, record := range records {
		switch record.Type {
		case "lineage":
			child, childOK := names[record.Child]
			parent, parentOK := names[record.Parent]
			relation, err := artifact.ParseRelation(record.Relation)
			if err != nil || !childOK || !parentOK {
				return Result{}, errors.New("repodb import: invalid lineage record")
			}
			batch.Lineage = append(batch.Lineage, artifact.Lineage{
				Child: child, Parent: parent, Relation: relation,
			})
		case "alias":
			target, ok := names[record.Target]
			if !ok || record.Name == "" {
				return Result{}, errors.New("repodb import: invalid alias record")
			}
			binding := artifact.AliasBinding{Name: record.Name, Target: target}
			if record.Previous != "" {
				previous, ok := names[record.Previous]
				if !ok {
					return Result{}, errors.New("repodb import: alias previous target is unknown")
				}
				binding.Previous = artifact.IDPointer(previous)
			}
			batch.Aliases = append(batch.Aliases, binding)
		}
	}
	sourceContent, sourceID, err := importSource(header, raw)
	if err != nil {
		return Result{}, err
	}
	batch.Contents = append(batch.Contents, sourceContent)
	for _, id := range names {
		batch.Lineage = append(batch.Lineage, artifact.Lineage{
			Child: id, Parent: sourceID, Relation: artifact.RelationConvertedFrom,
		})
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return Result{}, err
	}
	return Result{Commit: commit, Source: sourceID, Names: cloneNames(names)}, nil
}

func canonicalLines(raw []byte) ([][]byte, error) {
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 64<<10), maxLineBytes)
	lines := make([][]byte, 0)
	for scanner.Scan() {
		line := slices.Clone(scanner.Bytes())
		if len(bytes.TrimSpace(line)) == 0 || !bytes.Equal(line, bytes.TrimSpace(line)) {
			return nil, errors.New("repodb import: non-canonical line")
		}
		lines = append(lines, line)
		if len(lines) > maxRecords+1 {
			return nil, errors.New("repodb import: record limit exceeded")
		}
	}
	return lines, scanner.Err()
}

func resolveNode(
	node pendingNode,
	names map[string]artifact.ID,
) (bool, artifact.ID, *artifact.Content, *artifact.Manifest, error) {
	if node.record.Type == "manifest" {
		components := make([]artifact.Component, len(node.record.Components))
		for index, source := range node.record.Components {
			id, ok := names[source.Artifact]
			if !ok {
				return false, artifact.ID{}, nil, nil, nil
			}
			role, err := artifact.ParseComponentRole(source.Role)
			if err != nil {
				return false, artifact.ID{}, nil, nil, err
			}
			components[index] = artifact.Component{
				Role: role, Ordinal: source.Ordinal, Name: source.Name, Artifact: id,
			}
		}
		manifest, err := artifact.NewManifest(node.kind, components)
		return true, manifest.ID, nil, &manifest, err
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(node.record.Document))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	resolved, complete, err := resolveReferences(value, names)
	if err != nil || !complete {
		return false, artifact.ID{}, nil, nil, err
	}
	data, err := json.Marshal(resolved)
	if err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	// Known documents are canonicalized by their OWN codec before identity:
	// the reference-resolution marshal above is generic and cannot know each
	// type's canonical field order, and stored bytes are identity.
	data, schema, err := canonicalizeKnownDocument(node.record.MediaType, node.record.Schema, data)
	if err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	id, err := artifact.IdentifyBytes(node.kind, data)
	if err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	content := artifact.Content{Descriptor: artifact.Descriptor{
		ID: id, Size: uint64(len(data)), MediaType: node.record.MediaType, Schema: schema,
	}, Data: data}
	if err := content.Validate(); err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	return true, id, &content, nil, nil
}

// canonicalizeKnownDocument returns the domain codec's canonical bytes for a
// known media type (admitting any field order), and the input unchanged for
// unknown types. Each codec owns its canonical-form fact.
func canonicalizeKnownDocument(mediaType, schema string, data []byte) ([]byte, string, error) {
	canonical := data
	resolvedSchema := schema
	var err error
	requireSchema := func(allowed ...string) {
		if !slices.Contains(allowed, schema) {
			err = errors.New("schema differs from media type")
		}
	}
	switch mediaType {
	case recipe.MediaType:
		requireSchema(recipe.Schema)
		if err != nil {
			break
		}
		_, canonical, err = recipe.NormalizeDefinition(data)
	case recipe.LifecycleMediaType:
		requireSchema(recipe.LifecycleSchema)
		if err != nil {
			break
		}
		_, canonical, err = recipe.NormalizeLifecycleEvent(data)
	case recipe.DecisionMediaType:
		requireSchema(recipe.DecisionSchema)
		if err != nil {
			break
		}
		_, canonical, err = recipe.NormalizeDecision(data)
	case modelrecipe.ProfileMediaType:
		requireSchema(modelrecipe.ProfileSchema, legacyProfileSchema, renamedProfileSchema)
		if err != nil {
			break
		}
		switch schema {
		case legacyProfileSchema:
			canonical, err = upgradeLegacyProfile(data)
			resolvedSchema = modelrecipe.ProfileSchema
		case renamedProfileSchema:
			canonical, err = upgradeRenamedProfile(data)
			resolvedSchema = modelrecipe.ProfileSchema
		default:
			_, canonical, err = modelrecipe.NormalizeProfileDocument(data)
		}
	case modelrecipe.CatalogProfileDerivationMediaType:
		requireSchema(modelrecipe.CatalogProfileDerivationSchema)
		if err != nil {
			break
		}
		_, canonical, err = modelrecipe.NormalizeCatalogProfileDerivation(data)
	case modelrecipe.ModelDefinitionMediaType:
		requireSchema(modelrecipe.ModelDefinitionSchema)
		if err != nil {
			break
		}
		_, canonical, err = modelrecipe.NormalizeModelDefinitionDocument(data)
	case dataset.MediaType:
		requireSchema(dataset.Schema)
		if err != nil {
			break
		}
		_, canonical, err = dataset.Normalize(data)
	case dataset.MembershipMediaType:
		requireSchema(dataset.MembershipSchema)
		if err != nil {
			break
		}
		_, canonical, err = dataset.NormalizeMembership(data)
	case closureledger.MediaType:
		requireSchema(closureledger.Schema)
		if err != nil {
			break
		}
		_, canonical, err = closureledger.Normalize(data)
	case finding.MediaType:
		requireSchema(finding.Schema)
		if err != nil {
			break
		}
		_, canonical, err = finding.Normalize(data)
	case runrecord.RunMediaType:
		requireSchema(runrecord.RunSchema, runrecord.LegacyRunSchema)
		if err != nil {
			break
		}
		_, canonical, err = runrecord.NormalizeRun(data)
	case runrecord.EvaluationMediaType:
		requireSchema(runrecord.EvaluationSchema)
		if err != nil {
			break
		}
		_, canonical, err = runrecord.NormalizeEvaluation(data)
	case evaluation.EvaluationPlanMediaType:
		requireSchema(evaluation.EvaluationPlanSchema)
		if err != nil {
			break
		}
		_, canonical, err = evaluation.NormalizeEvaluationPlan(data)
	case evaluation.EvaluationEvidenceMediaType:
		requireSchema(evaluation.EvaluationEvidenceSchema)
		if err != nil {
			break
		}
		_, canonical, err = evaluation.NormalizeEvaluationEvidence(data)
	case runrecord.EnvironmentMediaType:
		requireSchema(runrecord.EnvironmentSchema)
		if err != nil {
			break
		}
		_, canonical, err = runrecord.NormalizeEnvironment(data)
	case runrecord.AdvisoryMediaType:
		requireSchema(runrecord.AdvisorySchema)
		if err != nil {
			break
		}
		_, canonical, err = runrecord.NormalizeAdvisory(data)
	case runrecord.GateMediaType:
		requireSchema(runrecord.GateSchema)
		if err != nil {
			break
		}
		_, canonical, err = runrecord.NormalizeGateResult(data)
	case modelartifact.TensorInventoryMediaType:
		requireSchema(modelartifact.TensorInventorySchema)
		if err != nil {
			break
		}
		_, canonical, err = modelartifact.NormalizeTensorInventoryDocument(data)
	case modelartifact.TensorMeasurementMediaType:
		requireSchema(modelartifact.TensorMeasurementSchema)
		if err != nil {
			break
		}
		_, canonical, err = modelartifact.NormalizeTensorMeasurementDocument(data)
	}
	if err != nil {
		return nil, "", fmt.Errorf("repodb import: invalid %q document: %w", mediaType, err)
	}
	return canonical, resolvedSchema, nil
}

// nativeDocumentLineage maps imported execution documents into OvergoDB's
// authority graph. External lineage may add provenance but cannot replace the
// relationships derived by the native codecs.
func nativeDocumentLineage(content artifact.Content) ([]artifact.Lineage, error) {
	switch content.Descriptor.MediaType {
	case runrecord.RunMediaType:
		value, err := runrecord.ParseRun(content.Data)
		return value.Lineage(), err
	case runrecord.EvaluationMediaType:
		value, err := runrecord.ParseEvaluation(content.Data)
		return value.Lineage(), err
	case evaluation.EvaluationPlanMediaType:
		value, _, err := evaluation.NormalizeEvaluationPlan(content.Data)
		return value.Lineage(), err
	case evaluation.EvaluationEvidenceMediaType:
		value, _, err := evaluation.NormalizeEvaluationEvidence(content.Data)
		return value.Lineage(), err
	default:
		return nil, nil
	}
}

func upgradeLegacyProfile(data []byte) ([]byte, error) {
	var legacy struct {
		Version      uint16                    `json:"version"`
		Architecture string                    `json:"architecture"`
		Policy       model.ArchitectureProfile `json:"policy"`
	}
	if err := strictjson.DecodeBytes(data, &legacy); err != nil {
		return nil, err
	}
	if legacy.Version != 1 || legacy.Architecture == "" || legacy.Policy.Name != legacy.Architecture {
		return nil, errors.New("invalid legacy model profile")
	}
	document, err := modelrecipe.NewProfileDocument(legacy.Policy)
	if err != nil {
		return nil, err
	}
	return document.Content()
}

// renamedProfileExcludedExperts is the expert count the pre-v3 boolean held
// implicitly: PostRotaryRMSNon128 was true exactly when post-rotary RMS applied
// to every expert count except this one. v3 declares the count as data, so the
// upgrade is exact in both directions -- true becomes this count, false becomes
// the disabled sentinel.
const renamedProfileExcludedExperts float64 = 128

// upgradeRenamedProfile rewrites a v2 profile whose DenseStages still carries
// PostRotaryRMSNon128. The policy is remapped as generic JSON because the
// current ArchitectureProfile has no field of that name and strict decoding
// would refuse the document before the rename could run.
// catalogPolicyBase renders the registered profile for one architecture as
// generic JSON, supplying defaults for policy fields that postdate a published
// document. An architecture the catalog does not register cannot be upgraded:
// there is no declared authority to complete it from.
func catalogPolicyBase(architecture string) (map[string]any, error) {
	registered, found := model.LookupArchitecture(architecture)
	if !found {
		return nil, fmt.Errorf("renamed model profile: architecture %q is not registered", architecture)
	}
	encoded, err := json.Marshal(registered)
	if err != nil {
		return nil, err
	}
	var base map[string]any
	if err := json.Unmarshal(encoded, &base); err != nil {
		return nil, err
	}
	return base, nil
}

func upgradeRenamedProfile(data []byte) ([]byte, error) {
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}
	stored, ok := body["policy"].(map[string]any)
	if !ok {
		return nil, errors.New("renamed model profile: policy is absent")
	}
	architecture, _ := body["architecture"].(string)
	// Documents published before v3 predate more than the rename: policy fields
	// added to ArchitectureProfile since then are simply absent, and decoding
	// them as zero values fails the current validator. The registered catalog is
	// the declared authority for architecture facts, so it supplies whatever the
	// document never recorded while every value it did record still wins. No
	// model fact is named here; the architecture the document declares selects
	// its own defaults.
	policy, err := catalogPolicyBase(architecture)
	if err != nil {
		return nil, err
	}
	maps.Copy(policy, stored)
	stages, ok := policy["DenseStages"].(map[string]any)
	if !ok {
		return nil, errors.New("renamed model profile: dense stages are absent")
	}
	if legacy, present := stages["PostRotaryRMSNon128"]; present {
		applied, ok := legacy.(bool)
		if !ok {
			return nil, errors.New("renamed model profile: PostRotaryRMSNon128 is not boolean")
		}
		delete(stages, "PostRotaryRMSNon128")
		excluded := float64(tensor.FirstOffset)
		if applied {
			excluded = renamedProfileExcludedExperts
		}
		stages["PostRotaryRMSExcludedExperts"] = excluded
	}
	remapped, err := json.Marshal(policy)
	if err != nil {
		return nil, err
	}
	var profile model.ArchitectureProfile
	if err := strictjson.DecodeBytes(remapped, &profile); err != nil {
		return nil, err
	}
	if architecture == "" || profile.Name != architecture {
		return nil, errors.New("invalid renamed model profile")
	}
	document, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		return nil, err
	}
	return document.Content()
}

func resolveReferences(value any, names map[string]artifact.ID) (any, bool, error) {
	switch typed := value.(type) {
	case map[string]any:
		if len(typed) == 1 {
			if logical, ok := typed["$artifact"].(string); ok {
				id, found := names[logical]
				return id.String(), found, nil
			}
		}
		result := make(map[string]any, len(typed))
		for key, child := range typed {
			resolved, complete, err := resolveReferences(child, names)
			if err != nil || !complete {
				return nil, complete, err
			}
			result[key] = resolved
		}
		return result, true, nil
	case []any:
		result := make([]any, len(typed))
		for index, child := range typed {
			resolved, complete, err := resolveReferences(child, names)
			if err != nil || !complete {
				return nil, complete, err
			}
			result[index] = resolved
		}
		return result, true, nil
	default:
		return value, true, nil
	}
}

func importFile(root string, kind artifact.Kind, relative string) (
	artifact.ID, artifact.Descriptor, artifact.LocationEvent, error,
) {
	resolved, file, err := openImportFile(root, relative)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	defer file.Close()
	id, size, err := artifact.Identify(kind, file)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	location, err := artifact.CanonicalLocalLocation(id, artifact.LocationFile, resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	return id, artifact.Descriptor{ID: id, Size: size}, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd}, nil
}

func importStoredFile(ctx context.Context, root string, objects *objectstore.Store, kind artifact.Kind, record wireRecord) (objectstore.Publication, error) {
	_, file, err := openImportFile(root, record.Path)
	if err != nil {
		return objectstore.Publication{}, err
	}
	defer file.Close()
	return objects.Publish(ctx, objectstore.PublishRequest{
		Kind: kind, MediaType: record.MediaType, Schema: record.Schema, Reader: file,
	})
}

func openImportFile(root, relative string) (string, *os.File, error) {
	if filepath.IsAbs(relative) || filepath.Clean(relative) != relative {
		return "", nil, errors.New("repodb import: unsafe file path")
	}
	path := filepath.Join(root, relative)
	resolved, err := filepath.Abs(path)
	if err != nil {
		return "", nil, err
	}
	resolved, err = filepath.EvalSymlinks(resolved)
	if err != nil {
		return "", nil, err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", nil, errors.New("repodb import: file path escapes root")
	}
	file, err := os.Open(resolved)
	if err != nil {
		return "", nil, err
	}
	return resolved, file, nil
}

func importSource(header Header, raw []byte) (artifact.Content, artifact.ID, error) {
	body := sourceBody{
		Version: Version, SourceCommit: header.SourceCommit,
		ExportSHA256: exportDigest(raw), Counts: header.Counts,
	}
	data, err := json.Marshal(body)
	if err != nil {
		return artifact.Content{}, artifact.ID{}, err
	}
	content, err := sourceContract.ContentBytes(data)
	return content, content.Descriptor.ID, err
}

func exportDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func containsPending(nodes []pendingNode, name string) bool {
	for _, node := range nodes {
		if node.record.Name == name {
			return true
		}
	}
	return false
}

func validCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil && value == strings.ToLower(value)
}

func equalCounts(left, right map[string]uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if value == 0 || right[key] != value {
			return false
		}
	}
	return true
}

func recordCountKey(record wireRecord) string {
	switch record.Type {
	case "artifact", "manifest":
		return "kind:" + record.Kind
	default:
		return record.Type
	}
}

func cloneNames(names map[string]artifact.ID) map[string]artifact.ID {
	cloned := make(map[string]artifact.ID, len(names))
	maps.Copy(cloned, names)
	return cloned
}
