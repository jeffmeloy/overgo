package repodbimport

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
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/dataset"
	"overgo/internal/finding"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

const (
	Version         uint16 = 1
	SourceMediaType        = "application/vnd.overgo.repodb-import+json"
	SourceSchema           = "overgo/repodb-import/v1"
	maxLineBytes           = artifact.MaxContentBytes
	maxImportBytes         = 512 << 20
	maxRecords             = 1_000_000
)

var sourceContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: SourceMediaType, Schema: SourceSchema,
}

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
	Name       string            `json:"name,omitempty"`
	Kind       string            `json:"kind,omitempty"`
	Path       string            `json:"path,omitempty"`
	Document   json.RawMessage   `json:"document,omitempty"`
	MediaType  string            `json:"media_type,omitempty"`
	Schema     string            `json:"schema,omitempty"`
	Components []componentRecord `json:"components,omitempty"`
	Child      string            `json:"child,omitempty"`
	Parent     string            `json:"parent,omitempty"`
	Relation   string            `json:"relation,omitempty"`
	Target     string            `json:"target,omitempty"`
	Previous   string            `json:"previous,omitempty"`
}

type sourceBody struct {
	Version      uint16            `json:"version"`
	SourceCommit string            `json:"source_commit"`
	ExportSHA256 string            `json:"export_sha256"`
	Counts       map[string]uint64 `json:"counts"`
}

type Result struct {
	Commit artifact.CommitID
	Source artifact.ID
	Names  map[string]artifact.ID
}

type pendingNode struct {
	record wireRecord
	kind   artifact.Kind
}

func Import(
	ctx context.Context,
	repository artifact.Repository,
	root string,
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
			if err != nil || !validLogicalName(record.Name) {
				return Result{}, errors.New("repodb import: invalid artifact record")
			}
			if _, duplicate := names[record.Name]; duplicate || containsPending(pending, record.Name) {
				return Result{}, errors.New("repodb import: duplicate logical name")
			}
			if record.Path != "" {
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
			if err != nil || !validLogicalName(record.Name) || len(record.Components) == 0 ||
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
	data, err = canonicalizeKnownDocument(node.record.MediaType, node.record.Schema, data)
	if err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	id, err := artifact.IdentifyBytes(node.kind, data)
	if err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	content := artifact.Content{Descriptor: artifact.Descriptor{
		ID: id, Size: uint64(len(data)), MediaType: node.record.MediaType, Schema: node.record.Schema,
	}, Data: data}
	if err := content.Validate(); err != nil {
		return false, artifact.ID{}, nil, nil, err
	}
	return true, id, &content, nil, nil
}

// canonicalizeKnownDocument returns the domain codec's canonical bytes for a
// known media type (admitting any field order), and the input unchanged for
// unknown types. Each codec owns its canonical-form fact.
func canonicalizeKnownDocument(mediaType, schema string, data []byte) ([]byte, error) {
	canonical := data
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
		requireSchema(modelrecipe.ProfileSchema, modelrecipe.LegacyProfileSchema)
		if err != nil {
			break
		}
		_, canonical, err = modelrecipe.NormalizeProfileDocument(data)
	case modelrecipe.ProfileParityMediaType:
		requireSchema(modelrecipe.ProfileParitySchema)
		if err != nil {
			break
		}
		_, canonical, err = modelrecipe.NormalizeProfileParityEvidence(data)
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
		return nil, fmt.Errorf("repodb import: invalid %q document: %w", mediaType, err)
	}
	return canonical, nil
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
	if filepath.IsAbs(relative) || filepath.Clean(relative) != relative {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, errors.New("repodb import: unsafe file path")
	}
	path := filepath.Join(root, relative)
	resolved, err := filepath.Abs(path)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	resolved, err = filepath.EvalSymlinks(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, errors.New("repodb import: file path escapes root")
	}
	file, err := os.Open(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	defer file.Close()
	id, size, err := artifact.Identify(kind, file)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	return id, artifact.Descriptor{ID: id, Size: size}, artifact.LocationEvent{
		Location: artifact.Location{Artifact: id, Kind: artifact.LocationFile, Value: resolved},
		Action:   artifact.LocationAdd,
	}, nil
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
	id, err := sourceContract.Identify(data)
	if err != nil {
		return artifact.Content{}, artifact.ID{}, err
	}
	content, err := sourceContract.Content(id, data)
	return content, id, err
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

func validLogicalName(value string) bool {
	return value != "" && len(value) <= 512 && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r\n")
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
	for name, id := range names {
		cloned[name] = id
	}
	return cloned
}
