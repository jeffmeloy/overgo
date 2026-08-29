// Package trainingdata materializes immutable datasets into typed training streams.
package trainingdata

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
)

// Authority seals dataset, split, processor, and order facts.
type Authority struct {
	Dataset    artifact.ID
	Split      artifact.ID
	Processors []artifact.ID
	Seed       uint64
	Shuffle    bool
	Signature  recipecontract.ModalitySignature
}

// RawRecord is one immutable source record before processor conversion.
type RawRecord struct {
	ID     string
	Group  string
	Fields []string
	Data   []byte
}

type ValueRole string

const (
	RoleInput    ValueRole = "input"
	RoleTarget   ValueRole = "target"
	RoleChosen   ValueRole = "chosen"
	RoleRejected ValueRole = "rejected"
)

// Value is one processor-produced typed payload.
type Value struct {
	Role       ValueRole
	Modality   recipecontract.Modality
	Encoding   string
	Shape      []int
	SampleRate int
	Data       []byte
	Completion []bool
}

// Example preserves source identity across processing and batching.
type Example struct {
	ID     string
	Group  string
	Values []Value
}

// Processor converts one raw record. Binding supplies its profile identity.
type Processor func(context.Context, RawRecord) (Example, error)

// ProcessorBinding maps dataset assets to one run-bound processor profile.
// A sole binding may omit Assets and becomes the dataset default.
type ProcessorBinding struct {
	Artifact   artifact.ID
	Assets     []string
	Modalities []recipecontract.Modality
	Process    Processor
}

type compiledProcessor struct {
	process    Processor
	modalities map[recipecontract.Modality]bool
}

// Passthrough returns a processor for already encoded single-value records.
func Passthrough(role ValueRole, modality recipecontract.Modality, encoding string) Processor {
	return func(_ context.Context, record RawRecord) (Example, error) {
		if !validRole(role) || !recipecontract.ValidModality(modality) || strings.TrimSpace(encoding) == "" {
			return Example{}, errors.New("training data: invalid passthrough processor")
		}
		return Example{
			ID: record.ID, Group: record.Group,
			Values: []Value{{Role: role, Modality: modality, Encoding: encoding, Data: record.Data}},
		}, nil
	}
}

type recordRef struct {
	id        string
	group     string
	fields    []string
	processor artifact.ID
	file      *indexedFile
	offset    int64
	length    int64
	digest    [sha256.Size]byte
	inline    []byte
}

func (record recordRef) read() ([]byte, error) {
	if record.inline != nil {
		return slices.Clone(record.inline), nil
	}
	if record.file == nil || record.length < 0 {
		return nil, errors.New("training data: invalid record source")
	}
	length, ok := checked.Int(uint64(record.length))
	if !ok {
		return nil, errors.New("training data: record exceeds addressable memory")
	}
	data := make([]byte, length)
	n, err := record.file.file.ReadAt(data, record.offset)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	if n != length {
		return nil, fmt.Errorf("short read: got %d of %d bytes", n, length)
	}
	return data, nil
}

type member struct {
	identity string
	weight   uint64
	records  []recordRef
}

// Dataset owns indexed files and canonical member records.
type Dataset struct {
	authority  Authority
	identity   artifact.ID
	members    []member
	processors map[artifact.ID]compiledProcessor
	files      []*indexedFile
}

func (d *Dataset) Identity() artifact.ID {
	if d == nil {
		return artifact.ID{}
	}
	return d.identity
}

func (d *Dataset) Records() int {
	if d == nil {
		return 0
	}
	total := 0
	for _, member := range d.members {
		total += len(member.records)
	}
	return total
}

func (d *Dataset) Close() error {
	if d == nil {
		return nil
	}
	var first error
	for _, file := range d.files {
		if err := file.file.Close(); err != nil && first == nil {
			first = err
		}
	}
	d.files = nil
	return first
}

// Materialize resolves dataset composition, file locations, memberships, and deduplication.
func Materialize(
	ctx context.Context,
	reader artifact.Reader,
	authority Authority,
	bindings []ProcessorBinding,
) (*Dataset, error) {
	if ctx == nil || reader == nil {
		return nil, errors.New("training data: nil materializer input")
	}
	if authority.Dataset.Kind() != artifact.KindDataset || authority.Split.Kind() != artifact.KindDatasetShard {
		return nil, errors.New("training data: invalid dataset authority")
	}
	processors, assets, fallback, err := compileBindings(authority, bindings)
	if err != nil {
		return nil, err
	}
	state := materializeState{
		reader: reader, processors: processors, assets: assets, fallback: fallback,
		filesByArtifact: map[artifact.ID]*indexedFile{},
	}
	members, err := state.resolve(ctx, authority.Dataset, nil, map[artifact.ID]bool{})
	if err != nil {
		state.close()
		return nil, err
	}
	membership, ok, err := dataset.LoadMembership(ctx, reader, authority.Split)
	if err != nil || !ok {
		state.close()
		if err == nil {
			err = errors.New("split membership absent")
		}
		return nil, fmt.Errorf("training data: load split: %w", err)
	}
	if membership.Source != authority.Dataset {
		state.close()
		return nil, errors.New("training data: split source differs from dataset")
	}
	if err := filterMembership(members, membership); err != nil {
		state.close()
		return nil, err
	}
	deduplicate(members)
	members = nonemptyMembers(members)
	if len(members) == 0 {
		state.close()
		return nil, errors.New("training data: split has no records after deduplication")
	}
	identity, err := identifyAuthority(authority)
	if err != nil {
		state.close()
		return nil, err
	}
	return &Dataset{authority: cloneAuthority(authority), identity: identity, members: members, processors: processors, files: state.files}, nil
}

// MaterializeDocuments routes in-memory documents through the same stream owner.
// Dataset and split identities remain caller-owned authority facts.
func MaterializeDocuments(authority Authority, processor artifact.ID, documents []string, binding ProcessorBinding) (*Dataset, error) {
	records := make([]RawRecord, len(documents))
	for index, document := range documents {
		if document == "" {
			return nil, errors.New("training data: empty document")
		}
		id := fmt.Sprintf("%s/document/%d", authority.Dataset, index)
		records[index] = RawRecord{ID: id, Group: id, Data: []byte(document)}
	}
	return MaterializeRecords(authority, processor, records, binding)
}

// MaterializeRecords routes in-memory binary records through the shared stream.
func MaterializeRecords(authority Authority, processor artifact.ID, records []RawRecord, binding ProcessorBinding) (*Dataset, error) {
	if authority.Dataset.Kind() != artifact.KindDataset || authority.Split.Kind() != artifact.KindDatasetShard ||
		processor.Kind() != artifact.KindProfile || binding.Artifact != processor || len(records) == 0 {
		return nil, errors.New("training data: invalid record materialization")
	}
	processors, _, _, err := compileBindings(authority, []ProcessorBinding{binding})
	if err != nil {
		return nil, err
	}
	indexed := make([]recordRef, len(records))
	for index, record := range records {
		if strings.TrimSpace(record.ID) == "" || len(record.Data) == 0 {
			return nil, errors.New("training data: record identity and data required")
		}
		group := record.Group
		if group == "" {
			group = record.ID
		}
		data := slices.Clone(record.Data)
		indexed[index] = recordRef{
			id: record.ID, group: group, fields: slices.Clone(record.Fields), processor: processor,
			length: int64(len(data)), digest: sha256.Sum256(data), inline: data,
		}
	}
	members := []member{{identity: authority.Dataset.String(), weight: 1, records: indexed}}
	deduplicate(members)
	identity, err := identifyAuthority(authority)
	if err != nil {
		return nil, err
	}
	return &Dataset{
		authority: cloneAuthority(authority), identity: identity, processors: processors,
		members: members,
	}, nil
}

type materializeState struct {
	reader          artifact.Reader
	processors      map[artifact.ID]compiledProcessor
	assets          map[string]artifact.ID
	fallback        artifact.ID
	files           []*indexedFile
	filesByArtifact map[artifact.ID]*indexedFile
}

func (state *materializeState) close() {
	for _, file := range state.files {
		_ = file.file.Close()
	}
	state.files = nil
}

func (state *materializeState) resolve(
	ctx context.Context,
	id artifact.ID,
	fields []string,
	stack map[artifact.ID]bool,
) ([]member, error) {
	if stack[id] {
		return nil, errors.New("training data: cyclic dataset composition")
	}
	stack[id] = true
	defer delete(stack, id)
	document, ok, err := dataset.Load(ctx, state.reader, id)
	if err != nil || !ok {
		if err == nil {
			err = errors.New("dataset document absent")
		}
		return nil, err
	}
	switch document.Type {
	case dataset.TypeVersion:
		records := make([]recordRef, 0)
		for _, asset := range document.Assets {
			processor, err := state.processorFor(asset.Name)
			if err != nil {
				return nil, err
			}
			indexed, err := state.openAsset(ctx, id, asset, processor, fields)
			if err != nil {
				return nil, err
			}
			records = append(records, indexed...)
		}
		return []member{{identity: id.String(), weight: 1, records: records}}, nil
	case dataset.TypeView:
		selected := fields
		if len(document.Fields) != 0 {
			selected = document.Fields
		}
		members, err := state.resolve(ctx, *document.Source, selected, stack)
		if err != nil {
			return nil, err
		}
		if document.Selector != nil {
			membership, ok, err := dataset.LoadMembership(ctx, state.reader, *document.Selector)
			if err != nil || !ok {
				if err == nil {
					err = errors.New("view selector absent")
				}
				return nil, err
			}
			if membership.Source != *document.Source {
				return nil, errors.New("training data: view selector source differs")
			}
			if err := filterMembership(members, membership); err != nil {
				return nil, err
			}
		}
		return members, nil
	case dataset.TypeMixture:
		var members []member
		for _, source := range document.Members {
			resolved, err := state.resolve(ctx, source.Dataset, fields, stack)
			if err != nil {
				return nil, err
			}
			for index := range resolved {
				weight, valid := checked.Mul64(resolved[index].weight, source.Weight)
				if !valid {
					return nil, errors.New("training data: mixture weight overflows")
				}
				resolved[index].weight = weight
				resolved[index].identity = source.Dataset.String() + "/" + resolved[index].identity
			}
			members = append(members, resolved...)
		}
		return members, nil
	case dataset.TypeSplit:
		return nil, errors.New("training data: materialize a split view, not the split document")
	default:
		return nil, errors.New("training data: unsupported dataset document")
	}
}

func (state *materializeState) processorFor(asset string) (artifact.ID, error) {
	if id, ok := state.assets[asset]; ok {
		return id, nil
	}
	if state.fallback.Valid() {
		return state.fallback, nil
	}
	return artifact.ID{}, fmt.Errorf("training data: asset %q has no processor binding", asset)
}

func (state *materializeState) openAsset(
	ctx context.Context,
	datasetID artifact.ID,
	asset dataset.Asset,
	processor artifact.ID,
	fields []string,
) ([]recordRef, error) {
	if asset.Records == 0 {
		return nil, fmt.Errorf("training data: asset %q has zero records", asset.Name)
	}
	indexed := state.filesByArtifact[asset.Artifact]
	if indexed == nil {
		descriptor, ok, err := state.reader.Artifact(ctx, asset.Artifact)
		if err != nil || !ok {
			if err == nil {
				err = errors.New("artifact descriptor absent")
			}
			return nil, fmt.Errorf("training data: asset %q: %w", asset.Name, err)
		}
		locations, err := state.reader.Locations(ctx, asset.Artifact)
		if err != nil {
			return nil, err
		}
		path := fileLocation(locations)
		if path == "" {
			return nil, fmt.Errorf("training data: asset %q has no file location", asset.Name)
		}
		indexed, err = openIndexedFile(path, descriptor.Size, asset.Records)
		if err != nil {
			return nil, fmt.Errorf("training data: index %q: %w", asset.Name, err)
		}
		state.files = append(state.files, indexed)
		state.filesByArtifact[asset.Artifact] = indexed
	} else if uint64(len(indexed.spans)) != asset.Records {
		return nil, fmt.Errorf("training data: asset %q record declaration differs across uses", asset.Name)
	}
	records := make([]recordRef, len(indexed.spans))
	for index, span := range indexed.spans {
		id := fmt.Sprintf("%s/%s/%d", datasetID, asset.Name, index)
		records[index] = recordRef{
			id: id, group: id, fields: slices.Clone(fields), processor: processor,
			file: indexed, offset: span.offset, length: span.length, digest: span.digest,
		}
	}
	return records, nil
}

type indexedFile struct {
	file  *os.File
	spans []recordSpan
}

type recordSpan struct {
	offset int64
	length int64
	digest [sha256.Size]byte
}

func openIndexedFile(path string, expectedBytes, records uint64) (*indexedFile, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(absolute)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*indexedFile, error) {
		_ = file.Close()
		return nil, err
	}
	info, err := file.Stat()
	if err != nil {
		return fail(err)
	}
	if !info.Mode().IsRegular() || uint64(info.Size()) != expectedBytes {
		return fail(errors.New("file size differs from artifact descriptor"))
	}
	indexed := &indexedFile{file: file}
	if records == 1 {
		if info.Size() == 0 {
			return fail(errors.New("empty record file"))
		}
		digest, err := hashSection(file, 0, info.Size())
		if err != nil {
			return fail(err)
		}
		indexed.spans = []recordSpan{{length: info.Size(), digest: digest}}
		return indexed, nil
	}
	reader := bufio.NewReader(file)
	var offset int64
	for {
		line, readErr := reader.ReadBytes('\n')
		if len(line) != 0 {
			length := len(line)
			if line[length-1] == '\n' {
				length--
			}
			if length > 0 && line[length-1] == '\r' {
				length--
			}
			if length == 0 {
				return fail(errors.New("empty line record"))
			}
			indexed.spans = append(indexed.spans, recordSpan{
				offset: offset, length: int64(length), digest: sha256.Sum256(line[:length]),
			})
			offset += int64(len(line))
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return fail(readErr)
		}
	}
	if uint64(len(indexed.spans)) != records {
		return fail(fmt.Errorf("record count %d differs from declared %d", len(indexed.spans), records))
	}
	return indexed, nil
}

func hashSection(file *os.File, offset, length int64) ([sha256.Size]byte, error) {
	hasher := sha256.New()
	if _, err := io.CopyN(hasher, io.NewSectionReader(file, offset, length), length); err != nil {
		return [sha256.Size]byte{}, err
	}
	var digest [sha256.Size]byte
	copy(digest[:], hasher.Sum(nil))
	return digest, nil
}

func fileLocation(locations []artifact.Location) string {
	paths := make([]string, 0, len(locations))
	for _, location := range locations {
		if location.Kind == artifact.LocationFile {
			paths = append(paths, location.Value)
		}
	}
	slices.Sort(paths)
	if len(paths) == 0 {
		return ""
	}
	return paths[0]
}

func compileBindings(authority Authority, bindings []ProcessorBinding) (
	map[artifact.ID]compiledProcessor,
	map[string]artifact.ID,
	artifact.ID,
	error,
) {
	if len(authority.Processors) == 0 || len(bindings) != len(authority.Processors) || authority.Signature.Validate() != nil {
		return nil, nil, artifact.ID{}, errors.New("training data: processor bindings differ from authority")
	}
	want := make(map[artifact.ID]bool, len(authority.Processors))
	for _, id := range authority.Processors {
		if id.Kind() != artifact.KindProfile || want[id] {
			return nil, nil, artifact.ID{}, errors.New("training data: invalid processor authority")
		}
		want[id] = true
	}
	processors := make(map[artifact.ID]compiledProcessor, len(bindings))
	covered := map[recipecontract.Modality]bool{}
	assets := map[string]artifact.ID{}
	var fallback artifact.ID
	for _, binding := range bindings {
		if !want[binding.Artifact] || binding.Process == nil {
			return nil, nil, artifact.ID{}, errors.New("training data: invalid processor binding")
		}
		if _, duplicate := processors[binding.Artifact]; duplicate || len(binding.Modalities) == 0 {
			return nil, nil, artifact.ID{}, errors.New("training data: invalid processor binding")
		}
		modalities := make(map[recipecontract.Modality]bool, len(binding.Modalities))
		for _, modality := range binding.Modalities {
			if !recipecontract.ValidModality(modality) || modalities[modality] {
				return nil, nil, artifact.ID{}, errors.New("training data: invalid processor modality binding")
			}
			modalities[modality] = true
			covered[modality] = true
		}
		processors[binding.Artifact] = compiledProcessor{process: binding.Process, modalities: modalities}
		if len(binding.Assets) == 0 {
			if len(bindings) != 1 {
				return nil, nil, artifact.ID{}, errors.New("training data: multiple processors require asset bindings")
			}
			fallback = binding.Artifact
		}
		for _, asset := range binding.Assets {
			if strings.TrimSpace(asset) == "" || assets[asset].Valid() {
				return nil, nil, artifact.ID{}, errors.New("training data: invalid processor asset binding")
			}
			assets[asset] = binding.Artifact
		}
	}
	for _, modalities := range [][]recipecontract.Modality{authority.Signature.Inputs, authority.Signature.Outputs} {
		for _, modality := range modalities {
			if !covered[modality] {
				return nil, nil, artifact.ID{}, fmt.Errorf("training data: modality %q has no processor", modality)
			}
		}
	}
	return processors, assets, fallback, nil
}

func filterMembership(members []member, membership dataset.Membership) error {
	selected := make(map[string]string, len(membership.Records))
	for _, record := range membership.Records {
		selected[record.ID] = record.Group
	}
	found := map[string]bool{}
	for memberIndex := range members {
		kept := members[memberIndex].records[:0]
		for _, record := range members[memberIndex].records {
			group, ok := selected[record.id]
			if !ok {
				continue
			}
			record.group = group
			kept = append(kept, record)
			found[record.id] = true
		}
		members[memberIndex].records = kept
	}
	if len(found) != len(selected) {
		return fmt.Errorf("training data: membership resolved %d of %d records", len(found), len(selected))
	}
	return nil
}

func deduplicate(members []member) {
	type key struct {
		processor artifact.ID
		digest    [sha256.Size]byte
		fields    string
	}
	seen := map[key]bool{}
	for memberIndex := range members {
		kept := members[memberIndex].records[:0]
		for _, record := range members[memberIndex].records {
			identity := key{processor: record.processor, digest: record.digest, fields: strings.Join(record.fields, "\x00")}
			if seen[identity] {
				continue
			}
			seen[identity] = true
			kept = append(kept, record)
		}
		members[memberIndex].records = kept
	}
}

func nonemptyMembers(source []member) []member {
	result := source[:0]
	for _, member := range source {
		if len(member.records) != 0 && member.weight != 0 {
			result = append(result, member)
		}
	}
	return result
}

func identifyAuthority(authority Authority) (artifact.ID, error) {
	payload, err := json.Marshal(struct {
		Dataset    artifact.ID                      `json:"dataset"`
		Split      artifact.ID                      `json:"split"`
		Processors []artifact.ID                    `json:"processors"`
		Seed       uint64                           `json:"seed"`
		Shuffle    bool                             `json:"shuffle"`
		Signature  recipecontract.ModalitySignature `json:"signature"`
	}{authority.Dataset, authority.Split, slices.Clone(authority.Processors), authority.Seed, authority.Shuffle, authority.Signature.Clone()})
	if err != nil {
		return artifact.ID{}, err
	}
	return artifact.IdentifyBytes(artifact.KindProfile, payload)
}

func cloneAuthority(authority Authority) Authority {
	authority.Processors = slices.Clone(authority.Processors)
	authority.Signature = authority.Signature.Clone()
	return authority
}

func validRole(role ValueRole) bool {
	return role == RoleInput || role == RoleTarget || role == RoleChosen || role == RoleRejected
}
