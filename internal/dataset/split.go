package dataset

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	MembershipVersion   uint16 = 1
	MembershipMediaType        = "application/vnd.overgo.dataset-membership+json"
	MembershipSchema           = "overgo/dataset-membership/v1"
)

var membershipContract = artifact.DocumentContract{
	Kind: artifact.KindDatasetShard, MediaType: MembershipMediaType, Schema: MembershipSchema,
}

type Record struct {
	ID    string `json:"id"`
	Group string `json:"group"`
}

type SplitPartition struct {
	Name   string `json:"name"`
	Weight uint64 `json:"weight"`
}

// Membership: immutable group-safe partition assignment.
type Membership struct {
	Version   uint16      `json:"version"`
	Source    artifact.ID `json:"source"`
	Seed      uint64      `json:"seed"`
	Partition string      `json:"partition"`
	Records   []Record    `json:"records,omitempty"`
	ID        artifact.ID `json:"-"`
}

type SplitPlan struct {
	Memberships []Membership
	Views       []Document
	Split       Document
}

func BuildGroupSplit(
	source artifact.ID,
	records []Record,
	seed uint64,
	partitions []SplitPartition,
) (SplitPlan, error) {
	if source.Kind() != artifact.KindDataset {
		return SplitPlan{}, errors.New("dataset: split source is not a dataset")
	}
	canonicalRecords := slices.Clone(records)
	if err := canonicalizeRecords(&canonicalRecords, false); err != nil {
		return SplitPlan{}, err
	}
	canonicalPartitions := slices.Clone(partitions)
	if err := canonicalizeSplitPartitions(&canonicalPartitions); err != nil {
		return SplitPlan{}, err
	}
	assigned := make(map[string]int)
	partitionRecords := make([][]Record, len(canonicalPartitions))
	for _, record := range canonicalRecords {
		index, ok := assigned[record.Group]
		if !ok {
			index = selectPartition(seed, record.Group, canonicalPartitions)
			assigned[record.Group] = index
		}
		partitionRecords[index] = append(partitionRecords[index], record)
	}
	plan := SplitPlan{
		Memberships: make([]Membership, len(canonicalPartitions)),
		Views:       make([]Document, len(canonicalPartitions)),
	}
	partitionViews := make([]Partition, len(canonicalPartitions))
	for index, partition := range canonicalPartitions {
		membership, err := newMembership(source, seed, partition.Name, partitionRecords[index])
		if err != nil {
			return SplitPlan{}, err
		}
		view, err := NewView(source, &membership.ID, nil)
		if err != nil {
			return SplitPlan{}, err
		}
		plan.Memberships[index] = membership
		plan.Views[index] = view
		partitionViews[index] = Partition{Name: partition.Name, View: view.ID}
	}
	var err error
	plan.Split, err = NewSplit(source, partitionViews)
	return plan, err
}

func ParseMembership(content []byte) (Membership, error) {
	var body Membership
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return Membership{}, fmt.Errorf("dataset: decode membership: %w", err)
	}
	membership, err := newMembership(body.Source, body.Seed, body.Partition, body.Records)
	if err != nil {
		return Membership{}, err
	}
	canonical, err := membership.ContentBytes()
	if err != nil {
		return Membership{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Membership{}, errors.New("dataset: non-canonical membership")
	}
	return membership, nil
}

func (m Membership) ValidateIdentity() error {
	if m.ID.Kind() != artifact.KindDatasetShard {
		return errors.New("dataset: invalid membership identity")
	}
	canonical := m
	canonical.ID = artifact.ID{}
	canonical.Records = slices.Clone(m.Records)
	if err := canonicalizeMembership(&canonical); err != nil {
		return err
	}
	canonical.ID = m.ID
	if m.Version != canonical.Version || m.Source != canonical.Source || m.Seed != canonical.Seed ||
		m.Partition != canonical.Partition || !slices.Equal(m.Records, canonical.Records) {
		return errors.New("dataset: membership is not canonical")
	}
	canonical.ID = artifact.ID{}
	content, err := membershipContent(canonical)
	if err != nil {
		return err
	}
	if err := membershipContract.ValidateIdentity(m.ID, content); err != nil {
		return errors.New("dataset: membership identity mismatch")
	}
	return nil
}

func (m Membership) ContentBytes() ([]byte, error) {
	if err := m.ValidateIdentity(); err != nil {
		return nil, err
	}
	m.ID = artifact.ID{}
	return membershipContent(m)
}

func (m Membership) Content() (artifact.Content, error) {
	id := m.ID
	content, err := m.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return membershipContract.Content(id, content)
}

func (m Membership) Lineage() artifact.Lineage {
	return artifact.Lineage{Child: m.ID, Parent: m.Source, Relation: artifact.RelationDerivedFrom}
}

func (p SplitPlan) PublicationBatch(key string, aliases []artifact.AliasBinding) (artifact.Batch, error) {
	if err := p.validate(); err != nil {
		return artifact.Batch{}, err
	}
	documents := append(slices.Clone(p.Views), p.Split)
	batch, err := PublicationBatch(key, documents, aliases)
	if err != nil {
		return artifact.Batch{}, err
	}
	for _, membership := range p.Memberships {
		content, err := membership.Content()
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, membership.Lineage())
	}
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}

func (p SplitPlan) validate() error {
	if len(p.Memberships) != len(p.Views) || len(p.Views) < 2 ||
		len(p.Split.Partitions) != len(p.Views) || p.Split.Type != TypeSplit || p.Split.Source == nil {
		return errors.New("dataset: invalid split plan")
	}
	if err := p.Split.ValidateIdentity(); err != nil {
		return err
	}
	source := *p.Split.Source
	membershipByID := make(map[artifact.ID]Membership, len(p.Memberships))
	for _, membership := range p.Memberships {
		if err := membership.ValidateIdentity(); err != nil {
			return err
		}
		if membership.Source != source {
			return errors.New("dataset: split membership source differs")
		}
		if _, duplicate := membershipByID[membership.ID]; duplicate {
			return errors.New("dataset: duplicate split membership")
		}
		membershipByID[membership.ID] = membership
	}
	viewByID := make(map[artifact.ID]Document, len(p.Views))
	for _, view := range p.Views {
		if err := view.ValidateIdentity(); err != nil {
			return err
		}
		if view.Type != TypeView || view.Source == nil || *view.Source != source || view.Selector == nil {
			return errors.New("dataset: split view differs from membership source")
		}
		if _, ok := membershipByID[*view.Selector]; !ok {
			return errors.New("dataset: split view selector is not a membership")
		}
		if _, duplicate := viewByID[view.ID]; duplicate {
			return errors.New("dataset: duplicate split view")
		}
		viewByID[view.ID] = view
	}
	usedMemberships := make(map[artifact.ID]struct{}, len(p.Memberships))
	for _, partition := range p.Split.Partitions {
		view, ok := viewByID[partition.View]
		if !ok {
			return errors.New("dataset: split partition view is absent")
		}
		membership := membershipByID[*view.Selector]
		if membership.Partition != partition.Name {
			return errors.New("dataset: split partition name differs from membership")
		}
		usedMemberships[membership.ID] = struct{}{}
	}
	if len(usedMemberships) != len(p.Memberships) {
		return errors.New("dataset: split plan has unused membership")
	}
	return nil
}

func DuplicateLineage(duplicate, canonical artifact.ID) (artifact.Lineage, error) {
	edge := artifact.Lineage{Child: duplicate, Parent: canonical, Relation: artifact.RelationDuplicateOf}
	if duplicate.Kind() != canonical.Kind() {
		return artifact.Lineage{}, errors.New("dataset: duplicate kinds differ")
	}
	if err := edge.Validate(); err != nil {
		return artifact.Lineage{}, err
	}
	return edge, nil
}

func newMembership(source artifact.ID, seed uint64, partition string, records []Record) (Membership, error) {
	membership := Membership{
		Version: MembershipVersion, Source: source, Seed: seed,
		Partition: partition, Records: slices.Clone(records),
	}
	if err := canonicalizeMembership(&membership); err != nil {
		return Membership{}, err
	}
	content, err := membershipContent(membership)
	if err != nil {
		return Membership{}, err
	}
	membership.ID, err = membershipContract.Identify(content)
	return membership, err
}

func canonicalizeMembership(membership *Membership) error {
	if membership == nil || membership.Version != MembershipVersion ||
		membership.Source.Kind() != artifact.KindDataset || !validName(membership.Partition) {
		return errors.New("dataset: invalid membership")
	}
	return canonicalizeRecords(&membership.Records, true)
}

func canonicalizeRecords(records *[]Record, allowEmpty bool) error {
	if !allowEmpty && len(*records) == 0 || len(*records) > maxEntries {
		return errors.New("dataset: invalid split record count")
	}
	sort.Slice(*records, func(i, j int) bool { return (*records)[i].ID < (*records)[j].ID })
	for index, record := range *records {
		if !validName(record.ID) || !validName(record.Group) || index > 0 && (*records)[index-1].ID == record.ID {
			return errors.New("dataset: invalid split record")
		}
	}
	return nil
}

func canonicalizeSplitPartitions(partitions *[]SplitPartition) error {
	if len(*partitions) < 2 || len(*partitions) > maxEntries {
		return errors.New("dataset: invalid split partition count")
	}
	sort.Slice(*partitions, func(i, j int) bool { return (*partitions)[i].Name < (*partitions)[j].Name })
	var total uint64
	for index, partition := range *partitions {
		if !validName(partition.Name) || partition.Weight == 0 ||
			index > 0 && (*partitions)[index-1].Name == partition.Name || ^uint64(0)-total < partition.Weight {
			return errors.New("dataset: invalid split partition")
		}
		total += partition.Weight
	}
	return nil
}

func selectPartition(seed uint64, group string, partitions []SplitPartition) int {
	hasher := sha256.New()
	var seedBytes [8]byte
	binary.BigEndian.PutUint64(seedBytes[:], seed)
	hasher.Write(seedBytes[:])
	hasher.Write([]byte{0})
	hasher.Write([]byte(group))
	value := binary.BigEndian.Uint64(hasher.Sum(nil)[:8])
	var total uint64
	for _, partition := range partitions {
		total += partition.Weight
	}
	target := value % total
	for index, partition := range partitions {
		if target < partition.Weight {
			return index
		}
		target -= partition.Weight
	}
	return len(partitions) - 1
}

func membershipContent(membership Membership) ([]byte, error) {
	membership.ID = artifact.ID{}
	content, err := json.Marshal(membership)
	if err != nil {
		return nil, fmt.Errorf("dataset: encode membership: %w", err)
	}
	return content, nil
}
