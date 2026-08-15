package trainingdata

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
)

// StreamState is the exact resumable sampler boundary.
type StreamState struct {
	Identity artifact.ID `json:"identity"`
	Position uint64      `json:"position"`
}

// Stream owns deterministic weighted order. Records cycle indefinitely.
type Stream struct {
	dataset     *Dataset
	position    uint64
	totalWeight uint64
	prefixes    []uint64
	orders      []memberOrder
}

type memberOrder struct {
	epoch uint64
	valid bool
	order []int
}

func NewStream(dataset *Dataset, resume *StreamState) (*Stream, error) {
	if dataset == nil || !dataset.identity.Valid() || len(dataset.members) == 0 {
		return nil, errors.New("training data: invalid stream dataset")
	}
	stream := &Stream{dataset: dataset, prefixes: make([]uint64, len(dataset.members)), orders: make([]memberOrder, len(dataset.members))}
	for index, member := range dataset.members {
		if len(member.records) == 0 || member.weight == 0 || math.MaxUint64-stream.totalWeight < member.weight {
			return nil, errors.New("training data: invalid stream member")
		}
		stream.prefixes[index] = stream.totalWeight
		stream.totalWeight += member.weight
	}
	if resume != nil {
		if resume.Identity != dataset.identity {
			return nil, errors.New("training data: resume authority differs")
		}
		stream.position = resume.Position
	}
	return stream, nil
}

func (stream *Stream) Snapshot() StreamState {
	if stream == nil || stream.dataset == nil {
		return StreamState{}
	}
	return StreamState{Identity: stream.dataset.identity, Position: stream.position}
}

func (stream *Stream) peek() (*recordRef, error) {
	if stream == nil || stream.dataset == nil || stream.totalWeight == 0 {
		return nil, errors.New("training data: invalid stream")
	}
	memberIndex := stream.memberAt(stream.position)
	member := &stream.dataset.members[memberIndex]
	ordinal := stream.memberOrdinal(stream.position, memberIndex)
	epoch := ordinal / uint64(len(member.records))
	local := ordinal % uint64(len(member.records))
	order := stream.order(memberIndex, epoch)
	return &member.records[order[local]], nil
}

func (stream *Stream) advance() error {
	if stream.position == math.MaxUint64 {
		return errors.New("training data: stream position exhausted")
	}
	stream.position++
	return nil
}

func (stream *Stream) memberAt(position uint64) int {
	slot := position % stream.totalWeight
	return sort.Search(len(stream.prefixes), func(index int) bool {
		return stream.prefixes[index]+stream.dataset.members[index].weight > slot
	})
}

func (stream *Stream) memberOrdinal(position uint64, memberIndex int) uint64 {
	prefix := stream.prefixes[memberIndex]
	weight := stream.dataset.members[memberIndex].weight
	cycles := position / stream.totalWeight
	remainder := position % stream.totalWeight
	ordinal := cycles * weight
	if remainder > prefix {
		ordinal += min(remainder-prefix, weight)
	}
	return ordinal
}

func (stream *Stream) order(memberIndex int, epoch uint64) []int {
	cached := &stream.orders[memberIndex]
	if cached.valid && cached.epoch == epoch {
		return cached.order
	}
	member := &stream.dataset.members[memberIndex]
	if !stream.dataset.authority.Shuffle {
		if len(cached.order) != len(member.records) {
			cached.order = make([]int, len(member.records))
			for index := range cached.order {
				cached.order[index] = index
			}
		}
		cached.epoch = epoch
		cached.valid = true
		return cached.order
	}
	type ranked struct {
		index  int
		digest [sha256.Size]byte
	}
	ranks := make([]ranked, len(member.records))
	for index, record := range member.records {
		hasher := sha256.New()
		var facts [16]byte
		binary.LittleEndian.PutUint64(facts[:8], stream.dataset.authority.Seed)
		binary.LittleEndian.PutUint64(facts[8:], epoch)
		hasher.Write(facts[:])
		hasher.Write([]byte(member.identity))
		hasher.Write([]byte{0})
		hasher.Write([]byte(record.id))
		copy(ranks[index].digest[:], hasher.Sum(nil))
		ranks[index].index = index
	}
	sort.Slice(ranks, func(left, right int) bool {
		return string(ranks[left].digest[:]) < string(ranks[right].digest[:])
	})
	cached.epoch = epoch
	cached.valid = true
	cached.order = make([]int, len(ranks))
	for index, rank := range ranks {
		cached.order[index] = rank.index
	}
	return cached.order
}

// BatchPolicy derives one packed batch and its bounded decode concurrency.
type BatchPolicy struct {
	Examples           int
	MicrobatchExamples int
	DecodeWorkers      int
	MaxBytes           uint64
}

// Batch preserves stream order after parallel processor execution.
type Batch struct {
	Examples []Example
	State    StreamState
	micro    int
}

func (batch Batch) Microbatches() [][]Example {
	if len(batch.Examples) == 0 {
		return nil
	}
	size := batch.micro
	if size <= 0 || size >= len(batch.Examples) {
		return [][]Example{slices.Clone(batch.Examples)}
	}
	result := make([][]Example, 0, (len(batch.Examples)+size-1)/size)
	for start := 0; start < len(batch.Examples); start += size {
		end := min(start+size, len(batch.Examples))
		result = append(result, slices.Clone(batch.Examples[start:end]))
	}
	return result
}

type Batcher struct {
	stream *Stream
	policy BatchPolicy
}

func NewBatcher(stream *Stream, policy BatchPolicy) (*Batcher, error) {
	if stream == nil || policy.Examples <= 0 || policy.MicrobatchExamples < 0 ||
		policy.MicrobatchExamples > policy.Examples || policy.DecodeWorkers <= 0 {
		return nil, errors.New("training data: invalid batch policy")
	}
	return &Batcher{stream: stream, policy: policy}, nil
}

func (batcher *Batcher) Next(ctx context.Context) (Batch, error) {
	if ctx == nil {
		return Batch{}, errors.New("training data: nil batch context")
	}
	start := batcher.stream.position
	fail := func(err error) (Batch, error) {
		batcher.stream.position = start
		return Batch{}, err
	}
	references := make([]recordRef, 0, batcher.policy.Examples)
	var bytes uint64
	for len(references) < batcher.policy.Examples {
		reference, err := batcher.stream.peek()
		if err != nil {
			return fail(err)
		}
		size := uint64(reference.length)
		if batcher.policy.MaxBytes > 0 && len(references) > 0 &&
			(bytes >= batcher.policy.MaxBytes || size > batcher.policy.MaxBytes-bytes) {
			break
		}
		references = append(references, *reference)
		bytes += size
		if err := batcher.stream.advance(); err != nil {
			return fail(err)
		}
	}
	examples, err := batcher.decode(ctx, references)
	if err != nil {
		return fail(err)
	}
	return Batch{Examples: examples, State: batcher.stream.Snapshot(), micro: batcher.policy.MicrobatchExamples}, nil
}

func (batcher *Batcher) decode(ctx context.Context, references []recordRef) ([]Example, error) {
	examples := make([]Example, len(references))
	jobs := make(chan int)
	workers := min(batcher.policy.DecodeWorkers, len(references))
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var first error
	var errorOnce sync.Once
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				if ctx.Err() != nil {
					continue
				}
				reference := references[index]
				data, err := reference.read()
				if err == nil {
					processor := batcher.stream.dataset.processors[reference.processor]
					examples[index], err = processor.process(ctx, RawRecord{
						ID: reference.id, Group: reference.group, Fields: slices.Clone(reference.fields), Data: data,
					})
				}
				if err != nil {
					errorOnce.Do(func() {
						first = fmt.Errorf("training data: process %q: %w", reference.id, err)
						cancel()
					})
				}
			}
		}()
	}
send:
	for index := range references {
		select {
		case jobs <- index:
		case <-ctx.Done():
			break send
		}
	}
	close(jobs)
	wg.Wait()
	if first != nil {
		return nil, first
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for index, example := range examples {
		if err := validateExample(example, references[index], batcher.stream.dataset.processors[references[index].processor]); err != nil {
			return nil, err
		}
	}
	return examples, nil
}

func validateExample(example Example, reference recordRef, processor compiledProcessor) error {
	if example.ID != reference.id || example.Group != reference.group || len(example.Values) == 0 {
		return errors.New("training data: processor changed record identity or emitted no values")
	}
	for _, value := range example.Values {
		if !validRole(value.Role) || !recipecontract.ValidModality(value.Modality) || !processor.modalities[value.Modality] || value.Encoding == "" || len(value.Data) == 0 {
			return errors.New("training data: processor emitted invalid value")
		}
		for _, extent := range value.Shape {
			if extent <= 0 {
				return errors.New("training data: processor emitted invalid shape")
			}
		}
	}
	return nil
}
