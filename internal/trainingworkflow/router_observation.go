package trainingworkflow

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/densecausal"
)

type routerStepObservation struct {
	Step        int                              `json:"step"`
	Observation densecausal.MoERouterObservation `json:"observation"`
}

type routerObservationRecord struct {
	step, layer int
	data        []byte
}

type routerObservationKey struct {
	step, layer int
}

type routerObservationExpectation struct {
	firstStep uint64
	steps     uint64
	layers    []uint32
}

type routerObservationCollector struct {
	expectation routerObservationExpectation
	endStep     uint64
	allowed     map[int]bool
	seen        map[routerObservationKey]bool
	records     []routerObservationRecord
	encoded     uint64
	expected    uint64
}

func newRouterObservationCollector(firstStep, steps int, layers []int) (*routerObservationCollector, error) {
	if firstStep < 0 || steps <= 0 || len(layers) == 0 {
		return nil, errors.New("training workflow: invalid router observation coverage")
	}
	canonicalLayers := slices.Clone(layers)
	slices.Sort(canonicalLayers)
	if len(slices.Compact(slices.Clone(canonicalLayers))) != len(canonicalLayers) {
		return nil, errors.New("training workflow: duplicate routed layer declaration")
	}
	first := uint64(firstStep)
	count := uint64(steps)
	end, ok := checked.Add64(first, count)
	if !ok {
		return nil, errors.New("training workflow: router observation step interval overflows")
	}
	expected, ok := checked.Mul64(count, uint64(len(canonicalLayers)))
	if !ok {
		return nil, errors.New("training workflow: router observation cardinality overflows")
	}
	allowed := make(map[int]bool, len(canonicalLayers))
	coverageLayers := make([]uint32, len(canonicalLayers))
	for index, layer := range canonicalLayers {
		if layer < 0 || uint64(layer) > math.MaxUint32 {
			return nil, errors.New("training workflow: invalid routed layer declaration")
		}
		allowed[layer] = true
		coverageLayers[index] = uint32(layer)
	}
	return &routerObservationCollector{
		expectation: routerObservationExpectation{firstStep: first, steps: count, layers: coverageLayers},
		endStep:     end, allowed: allowed, seen: make(map[routerObservationKey]bool), expected: expected,
	}, nil
}

func (collector *routerObservationCollector) observe(step int, observation densecausal.MoERouterObservation) error {
	if collector == nil || step < 0 || uint64(step) < collector.expectation.firstStep || uint64(step) >= collector.endStep ||
		!collector.allowed[observation.Layer] {
		return errors.New("training workflow: unexpected router observation step or layer")
	}
	key := routerObservationKey{step: step, layer: observation.Layer}
	if collector.seen[key] {
		return errors.New("training workflow: duplicate router observation")
	}
	data, err := json.Marshal(routerStepObservation{Step: step, Observation: observation})
	if err != nil {
		return fmt.Errorf("training workflow: encode router observation: %w", err)
	}
	next, ok := checked.Add64(collector.encoded, uint64(len(data)))
	if !ok || next > artifact.MaxContentBytes {
		return errors.New("training workflow: router observations exceed bounded capture")
	}
	collector.encoded = next
	collector.seen[key] = true
	collector.records = append(collector.records, routerObservationRecord{
		step: step, layer: observation.Layer, data: data,
	})
	return nil
}

func (collector *routerObservationCollector) complete() ([]routerObservationRecord, routerObservationExpectation, error) {
	if collector == nil || uint64(len(collector.records)) != collector.expected {
		return nil, routerObservationExpectation{}, errors.New("training workflow: incomplete router observation coverage")
	}
	slices.SortFunc(collector.records, func(left, right routerObservationRecord) int {
		if order := cmp.Compare(left.step, right.step); order != 0 {
			return order
		}
		return cmp.Compare(left.layer, right.layer)
	})
	return slices.Clone(collector.records), collector.expectation, nil
}

func decodeRouterObservation(record routerObservationRecord) (routerStepObservation, error) {
	var value routerStepObservation
	if err := json.Unmarshal(record.data, &value); err != nil {
		return routerStepObservation{}, fmt.Errorf("training workflow: decode router observation: %w", err)
	}
	if value.Step != record.step || value.Observation.Layer != record.layer {
		return routerStepObservation{}, errors.New("training workflow: router observation record identity differs")
	}
	return value, nil
}
