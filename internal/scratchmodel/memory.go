package scratchmodel

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingprogram"
)

// TrainingMemorySegments derives exact flat training-state transfer geometry.
func (c Construction) TrainingMemorySegments() ([]trainingprogram.MemorySegment, error) {
	if c.optimizer.Identity() == "" || c.optimizer.GroupCount() == 0 {
		return nil, errors.New("scratch model: optimizer memory geometry absent")
	}
	type aggregate struct {
		parameters uint64
		retained   bool
	}
	groups := make(map[string]aggregate)
	for index := range c.optimizer.GroupCount() {
		group, ok := c.optimizer.Group(index)
		if !ok || group.End <= group.Start {
			return nil, errors.New("scratch model: optimizer memory group differs")
		}
		name, retained, err := memorySegmentName(group.Name)
		if err != nil {
			return nil, err
		}
		bytes, err := dtype.F32.StorageBytes(uint64(group.End-group.Start), uint64(group.Cols))
		if err != nil {
			return nil, err
		}
		value := groups[name]
		value.parameters += bytes
		value.retained = value.retained || retained
		groups[name] = value
	}
	result := make([]trainingprogram.MemorySegment, 0, len(groups))
	for name, value := range groups {
		result = append(result, trainingprogram.MemorySegment{
			Name: name, ParameterBytes: value.parameters, GradientBytes: value.parameters,
			OptimizerBytes: value.parameters, Retained: value.retained,
		})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result, nil
}

func memorySegmentName(parameter string) (string, bool, error) {
	if parameter == "lt" {
		return "00-retained-temperature", true, nil
	}
	switch parameter {
	case "wte", "wpe", "pos_bias":
		return "01-input", false, nil
	case "lm_head":
		return "99-output", false, nil
	}
	if strings.HasPrefix(parameter, "l") {
		separator := strings.IndexByte(parameter, '.')
		if separator <= 1 {
			return "", false, fmt.Errorf("scratch model: invalid layer parameter %q", parameter)
		}
		layer, err := strconv.Atoi(parameter[1:separator])
		if err != nil || layer < 0 {
			return "", false, fmt.Errorf("scratch model: invalid layer parameter %q", parameter)
		}
		return fmt.Sprintf("10-layer-%06d", layer), false, nil
	}
	return "", false, fmt.Errorf("scratch model: unclassified memory parameter %q", parameter)
}
