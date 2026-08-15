package trainingprogram

import (
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

type MemoryTier string

const (
	MemoryTierDevice MemoryTier = "device"
	MemoryTierHost   MemoryTier = "host"
)

// MemorySegment: one ordered trainable-state transfer unit.
type MemorySegment struct {
	Name           string
	ParameterBytes uint64
	GradientBytes  uint64
	OptimizerBytes uint64
	Retained       bool
}

type MemoryScheduleSpec struct {
	Calibration           artifact.ID
	AllocationGranularity uint64
	MeasuredResidentPeak  uint64
	ConnectorDeviceBytes  uint64
	DeviceBudget          uint64
	HostBudget            uint64
	Segments              []MemorySegment
}

// MemorySchedule: measured resident or forced host-streaming plan.
type MemorySchedule struct {
	id                    artifact.ID
	tier                  MemoryTier
	calibration           artifact.ID
	segments              []MemorySegment
	residentPeakBytes     uint64
	tier1PeakBytes        uint64
	residentCapacityBytes uint64
	tier1CapacityBytes    uint64
	fixedDeviceBytes      uint64
	hostBytes             uint64
	connectorBytes        uint64
	doubleBuffered        bool
}

func CompileMemorySchedule(spec MemoryScheduleSpec) (MemorySchedule, error) {
	segments, totalState, retainedState, streamedState, maxStream, maxPrefetch, err := compileMemorySegments(spec.Segments)
	if err != nil {
		return MemorySchedule{}, err
	}
	if spec.Calibration.Kind() != artifact.KindEvidence || spec.AllocationGranularity == 0 ||
		spec.MeasuredResidentPeak == 0 || spec.DeviceBudget == 0 || spec.HostBudget == 0 ||
		spec.MeasuredResidentPeak < totalState || spec.ConnectorDeviceBytes > spec.MeasuredResidentPeak-totalState {
		return MemorySchedule{}, errors.New("training program: invalid memory schedule measurement")
	}
	fixed := spec.MeasuredResidentPeak - totalState
	residentCapacity, ok := roundedMemory(spec.MeasuredResidentPeak, spec.AllocationGranularity)
	if !ok {
		return MemorySchedule{}, errors.New("training program: resident capacity overflows")
	}
	active, ok := sumMemory(fixed, retainedState, maxStream, maxPrefetch)
	if !ok {
		return MemorySchedule{}, errors.New("training program: Tier-1 capacity overflows")
	}
	tier1Capacity, ok := roundedMemory(active, spec.AllocationGranularity)
	if !ok {
		return MemorySchedule{}, errors.New("training program: Tier-1 rounded capacity overflows")
	}
	tier := MemoryTierDevice
	hostBytes, doubleBuffered := uint64(0), false
	if spec.DeviceBudget < spec.MeasuredResidentPeak {
		if streamedState == 0 || spec.DeviceBudget < active || spec.HostBudget < streamedState {
			return MemorySchedule{}, errors.New("training program: measured Tier-1 streaming does not fit; optional mechanisms lack evidence")
		}
		tier, hostBytes, doubleBuffered = MemoryTierHost, streamedState, streamedState > maxStream
	}
	body := struct {
		Calibration           artifact.ID
		AllocationGranularity uint64
		MeasuredResidentPeak  uint64
		ConnectorDeviceBytes  uint64
		DeviceBudget          uint64
		HostBudget            uint64
		Segments              []MemorySegment
		Tier                  MemoryTier
		Tier1Peak             uint64
		ResidentCapacity      uint64
		Tier1Capacity         uint64
		FixedDevice           uint64
		HostBytes             uint64
		DoubleBuffered        bool
	}{
		Calibration: spec.Calibration, AllocationGranularity: spec.AllocationGranularity,
		MeasuredResidentPeak: spec.MeasuredResidentPeak, ConnectorDeviceBytes: spec.ConnectorDeviceBytes,
		DeviceBudget: spec.DeviceBudget, HostBudget: spec.HostBudget, Segments: segments, Tier: tier,
		Tier1Peak: active, ResidentCapacity: residentCapacity, Tier1Capacity: tier1Capacity,
		FixedDevice: fixed, HostBytes: hostBytes, DoubleBuffered: doubleBuffered,
	}
	id, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return MemorySchedule{}, err
	}
	return MemorySchedule{
		id: id, tier: tier, calibration: spec.Calibration, segments: segments,
		residentPeakBytes: spec.MeasuredResidentPeak, tier1PeakBytes: active,
		residentCapacityBytes: residentCapacity, tier1CapacityBytes: tier1Capacity,
		fixedDeviceBytes: fixed, hostBytes: hostBytes, connectorBytes: spec.ConnectorDeviceBytes,
		doubleBuffered: doubleBuffered,
	}, nil
}

func (s MemorySchedule) ID() artifact.ID                   { return s.id }
func (s MemorySchedule) Tier() MemoryTier                  { return s.tier }
func (s MemorySchedule) Calibration() artifact.ID          { return s.calibration }
func (s MemorySchedule) Segments() []MemorySegment         { return slices.Clone(s.segments) }
func (s MemorySchedule) ResidentPeakBytes() uint64         { return s.residentPeakBytes }
func (s MemorySchedule) Tier1PeakBytes() uint64            { return s.tier1PeakBytes }
func (s MemorySchedule) ResidentCapacityBytes() uint64     { return s.residentCapacityBytes }
func (s MemorySchedule) Tier1CapacityBytes() uint64        { return s.tier1CapacityBytes }
func (s MemorySchedule) FixedDeviceBytes() uint64          { return s.fixedDeviceBytes }
func (s MemorySchedule) HostBytes() uint64                 { return s.hostBytes }
func (s MemorySchedule) ConnectorDeviceBytes() uint64      { return s.connectorBytes }
func (s MemorySchedule) DoubleBuffered() bool              { return s.doubleBuffered }
func (s MemorySchedule) UsesActivationCheckpointing() bool { return false }
func (s MemorySchedule) UsesLayerMajorAccumulation() bool  { return false }
func (s MemorySchedule) UsesOptimizerPaging() bool         { return false }

func compileMemorySegments(source []MemorySegment) ([]MemorySegment, uint64, uint64, uint64, uint64, uint64, error) {
	segments := slices.Clone(source)
	if len(segments) < 2 {
		return nil, 0, 0, 0, 0, 0, errors.New("training program: memory schedule requires multiple segments")
	}
	sort.Slice(segments, func(i, j int) bool { return segments[i].Name < segments[j].Name })
	var total, retained, streamed, maxStream, maxPrefetch uint64
	for index, segment := range segments {
		if strings.TrimSpace(segment.Name) == "" || strings.TrimSpace(segment.Name) != segment.Name ||
			strings.ContainsAny(segment.Name, "\x00\r\n") || index > 0 && segments[index-1].Name == segment.Name ||
			segment.ParameterBytes == 0 || segment.GradientBytes == 0 || segment.OptimizerBytes == 0 {
			return nil, 0, 0, 0, 0, 0, errors.New("training program: invalid memory segment")
		}
		segmentTotal, ok := sumMemory(segment.ParameterBytes, segment.GradientBytes, segment.OptimizerBytes)
		if !ok {
			return nil, 0, 0, 0, 0, 0, errors.New("training program: memory segment overflows")
		}
		total, ok = addMemory(total, segmentTotal)
		if !ok {
			return nil, 0, 0, 0, 0, 0, errors.New("training program: memory state overflows")
		}
		if segment.Retained {
			retained, ok = addMemory(retained, segmentTotal)
		} else {
			streamed, ok = addMemory(streamed, segmentTotal)
			maxStream = max(maxStream, segmentTotal)
			maxPrefetch = max(maxPrefetch, segment.ParameterBytes)
		}
		if !ok {
			return nil, 0, 0, 0, 0, 0, errors.New("training program: memory class overflows")
		}
	}
	return segments, total, retained, streamed, maxStream, maxPrefetch, nil
}

func roundedMemory(value, granularity uint64) (uint64, bool) {
	if value == 0 || granularity == 0 {
		return 0, false
	}
	remainder := value % granularity
	if remainder == 0 {
		return value, true
	}
	return addMemory(value, granularity-remainder)
}

func sumMemory(values ...uint64) (uint64, bool) {
	var total uint64
	for _, value := range values {
		var ok bool
		total, ok = addMemory(total, value)
		if !ok {
			return 0, false
		}
	}
	return total, true
}

func addMemory(left, right uint64) (uint64, bool) {
	if left > math.MaxUint64-right {
		return 0, false
	}
	return left + right, true
}
