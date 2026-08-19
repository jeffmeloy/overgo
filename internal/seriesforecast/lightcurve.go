package seriesforecast

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"

	"overgo/internal/binaryschema"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

// LightCurveProcessor converts (time, wavelength, flux, error) tuples into a
// forecast pair. Band selection is data-derived; the boundary is model-derived.
func LightCurveProcessor(patchLen, horizon int) (trainingdata.Processor, error) {
	return lightCurveProcessor(patchLen, horizon, patchLen)
}

// LightCurveTrainingProcessor pairs like LightCurveProcessor but reserves at
// least TWO patches of context. A single-patch context degenerates attention
// to one position, where the softmax is constant and the q/k projections,
// both attention norms and the per-dim scale receive exactly zero gradient —
// an evaluation pairing, not a trainable one.
func LightCurveTrainingProcessor(patchLen, horizon int) (trainingdata.Processor, error) {
	return lightCurveProcessor(patchLen, horizon, 2*patchLen)
}

func lightCurveProcessor(patchLen, horizon, minContext int) (trainingdata.Processor, error) {
	if patchLen <= 0 || horizon <= 0 {
		return nil, errors.New("seriesforecast: invalid light-curve geometry")
	}
	return func(_ context.Context, record trainingdata.RawRecord) (trainingdata.Example, error) {
		contextValues, target, err := lightCurvePair(record.Data, minContext, horizon)
		if err != nil {
			return trainingdata.Example{}, err
		}
		return trainingdata.Example{
			ID: record.ID, Group: record.Group,
			Values: []trainingdata.Value{
				seriesValue(trainingdata.RoleInput, contextValues),
				seriesValue(trainingdata.RoleTarget, target),
			},
		}, nil
	}, nil
}

type lightCurveRow struct {
	Values []float64 `json:"x"`
}

type lightObservation struct {
	time float64
	flux float32
}

func lightCurvePair(data []byte, minContext, horizon int) ([]float32, []float32, error) {
	var row lightCurveRow
	if err := json.Unmarshal(data, &row); err != nil {
		return nil, nil, fmt.Errorf("seriesforecast: decode light curve: %w", err)
	}
	if len(row.Values) == 0 || len(row.Values)%4 != 0 {
		return nil, nil, errors.New("seriesforecast: malformed light curve")
	}
	bands := make(map[float64][]lightObservation)
	for offset := 0; offset < len(row.Values); offset += 4 {
		timestamp, wavelength, flux, uncertainty := row.Values[offset], row.Values[offset+1], row.Values[offset+2], row.Values[offset+3]
		if timestamp == 0 && wavelength == 0 && flux == 0 && uncertainty == 0 {
			continue
		}
		if !finite64(timestamp) || !finite64(wavelength) || !finite64(flux) || !finite64(uncertainty) || wavelength <= 0 || uncertainty < 0 {
			return nil, nil, errors.New("seriesforecast: invalid light-curve observation")
		}
		value := float32(flux)
		if math.IsInf(float64(value), 0) {
			return nil, nil, errors.New("seriesforecast: light-curve flux exceeds float32")
		}
		bands[wavelength] = append(bands[wavelength], lightObservation{time: timestamp, flux: value})
	}
	var selected float64
	for wavelength, observations := range bands {
		if len(observations) > len(bands[selected]) || len(observations) == len(bands[selected]) && (selected == 0 || wavelength < selected) {
			selected = wavelength
		}
	}
	observations := bands[selected]
	if len(observations) <= minContext {
		return nil, nil, fmt.Errorf("seriesforecast: light curve has %d usable band samples; need more than context floor %d", len(observations), minContext)
	}
	sort.SliceStable(observations, func(left, right int) bool { return observations[left].time < observations[right].time })
	values := make([]float32, len(observations))
	for index, observation := range observations {
		values[index] = observation.flux
	}
	targetLen := min(horizon, len(values)-minContext)
	boundary := len(values) - targetLen
	return values[:boundary], values[boundary:], nil
}

func finite64(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func seriesValue(role trainingdata.ValueRole, values []float32) trainingdata.Value {
	return trainingdata.Value{
		Role: role, Modality: recipecontract.ModalityTimeSeries, Encoding: trainingdata.EncodingFloat32LE,
		Shape: []int{len(values)}, Data: binaryschema.LittleEndian.Float32s(values),
	}
}

// TrainingPair decodes one typed time-series example for forecast execution.
func TrainingPair(example trainingdata.Example) ([]float32, []float32, error) {
	var input, target trainingdata.Value
	for _, value := range example.Values {
		if value.Modality != recipecontract.ModalityTimeSeries {
			continue
		}
		switch value.Role {
		case trainingdata.RoleInput:
			if len(input.Data) != 0 {
				return nil, nil, errors.New("seriesforecast: multiple series inputs")
			}
			input = value
		case trainingdata.RoleTarget:
			if len(target.Data) != 0 {
				return nil, nil, errors.New("seriesforecast: multiple series targets")
			}
			target = value
		}
	}
	if len(input.Shape) != 1 || len(target.Shape) != 1 {
		return nil, nil, errors.New("seriesforecast: one-dimensional input and target required")
	}
	contextValues, err := trainingdata.Float32(input)
	if err != nil {
		return nil, nil, err
	}
	targetValues, err := trainingdata.Float32(target)
	if err != nil {
		return nil, nil, err
	}
	if len(contextValues) != input.Shape[0] || len(targetValues) != target.Shape[0] {
		return nil, nil, errors.New("seriesforecast: series payload differs from shape")
	}
	return contextValues, targetValues, nil
}
