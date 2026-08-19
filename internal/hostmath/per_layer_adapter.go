package hostmath

import "fmt"

// PerLayerAdapterTrace retains the exact adapter forward needed by its VJP.
type PerLayerAdapterTrace struct {
	Gate       []float32
	Activated  []float32
	Projection []float32
}

// PerLayerAdapterForward applies the artifact-declared per-layer residual adapter.
func PerLayerAdapterForward(input, side, gateWeight, projectionWeight, normWeight []float32, rows, hidden, width int, epsilon float64) ([]float32, PerLayerAdapterTrace, error) {
	if rows <= 0 || hidden <= 0 || width <= 0 || epsilon <= 0 ||
		len(input) != rows*hidden || len(side) != rows*width ||
		len(gateWeight) != width*hidden || len(projectionWeight) != hidden*width || len(normWeight) != hidden {
		return nil, PerLayerAdapterTrace{}, fmt.Errorf("per-layer adapter: shape mismatch")
	}
	trace := PerLayerAdapterTrace{
		Gate: make([]float32, rows*width), Activated: make([]float32, rows*width),
		Projection: make([]float32, rows*hidden),
	}
	Linear(trace.Gate, input, gateWeight, rows, hidden, width)
	for index, value := range trace.Gate {
		trace.Activated[index] = float32(GELUTanh(float64(value))) * side[index]
	}
	Linear(trace.Projection, trace.Activated, projectionWeight, rows, width, hidden)
	normalized := make([]float32, rows*hidden)
	RMSNormInto(normalized, trace.Projection, normWeight, rows, hidden, epsilon)
	output := make([]float32, len(input))
	for index := range output {
		output[index] = input[index] + normalized[index]
	}
	return output, trace, nil
}

// PerLayerAdapterBackward writes parameter VJPs to optimizer-owned storage.
func PerLayerAdapterBackward(
	input, side, gateWeight, projectionWeight, normWeight, dOutput,
	dGateWeight, dProjectionWeight, dNormWeight []float32,
	rows, hidden, width int, epsilon float64, trace PerLayerAdapterTrace,
) (dInput, dSide []float32, err error) {
	if len(dOutput) != rows*hidden || len(trace.Gate) != rows*width ||
		len(trace.Activated) != rows*width || len(trace.Projection) != rows*hidden ||
		len(dGateWeight) != len(gateWeight) || len(dProjectionWeight) != len(projectionWeight) || len(dNormWeight) != len(normWeight) {
		return nil, nil, fmt.Errorf("per-layer adapter backward: trace or destination mismatch")
	}
	dInput = append([]float32(nil), dOutput...)
	dSide = make([]float32, rows*width)
	clear(dGateWeight)
	clear(dProjectionWeight)
	clear(dNormWeight)
	dProjection := make([]float32, rows*hidden)
	RMSNormBackward(dProjection, dNormWeight, trace.Projection, normWeight, dOutput, rows, hidden, epsilon, false)
	dActivated := make([]float32, rows*width)
	LinearBackward(dActivated, dProjectionWeight, nil, trace.Activated, projectionWeight, dProjection, rows, width, hidden, false)
	dGate := make([]float32, rows*width)
	for index, value := range trace.Gate {
		gelu := GELUTanh(float64(value))
		dSide[index] = dActivated[index] * float32(gelu)
		dGate[index] = dActivated[index] * side[index] * float32(GELUTanhPrime(float64(value)))
	}
	LinearBackward(dInput, dGateWeight, nil, input, gateWeight, dGate, rows, hidden, width, true)
	return dInput, dSide, nil
}
