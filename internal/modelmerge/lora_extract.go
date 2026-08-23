package modelmerge

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/tensor"
)

const loRAExtractionSchema = "overgo/lora-extraction/v1"

var loRAExtractionContract = artifact.JSONContract(artifact.KindAdapter, loRAExtractionSchema)

// LoRAExtractionPolicy owns the acceptable relative Frobenius reconstruction
// residual. Rank is derived by meeting this bound, never supplied directly.
type LoRAExtractionPolicy struct {
	MaximumRelativeResidual float64 `json:"maximum_relative_residual"`
	Rationale               string  `json:"rationale"`
	ReopenTrigger           string  `json:"reopen_trigger"`
}

// LoRAFactors are row-major up and down matrices whose product reconstructs
// one fine-tuning delta. Down is rank-by-columns; Up is rows-by-rank.
type LoRAFactors struct {
	Rows             uint64    `json:"rows"`
	Columns          uint64    `json:"columns"`
	Rank             uint32    `json:"rank"`
	Down             []float32 `json:"down"`
	Up               []float32 `json:"up"`
	RelativeResidual float64   `json:"relative_residual"`
}

// LoRAExtraction is one content-addressed, exact-base adapter artifact.
type LoRAExtraction struct {
	Base       artifact.ID            `json:"base"`
	Tuned      artifact.ID            `json:"tuned"`
	Definition artifact.ID            `json:"definition"`
	Policy     LoRAExtractionPolicy   `json:"policy"`
	Weights    map[string]LoRAFactors `json:"weights"`
	ID         artifact.ID            `json:"-"`
}

// ExtractLoRA derives the minimum evidenced rank for every changed matrix in
// an exact-base fine-tuned snapshot.
func ExtractLoRA(base, tuned Snapshot, policy LoRAExtractionPolicy) (LoRAExtraction, error) {
	if err := validateLoRAExtractionInputs(base, tuned, policy); err != nil {
		return LoRAExtraction{}, err
	}
	weights := make(map[string]LoRAFactors)
	names := make([]string, 0, len(base.Tensors))
	for name := range base.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		baseWeight, tunedWeight := base.Tensors[name], tuned.Tensors[name]
		if slices.Equal(baseWeight.Values, tunedWeight.Values) {
			continue
		}
		columns, rows, matrix := tensor.MatrixExtents(baseWeight.Layout)
		if !matrix {
			return LoRAExtraction{}, errors.New("model merge: changed LoRA tensor is not a matrix")
		}
		factors, err := factorDelta(baseWeight.Values, tunedWeight.Values, int(rows), int(columns), policy.MaximumRelativeResidual)
		if err != nil {
			return LoRAExtraction{}, err
		}
		weights[name] = factors
	}
	if len(weights) == tensor.FirstOffset {
		return LoRAExtraction{}, errors.New("model merge: LoRA extraction has no changed matrices")
	}
	result := LoRAExtraction{
		Base: base.ID, Tuned: tuned.ID, Definition: base.Definition,
		Policy: policy, Weights: weights,
	}
	id, err := artifact.JSONID(artifact.KindAdapter, result)
	if err != nil {
		return LoRAExtraction{}, err
	}
	result.ID = id
	return result, nil
}

// Lineage binds the adapter to the exact frozen base, tuned derivative, and definition.
func (value LoRAExtraction) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Base, value.Tuned, value.Definition)
}

// Content returns the immutable extracted adapter document.
func (value LoRAExtraction) Content() (artifact.Content, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return artifact.Content{}, err
	}
	return loRAExtractionContract.Content(value.ID, data)
}

// Reconstruct applies extracted factors to a frozen base snapshot.
func (value LoRAExtraction) Reconstruct(base Snapshot) (Snapshot, error) {
	if base.ID != value.Base || base.Definition != value.Definition || base.ValidateIdentity() != nil {
		return Snapshot{}, errors.New("model merge: LoRA reconstruction base differs")
	}
	weights := cloneWeights(base.Tensors)
	for name, factors := range value.Weights {
		weight, found := weights[name]
		if !found || weight.Layout.Dims[tensor.FirstOffset] != factors.Columns ||
			weight.Layout.Dims[tensor.SingletonExtent] != factors.Rows {
			return Snapshot{}, errors.New("model merge: LoRA reconstruction layout differs")
		}
		for row := range int(factors.Rows) {
			for column := range int(factors.Columns) {
				var delta float32
				for rank := range int(factors.Rank) {
					delta += factors.Up[row*int(factors.Rank)+rank] * factors.Down[rank*int(factors.Columns)+column]
				}
				weight.Values[row*int(factors.Columns)+column] += delta
			}
		}
		weights[name] = weight
	}
	return (Compiler{}).Seal(base.Definition, base.ID, weights)
}

func validateLoRAExtractionInputs(base, tuned Snapshot, policy LoRAExtractionPolicy) error {
	if base.ValidateIdentity() != nil || tuned.ValidateIdentity() != nil || base.Base.Valid() ||
		tuned.Base != base.ID || compatibleInventory(base, tuned) != nil ||
		!checked.NonNegativeFinite64(policy.MaximumRelativeResidual) || policy.MaximumRelativeResidual >= 1 ||
		strings.TrimSpace(policy.Rationale) != policy.Rationale || strings.TrimSpace(policy.ReopenTrigger) != policy.ReopenTrigger ||
		policy.Rationale == "" || policy.ReopenTrigger == "" {
		return errors.New("model merge: invalid LoRA extraction authority or policy")
	}
	return nil
}

func factorDelta(base, tuned []float32, rows, columns int, budget float64) (LoRAFactors, error) {
	if rows <= 0 || columns <= 0 || len(base) != rows*columns || len(tuned) != len(base) {
		return LoRAFactors{}, errors.New("model merge: invalid LoRA delta geometry")
	}
	residual := make([]float64, len(base))
	var originalSquared float64
	for index := range residual {
		residual[index] = float64(tuned[index] - base[index])
		originalSquared += residual[index] * residual[index]
	}
	if !checked.PositiveFinite64(originalSquared) {
		return LoRAFactors{}, errors.New("model merge: invalid LoRA delta")
	}
	ceiling := min(rows, columns)
	down := make([]float32, 0, ceiling*columns)
	up := make([]float32, 0, rows*ceiling)
	var relative float64
	for component := range ceiling {
		u, v, sigma, ok := dominantDeltaDirection(residual, rows, columns)
		if !ok {
			return LoRAFactors{}, errors.New("model merge: LoRA factorization stalled before residual budget")
		}
		for column := range columns {
			down = append(down, float32(v[column]))
		}
		for row := range rows {
			up = append(up, float32(sigma*u[row]))
			for column := range columns {
				residual[row*columns+column] -= sigma * u[row] * v[column]
			}
		}
		relative = relativeFrobenius(residual, originalSquared)
		if relative <= budget {
			rank := component + tensor.SingletonExtent
			return LoRAFactors{
				Rows: uint64(rows), Columns: uint64(columns), Rank: uint32(rank),
				Down: down, Up: transposeRankMajorUp(up, rows, rank), RelativeResidual: relative,
			}, nil
		}
	}
	return LoRAFactors{}, errors.New("model merge: LoRA full-rank reconstruction exceeds residual budget")
}

func dominantDeltaDirection(matrix []float64, rows, columns int) ([]float64, []float64, float64, bool) {
	v := make([]float64, columns)
	bestRow, bestNorm := tensor.FirstOffset, float64(0)
	for row := range rows {
		var norm float64
		for column := range columns {
			value := matrix[row*columns+column]
			norm += value * value
		}
		if norm > bestNorm {
			bestRow, bestNorm = row, norm
		}
	}
	if !checked.PositiveFinite64(bestNorm) {
		return nil, nil, 0, false
	}
	copy(v, matrix[bestRow*columns:(bestRow+tensor.SingletonExtent)*columns])
	normalizeVector(v)
	u := make([]float64, rows)
	for range rows + columns {
		multiplyMatrixVector(u, matrix, v, rows, columns)
		if !normalizeVector(u) {
			return nil, nil, 0, false
		}
		multiplyTransposeVector(v, matrix, u, rows, columns)
		if !normalizeVector(v) {
			return nil, nil, 0, false
		}
	}
	multiplyMatrixVector(u, matrix, v, rows, columns)
	sigma := vectorNorm(u)
	if !checked.PositiveFinite64(sigma) || !normalizeVector(u) {
		return nil, nil, 0, false
	}
	return u, v, sigma, true
}

func multiplyMatrixVector(output, matrix, vector []float64, rows, columns int) {
	for row := range rows {
		var value float64
		for column := range columns {
			value += matrix[row*columns+column] * vector[column]
		}
		output[row] = value
	}
}

func multiplyTransposeVector(output, matrix, vector []float64, rows, columns int) {
	for column := range columns {
		var value float64
		for row := range rows {
			value += matrix[row*columns+column] * vector[row]
		}
		output[column] = value
	}
}

func normalizeVector(values []float64) bool {
	norm := vectorNorm(values)
	if !checked.PositiveFinite64(norm) {
		return false
	}
	for index := range values {
		values[index] /= norm
	}
	return true
}

func vectorNorm(values []float64) float64 {
	var squared float64
	for _, value := range values {
		squared += value * value
	}
	return math.Sqrt(squared)
}

func relativeFrobenius(residual []float64, originalSquared float64) float64 {
	var squared float64
	for _, value := range residual {
		squared += value * value
	}
	return math.Sqrt(squared / originalSquared)
}

func transposeRankMajorUp(rankMajor []float32, rows, rank int) []float32 {
	result := make([]float32, rows*rank)
	for component := range rank {
		for row := range rows {
			result[row*rank+component] = rankMajor[component*rows+row]
		}
	}
	return result
}
