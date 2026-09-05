package checked

import (
	"errors"
	"math"
	"unsafe"
)

func Nonzero[T comparable](value T) bool {
	var zero T
	return value != zero
}

// NonzeroAll reports whether every value differs from its type's zero value.
func NonzeroAll[T comparable](values ...T) bool {
	for _, value := range values {
		if !Nonzero(value) {
			return false
		}
	}
	return true
}

// Equal reports whether two comparable values are equal.
func Equal[T comparable](left, right T) bool { return left == right }

func Finite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func Finite64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func PositiveFinite32(value float32) bool { return value > 0 && Finite32(value) }

func PositiveFinite64(value float64) bool { return value > 0 && Finite64(value) }

func NonNegativeFinite32(value float32) bool { return value >= 0 && Finite32(value) }

// NonNegativeFinite64 reports whether value is finite and not negative.
func NonNegativeFinite64(value float64) bool { return value >= 0 && Finite64(value) }

// UnitInterval64 reports whether value is finite and lies in [0, 1].
func UnitInterval64(value float64) bool { return value >= 0 && value <= 1 && Finite64(value) }

// AtLeastFinite64 reports whether both operands are finite and value meets the minimum.
func AtLeastFinite64(value, minimum float64) bool {
	return Finite64(value) && Finite64(minimum) && value >= minimum
}

// NonNegativeInt64 reports whether value is not negative.
func NonNegativeInt64(value int64) bool { return value >= 0 }

// AtMostInt64 reports whether value does not exceed maximum.
func AtMostInt64(value, maximum int64) bool { return value <= maximum }

// GreaterInt reports whether left is greater than right.
func GreaterInt(left, right int) bool { return left > right }

// AtMostInt reports whether value does not exceed maximum.
func AtMostInt(value, maximum int) bool { return value <= maximum }

// AtLeastInt reports whether value meets minimum.
func AtLeastInt(value, minimum int) bool { return value >= minimum }

// UnknownCount returns the diagnostic sentinel for an unavailable count.
func UnknownCount() int { return -1 }

func PositiveInts(values ...int) bool {
	for _, value := range values {
		if value <= 0 {
			return false
		}
	}
	return true
}

func NonNegativeInts(values ...int) bool {
	for _, value := range values {
		if value < 0 {
			return false
		}
	}
	return true
}

// AtMost64 compares bounded unsigned extents.
func AtMost64(value, limit uint64) bool { return value <= limit }

// First returns the first value and reports whether it exists.
func First[T any](values []T) (T, bool) {
	var zero T
	if len(values) == 0 {
		return zero, false
	}
	return values[0], true
}

// Last returns the final value and reports whether it exists.
func Last[T any](values []T) (T, bool) {
	var zero T
	if len(values) == 0 {
		return zero, false
	}
	return values[len(values)-1], true
}

// Multiple reports whether count represents more than one item.
func Multiple(count int) bool { return count > 1 }

// EvenInt reports whether value is positive and even.
func EvenInt(value int) bool { return value > 0 && value%2 == 0 }

// Empty reports whether every supplied slice has no elements.
func Empty[T any](values ...[]T) bool {
	for _, value := range values {
		if len(value) != 0 {
			return false
		}
	}
	return true
}

// SlicesOverlap reports shared occupied storage in two typed slices in O(1).
// Empty and zero-sized-element slices occupy no writable bytes.
// Callers must pass the complete writable destination extent, not just its
// current length, when checking a buffer that will be extended to capacity.
func SlicesOverlap[T any](left, right []T) bool {
	if len(left) == 0 || len(right) == 0 {
		return false
	}
	// Addresses are compared only; no pointer is manufactured or dereferenced.
	// Actual Go slice storage guarantees each byte extent is addressable.
	width := unsafe.Sizeof(left[0])
	leftStart, rightStart := uintptr(unsafe.Pointer(unsafe.SliceData(left))), uintptr(unsafe.Pointer(unsafe.SliceData(right)))
	if leftStart <= rightStart {
		return rightStart-leftStart < uintptr(len(left))*width
	}
	return leftStart-rightStart < uintptr(len(right))*width
}

// Nonempty reports whether value contains at least one element.
func Nonempty[T any](value []T) bool { return len(value) != 0 }

// NonemptyAll reports whether every supplied slice contains at least one element.
func NonemptyAll[T any](values ...[]T) bool {
	for _, value := range values {
		if len(value) == 0 {
			return false
		}
	}
	return true
}

// Suffix returns the final count elements when count is in range.
func Suffix[T any](values []T, count int) ([]T, bool) {
	if count < 0 || count > len(values) {
		return nil, false
	}
	return values[len(values)-count:], true
}

// Prefix returns the first count elements when count is in range.
func Prefix[T any](values []T, count int) ([]T, bool) {
	if count < 0 || count > len(values) {
		return nil, false
	}
	return values[:count], true
}

// SuffixExact returns the final count elements or an error when count is invalid.
func SuffixExact[T any](values []T, count int) ([]T, error) {
	result, ok := Suffix(values, count)
	if !ok {
		return nil, errors.New("checked: invalid suffix count")
	}
	return result, nil
}

// Length validates flat storage against a product of positive extents.
func Length[T any](values []T, extents ...int) error {
	product := 1
	for _, extent := range extents {
		var ok bool
		product, ok = MulInt(product, extent)
		if !ok || extent <= 0 {
			return errors.New("checked: invalid storage extent")
		}
	}
	if len(values) != product {
		return errors.New("checked: storage length differs from extents")
	}
	return nil
}

// Rows derives a positive row count from flat storage and a fixed row width.
func Rows[T any](values []T, width int) (int, error) {
	if len(values) == 0 || width <= 0 || len(values)%width != 0 {
		return 0, errors.New("checked: storage is not row-aligned")
	}
	return len(values) / width, nil
}

// ValidIndex reports whether value is a zero-based index within length.
func ValidIndex(value, length int) bool { return value >= 0 && value < length }

// ReverseIndex maps a zero-based index to its position from the opposite end.
func ReverseIndex(value, length int) int { return length - 1 - value }

// ValidOneBasedIndex reports whether value addresses one of length elements
// using the one-based indexing convention common in external configurations.
func ValidOneBasedIndex(value, length int) bool { return value > 0 && value <= length }

// StrictlyIncreasingOneBased validates an ordered set of one-based indices.
func StrictlyIncreasingOneBased(values []int, length int) bool {
	if len(values) == 0 {
		return false
	}
	previous := 0
	for _, value := range values {
		if !ValidOneBasedIndex(value, length) || value <= previous {
			return false
		}
		previous = value
	}
	return true
}

func Add64(values ...uint64) (uint64, bool) {
	var total uint64
	for _, value := range values {
		if total > math.MaxUint64-value {
			return 0, false
		}
		total += value
	}
	return total, true
}

func AddInt(values ...int) (int, bool) {
	var total int
	for _, value := range values {
		if value < 0 || total > math.MaxInt-value {
			return 0, false
		}
		total += value
	}
	return total, true
}

func Mul64(left, right uint64) (uint64, bool) {
	if left != 0 && right > math.MaxUint64/left {
		return 0, false
	}
	return left * right, true
}

func MulInt(left, right int) (int, bool) {
	if left < 0 || right < 0 {
		return 0, false
	}
	product, ok := Mul64(uint64(left), uint64(right))
	if !ok {
		return 0, false
	}
	return Int(product)
}

// ProductInt multiplies non-negative integer factors with overflow checking.
func ProductInt(values ...int) (int, bool) {
	product := 1
	for _, value := range values {
		var ok bool
		product, ok = MulInt(product, value)
		if !ok {
			return 0, false
		}
	}
	return product, true
}

// PowInt raises a non-negative integer base to a non-negative integer power
// with overflow checking.
func PowInt(base, exponent int) (int, bool) {
	if exponent < 0 {
		return 0, false
	}
	product := 1
	for range exponent {
		var ok bool
		product, ok = MulInt(product, base)
		if !ok {
			return 0, false
		}
	}
	return product, true
}

func DivExact64(dividend, divisor uint64) (uint64, bool) {
	if divisor == 0 || dividend%divisor != 0 {
		return 0, false
	}
	return dividend / divisor, true
}

func DivExactInt(dividend, divisor int) (int, bool) {
	if dividend < 0 || divisor <= 0 {
		return 0, false
	}
	quotient, ok := DivExact64(uint64(dividend), uint64(divisor))
	if !ok {
		return 0, false
	}
	return Int(quotient)
}

func Int(value uint64) (int, bool) {
	if value > uint64(math.MaxInt) {
		return 0, false
	}
	return int(value), true
}

func Uint64(value int64) (uint64, bool) {
	if value < 0 {
		return 0, false
	}
	return uint64(value), true
}

func Bytes(elements, width uint64) (uint64, bool) {
	return Mul64(elements, width)
}

func Align(value, alignment uint64) (uint64, bool) {
	if alignment == 0 {
		return 0, false
	}
	mask := alignment - 1
	if alignment&mask != 0 || value > math.MaxUint64-mask {
		return 0, false
	}
	return (value + mask) &^ mask, true
}

// RoundUpMultiple rounds value to the next multiple without requiring a
// power-of-two divisor.
func RoundUpMultiple(value, multiple uint64) (uint64, bool) {
	if multiple == 0 {
		return 0, false
	}
	remainder := value % multiple
	if remainder == 0 {
		return value, true
	}
	return Add64(value, multiple-remainder)
}
