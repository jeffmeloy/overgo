package checked

import (
	"errors"
	"math"
)

func Nonzero[T comparable](value T) bool {
	var zero T
	return value != zero
}

func NonzeroAll[T comparable](values ...T) bool {
	for _, value := range values {
		if !Nonzero(value) {
			return false
		}
	}
	return true
}

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

func NonNegativeFinite64(value float64) bool { return value >= 0 && Finite64(value) }

func AtLeastFinite64(value, minimum float64) bool {
	return Finite64(value) && Finite64(minimum) && value >= minimum
}

func Negative64(value float64) bool { return value < 0 }

func NonNegativeInt64(value int64) bool { return value >= 0 }

func AtMostInt64(value, maximum int64) bool { return value <= maximum }

func GreaterInt(left, right int) bool { return left > right }

func AtMostInt(value, maximum int) bool { return value <= maximum }

func AtLeastInt(value, minimum int) bool { return value >= minimum }

func AutomaticOrNonNegative(value int) bool { return value >= -1 }

// UnknownCount returns the diagnostic sentinel for an unavailable count.
func UnknownCount() int { return -1 }

func ExactFloat32Uint(value uint32) bool { return uint32(float32(value)) == value }

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

// Less64 compares unsigned extents without assigning domain meaning.
func Less64(left, right uint64) bool { return left < right }

// AtMost64 compares bounded unsigned extents.
func AtMost64(value, limit uint64) bool { return value <= limit }

func First[T any](values []T) (T, bool) {
	var zero T
	if len(values) == 0 {
		return zero, false
	}
	return values[0], true
}

func Last[T any](values []T) (T, bool) {
	var zero T
	if len(values) == 0 {
		return zero, false
	}
	return values[len(values)-1], true
}

func Multiple(count int) bool { return count > 1 }

func EvenInt(value int) bool { return value > 0 && value%2 == 0 }

func Empty[T any](values ...[]T) bool {
	for _, value := range values {
		if len(value) != 0 {
			return false
		}
	}
	return true
}

func Nonempty[T any](value []T) bool { return len(value) != 0 }

func NonemptyAll[T any](values ...[]T) bool {
	for _, value := range values {
		if len(value) == 0 {
			return false
		}
	}
	return true
}

func Tail[T any](values []T) ([]T, bool) {
	if len(values) == 0 {
		return nil, false
	}
	return values[1:], true
}

func Init[T any](values []T) ([]T, bool) {
	if len(values) == 0 {
		return nil, false
	}
	return values[:len(values)-1], true
}

func Reset[T any](values []T) []T { return values[:0] }

func LastSlice[T any](values []T) ([]T, bool) {
	if len(values) == 0 {
		return nil, false
	}
	return values[len(values)-1:], true
}

func Suffix[T any](values []T, count int) ([]T, bool) {
	if count < 0 || count > len(values) {
		return nil, false
	}
	return values[len(values)-count:], true
}

func Prefix[T any](values []T, count int) ([]T, bool) {
	if count < 0 || count > len(values) {
		return nil, false
	}
	return values[:count], true
}

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

func Index(value, length int) int { return max(0, min(value, length)) }

func ValidIndex(value, length int) bool { return value >= 0 && value < length }

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

func ClampNonNegative(value int) int { return max(0, value) }

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
