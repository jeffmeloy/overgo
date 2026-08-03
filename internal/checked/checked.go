package checked

import "math"

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

func Mul64(left, right uint64) (uint64, bool) {
	if left != 0 && right > math.MaxUint64/left {
		return 0, false
	}
	return left * right, true
}

func Int(value uint64) (int, bool) {
	if value > uint64(math.MaxInt) {
		return 0, false
	}
	return int(value), true
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
