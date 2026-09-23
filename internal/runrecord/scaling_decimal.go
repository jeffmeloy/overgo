package runrecord

import (
	"math/big"

	"overgo/internal/binaryschema"
)

// ParseScalingDecimal returns an arbitrary-precision canonical nonnegative decimal, or nil for a noncanonical value.
func ParseScalingDecimal(value string) *big.Int {
	if value == "" || len(value) > 1 && value[0] == '0' {
		return nil
	}
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return nil
		}
	}
	parsed, ok := new(big.Int).SetString(value, binaryschema.DecimalRadix)
	if !ok {
		return nil
	}
	return parsed
}
