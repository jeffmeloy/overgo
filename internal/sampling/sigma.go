package sampling

import (
	"errors"

	"overgo/internal/checked"
)

// ValidateSigmaSchedule validates an open-unit schedule whose terminal value
// may be zero while all preceding values are positive.
func ValidateSigmaSchedule(sigmas []float32) error {
	if len(sigmas) < 2 {
		return errors.New("sampling: sigma schedule is too short")
	}
	for index, sigma := range sigmas {
		if !checked.NonNegativeFinite32(sigma) || sigma >= 1 || index < len(sigmas)-1 && sigma <= 0 {
			return errors.New("sampling: sigma schedule is invalid")
		}
	}
	return nil
}
