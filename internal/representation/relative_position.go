package representation

import (
	"errors"
	"math"

	"overgo/internal/checked"
)

// RelativePositionBuckets compiles the T5-style relative-position bucket for
// every query/key pair.
func RelativePositionBuckets(queryRows, keyRows, bucketCount, maximumDistance int, bidirectional bool) ([]int, error) {
	if !checked.PositiveInts(queryRows, keyRows, bucketCount, maximumDistance) {
		return nil, errors.New("representation: invalid relative-position program")
	}
	output := make([]int, queryRows*keyRows)
	for query := range queryRows {
		for key := range keyRows {
			output[query*keyRows+key] = relativePositionBucket(key-query, bucketCount, bidirectional, maximumDistance)
		}
	}
	return output, nil
}

func relativePositionBucket(relative, bucketCount int, bidirectional bool, maximumDistance int) int {
	bucket, distance := 0, relative
	if bidirectional {
		half := bucketCount / 2
		if distance > 0 {
			bucket += half
		}
		if distance < 0 {
			distance = -distance
		}
		bucketCount = half
	} else if distance > 0 {
		distance = 0
	} else {
		distance = -distance
	}
	exact := bucketCount / 2
	if distance < exact {
		return bucket + distance
	}
	if exact <= 0 {
		return bucket
	}
	large := exact + int(math.Log(float64(distance)/float64(exact))/math.Log(float64(maximumDistance)/float64(exact))*float64(bucketCount-exact))
	return bucket + min(large, bucketCount-1)
}
