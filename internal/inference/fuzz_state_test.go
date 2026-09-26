package inference

import (
	"testing"
)

func FuzzCacheStateNeverPanics(f *testing.F) {
	runner := cacheTestRunner()
	valid, err := runner.SaveCache(cacheTestValue(f))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte(cacheStateMagic))
	f.Add([]byte("not-cache"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 8<<20 {
			t.Skip()
		}
		_, _ = cacheTestRunner().LoadCache(data)
	})
}
