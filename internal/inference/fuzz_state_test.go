package inference

import (
	"testing"

	"llamacpp2go/internal/sampling"
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

func FuzzSessionStateNeverPanics(f *testing.F) {
	f.Add([]byte(sessionStateMagic))
	f.Add([]byte("not-session"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 16<<20 {
			t.Skip()
		}
		runner := cacheTestRunner()
		runner.spec.Name = "session-fixture"
		runner.spec.ContextLength = 16
		runner.spec.VocabularySize = 32
		sampler, err := sampling.New(sampling.Config{})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = runner.LoadSession(data, sampler)
	})
}
