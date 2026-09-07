package modelswap

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// keyRecorder: the proxy's provider-key intake under test.
type keyRecorder struct {
	reference, key string
	err            error
}

func (k *keyRecorder) SetKey(_ context.Context, reference, key string) error {
	k.reference, k.key = reference, key
	return k.err
}

// TestModelSwapProxyTakesProviderKeys: the key route lands in the proxy's
// intake and rides on to the running child with its body intact; the
// intake's refusal is the answer; a proxy without an intake refuses the
// route.
func TestModelSwapProxyTakesProviderKeys(t *testing.T) {
	launcher := &upstreamLauncher{}
	supervisor, err := New(launcher, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	keys := &keyRecorder{}
	alpha := Servable{Name: "alpha", Location: "alpha.gguf"}
	proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"alpha": alpha}, Default: alpha, Keys: keys}
	front := httptest.NewServer(proxy)
	defer front.Close()
	body := `{"location":"remote://fake/vendor/model","key":"secret"}`
	placed, err := http.Post(front.URL+"/providers/key", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	placedBody, _ := io.ReadAll(placed.Body)
	placed.Body.Close()
	if placed.StatusCode != http.StatusOK || keys.reference != "remote://fake/vendor/model" || keys.key != "secret" ||
		!strings.Contains(string(placedBody), "served-by:alpha path:/providers/key body:"+body) {
		t.Fatalf("placed status=%d keys=%+v body=%s", placed.StatusCode, *keys, placedBody)
	}
	keys.err = errors.New("no declared model")
	refused, err := http.Post(front.URL+"/providers/key", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	refused.Body.Close()
	if refused.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("refused status=%d", refused.StatusCode)
	}
	keyless := httptest.NewServer(&Proxy{Supervisor: supervisor, Resolver: mapResolver{"alpha": alpha}, Default: alpha})
	defer keyless.Close()
	absent, err := http.Post(keyless.URL+"/providers/key", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	absent.Body.Close()
	if absent.StatusCode != http.StatusNotImplemented {
		t.Fatalf("proxy without an intake status=%d", absent.StatusCode)
	}
}
