package main

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/protection"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8099", "service address")
	storePath := flag.String("store", "sealed-repodb", "service-owned OvergoDB")
	publicText := flag.String("public-key", "", "base64 Ed25519 authority public key")
	maxRequestBytes := flag.Int64("max-request-bytes", 0, "required signed-request byte bound")
	readHeaderTimeout := flag.Duration("read-header-timeout", 0, "required HTTP header timeout")
	tlsCertificate := flag.String("tls-certificate", "", "TLS certificate path")
	tlsKey := flag.String("tls-key", "", "TLS private-key path")
	publishSpec := flag.String("publish-spec", "", "strict sealed-write specification")
	privateKey := flag.String("private-key", "", "base64 Ed25519 private-key file")
	endpoint := flag.String("endpoint", "", "sealed authority HTTPS endpoint")
	flag.Parse()
	if *publishSpec != "" {
		if err := publish(*publishSpec, *privateKey, *endpoint); err != nil {
			fmt.Fprintln(os.Stderr, "sealed-authority:", err)
			os.Exit(1)
		}
		return
	}
	if err := run(*listen, *storePath, *publicText, *tlsCertificate, *tlsKey, *maxRequestBytes, *readHeaderTimeout); err != nil {
		fmt.Fprintln(os.Stderr, "sealed-authority:", err)
		os.Exit(1)
	}
}

func publish(specPath, privateKeyPath, endpoint string) error {
	if specPath == "" || privateKeyPath == "" || endpoint == "" {
		return errors.New("publish spec, private key, and endpoint required")
	}
	var spec protection.SealedWriteSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	encoded, err := os.ReadFile(privateKeyPath)
	if err != nil {
		return err
	}
	private, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil || len(private) != ed25519.PrivateKeySize {
		return errors.New("valid private key required")
	}
	write, err := protection.SignSealedWrite(ed25519.PrivateKey(private), spec)
	if err != nil {
		return err
	}
	id, err := protection.PublishSealed(context.Background(), http.DefaultClient, endpoint, write)
	if err == nil {
		fmt.Println(id)
	}
	return err
}

func run(listen, storePath, publicText, tlsCertificate, tlsKey string, maxRequestBytes int64, readHeaderTimeout time.Duration) error {
	public, err := base64.StdEncoding.DecodeString(publicText)
	if err != nil || len(public) != ed25519.PublicKeySize || tlsCertificate == "" || tlsKey == "" || maxRequestBytes <= 0 || readHeaderTimeout <= 0 {
		return errors.New("valid public key, TLS identity, and positive request/time bounds required")
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	authority, err := protection.NewSealedAuthority(ed25519.PublicKey(public), store,
		protection.SealedAuthorityConfig{MaxRequestBytes: maxRequestBytes})
	if err != nil {
		return err
	}
	server := &http.Server{Addr: listen, Handler: authority, ReadHeaderTimeout: readHeaderTimeout}
	return server.ListenAndServeTLS(tlsCertificate, tlsKey)
}
