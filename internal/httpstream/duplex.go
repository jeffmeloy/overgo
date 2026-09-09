// Package httpstream owns bidirectional HTTP request lifetime.
package httpstream

import (
	"context"
	"net/http"
	"time"
)

// OpenDuplex permits responses before request EOF. Its release must run before
// the handler returns: it joins cancellation and closes the body while the
// handler still owns the connection, preventing a concurrent net/http drain.
func OpenDuplex(response http.ResponseWriter, request *http.Request) (func(), error) {
	controller := http.NewResponseController(response)
	if request.ProtoMajor == 1 {
		if err := controller.EnableFullDuplex(); err != nil {
			return nil, err
		}
		// Application finalization can precede request-body EOF. HTTP/1 must
		// advertise that this connection cannot carry a subsequent request;
		// otherwise a late chunk terminator can become a spurious next request.
		response.Header().Set("Connection", "close")
	}
	interrupted := make(chan struct{})
	stop := context.AfterFunc(request.Context(), func() {
		_ = controller.SetReadDeadline(time.Now())
		_ = controller.SetWriteDeadline(time.Now())
		close(interrupted)
	})
	return func() {
		if !stop() {
			<-interrupted
		}
		_ = controller.SetReadDeadline(time.Now())
		_ = request.Body.Close()
	}, nil
}
