package gpt2

// #cgo CFLAGS: -I${SRCDIR}/ggml.cpp/include/ -I${SRCDIR}/ggml.cpp/include/ggml/ -I${SRCDIR}/ggml.cpp/examples/ -I${SRCDIR}/ggml.cpp/src/
// #cgo CXXFLAGS: -I${SRCDIR}/ggml.cpp/include/ -I${SRCDIR}/ggml.cpp/include/ggml/ -I${SRCDIR}/ggml.cpp/examples/ -I${SRCDIR}/ggml.cpp/src/
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltransformers -lm -lstdc++
// #include <falcon.h>
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

type Falcon struct {
	state unsafe.Pointer
}

func NewFalcon(model string) (*Falcon, error) {
	state := C.falcon_allocate_state()
	modelPath := C.CString(model)
	result := C.falcon_bootstrap(modelPath, state)
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &Falcon{state: state}, nil
}

func (l *Falcon) Predict(text string, opts ...PredictOption) (string, error) {

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.falcon_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.int(po.Batch))
	ret := C.falcon_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")
	C.falcon_free_params(params)

	return res, nil
}

func (l *Falcon) Free() {
	C.falcon_free_model(l.state)
}
