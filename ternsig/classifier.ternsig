; ClassifierANT - Multi-class classification
; Input: 32, Hidden: 24, Classes: 4, Iterations: 3
;
; Architecture:
;   input -> MATMUL -> ReLU -> [recurrent x3] -> MATMUL -> class_logits
;   Use argmax on output for predicted class
;
; Total params: 24*32 + 24*24 + 24*24 + 4*24 = 1920 ternary weights

.registers
    H0: i32[32]      ; input
    H1: i32[24]      ; hidden state
    H2: i32[24]      ; update
    H3: i32[24]      ; gate
    H4: i32[4]       ; output (class logits)

    C0: ternary[24, 32]  key="ant.classifier.w_in"    ; input projection
    C1: ternary[24, 24]  key="ant.classifier.w_rec"   ; recurrent weights
    C2: ternary[24, 24]  key="ant.classifier.w_gate"  ; gate weights
    C3: ternary[4, 24]   key="ant.classifier.w_out"   ; output projection

.program
    ; Input projection
    load_input H0
    ternary_matmul H1, C0, H0
    relu H1, H1

    ; Recurrent iteration 1
    ternary_matmul H2, C1, H1
    relu H2, H2
    ternary_matmul H3, C2, H1
    sigmoid H3, H3
    mul H2, H2, H3
    add H1, H1, H2
    shift H1, H1, 1

    ; Recurrent iteration 2
    ternary_matmul H2, C1, H1
    relu H2, H2
    ternary_matmul H3, C2, H1
    sigmoid H3, H3
    mul H2, H2, H3
    add H1, H1, H2
    shift H1, H1, 1

    ; Recurrent iteration 3
    ternary_matmul H2, C1, H1
    relu H2, H2
    ternary_matmul H3, C2, H1
    sigmoid H3, H3
    mul H2, H2, H3
    add H1, H1, H2
    shift H1, H1, 1

    ; Output (class logits)
    ternary_matmul H4, C3, H1
    store_output H4
    halt
