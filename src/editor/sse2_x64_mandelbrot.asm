public read_timestamp_counter
public cpu_ident
public MandelbrotLineSSE2x64Two

.code

; x64 __fastcall:
;
; RCX: 1st integer argument
; RDX: 2nd integer argument
; R8: 3rd integer argument
; R9: 4th integer argument
;
; XMM0: 1st floating point parameter
; XMM1: 2nd floating point parameter
; XMM2: 3rd floating point parameter
; XMM3: 4th floating point parameter
; 
; - Integer arguments beyond the first four are passed on the stack.
; - Floating point arguments beyond the first four are passed on the stack.
; - The this pointer is considered an integer argument and is passed in RCX.
; - When mixing integer and floating point arguments, an argument of one type
;   also takes up one register of the opposite kind. So, if two arguments are
;   passed, the first an int and the second a float, the float is passed in
;   XMM1, not in XMM0.
;
; RAX: Integer return.
; XMM0: Floating point return
;
; Must be preserved across calls: RBX, RBP, RDI, RSI, R12, R13, R14, R15 and XMM6 to XMM15.
; Can be destroyed: RAX, RCX, RDX, R8, R9, R10, and R11.
;
; register usage: http://msdn.microsoft.com/en-us/library/9z1stfyw(VS.80).aspx
;
; Mandelbrot calc:
;
; double zi(0), zr(0), zr2(0), zi2(0);
; while(--iter && zr2 + zi2 > 4) {
;  zi = 2 * zr * zi + ci;
;  zr = (zr2 - zi2) + cr;
;  zr2 = zr * zr;
;  zi2 = zi * zi;
; }
;
; __int64 q = calc_sse2_x64(cr0, ci0, cr1, ci1, bailout);
;   cr0          xmm0
;   ci0          xmm1
;   cr1          xmm2
;   ci1          xmm3
;   bailout      rsp + 20
;   return       rax

align 8
MandelbrotLineSSE2x64Two proc frame
  ; FRAME SETUP

  push rbp
  .pushreg rbp

  sub rsp, 0200h
  .allocstack 0200h

  lea rbp, [rsp + 080]
  .setframe rbp, 080h

  movdqa [rbp], xmm5
  .Savexmm128 xmm5, 080h + 0

  movdqa [rbp + 010h], xmm6
  .Savexmm128 xmm6, 080h + 010h

  movdqa [rbp + 020h], xmm7
  .Savexmm128 xmm7, 080h + 020h

  movdqa [rbp + 030h], xmm8
  .Savexmm128 xmm8, 080h + 030h

  movdqa [rbp + 040h], xmm9
  .Savexmm128 xmm9, 080h + 040h

  movdqa [rbp + 050h], xmm10
  .Savexmm128 xmm10, 080h + 050h

  movdqa [rbp + 060h], xmm11
  .Savexmm128 xmm11, 080h + 060h
  
  movdqa [rbp + 070h], xmm12
  .Savexmm128 xmm12, 080h + 070h

  movdqa [rbp + 080h], xmm13
  .Savexmm128 xmm13, 080h + 080h

  movdqa [rbp + 090h], xmm14
  .Savexmm128 xmm14, 080h + 090h

  movdqa [rbp + 100h], xmm15
  .Savexmm128 xmm15, 080h + 100h

  .endprolog

  ; Attempt to make code faster by turning on
  ; faster handling of denormals.
  stmxcsr cw1
  mov eax, cw1
  or eax, 001000000001000000b
  ;         5432109876543210
  mov cw2, eax
  ldmxcsr cw2

  ; Get bailout from the stack.
  mov ecx, [rsp+230h]

  ; initial values for Z are zero
  xorpd xmm12, xmm12       ; zr1 zr0
  xorpd xmm13, xmm13       ; zi1 zi0
  xorpd xmm14, xmm14       ; zr21 zr20
  xorpd xmm15, xmm15       ; zi21 zi20

  movapd xmm6, fours       ; two copies of exit test: 4.0 4.0

  shufpd  xmm0, xmm2, 0    ; cr1 cr0
  shufpd  xmm1, xmm3, 0    ; ci1 ci0

  mov r10, 0
  mov r11, 0

  mov r8, rcx              ; preserve the bailout value
  ;mov r8, 2000              ; preserve the bailout value
  ;mov rcx, 2000

l1:  
  ; zi = 2 * zr * zi + ci;
  mulpd xmm13, xmm12       ; zr * zi
  addpd xmm13, xmm13       ; * 2
  addpd xmm13, xmm1        ; + ci
  
  ; zr = (zr2 - zi2) + cr
  movapd xmm11, xmm14      ;
  subpd xmm11, xmm15       ; zr2 - zi2
  addpd xmm11, xmm0        ; + cr
  movapd xmm12, xmm11
  
  ; zr2 = zr * zr
  movapd xmm14, xmm12
  mulpd xmm14, xmm14
  
  ; zi2 = zi * zi;
  movapd xmm15, xmm13
  mulpd  xmm15, xmm15

  ; zr2 + zi2
  movapd xmm11, xmm14
  addpd xmm11, xmm15
  
  ; set bits if less than 4.0 4.0
  cmpltpd xmm11, xmm6
  
  ; extract 2-bit sign mask of from xmm and store in r32
  movmskpd rdx, xmm11
  cmp rdx, 0
  je done

  ; two counters in one register
  ror edx, 1
  add r10, rdx
  
  ; limit iterations to bailout
  dec rcx
  jne l1

done:
  ; split the two counters in r10 into r10 and r11
  mov r11, r10
  and r10, 07fffffffh
  shr r11, 31
  
  ; set counters that reached bailout to zero
  xor r9, r9
  cmp r10, r8
  cmove r10, r9
  cmp r11, r8
  cmove r11, r9
  
  ; join two counters into one return register
  mov rax, r10
  shl rax, 32
  or rax, r11

  mov rcx, 0

  ; UNPREP
  
  movdqa      xmm5, [rbp]
  movdqa      xmm6, [rbp + 010h]
  movdqa      xmm7, [rbp + 020h]
  movdqa      xmm8, [rbp + 030h]
  movdqa      xmm9, [rbp + 040h]
  movdqa      xmm10, [rbp + 050h]
  movdqa      xmm11, [rbp + 060h]
  movdqa      xmm12, [rbp + 070h]
  movdqa      xmm13, [rbp + 080h]
  movdqa      xmm14, [rbp + 090h]
  movdqa      xmm15, [rbp + 100h]

  ;ldmxcsr cw1

  ; official epilog
  add rsp, 0200h
  pop rbp
  ret

align 16

fours  real8 4.0
    real8 4.0

.data

cw1  dword 0
cw2 dword 0

.code

align 8
MandelbrotLineSSE2x64Two endp

;-------------------------------------------------------------------------------------------
; void ReadTimeStampCounter (void *pui64Clocks (rcx));
;
; Query the cpu for clock count.
; Returns non-zero if the rdtsc instruction is available and clock count in *pui64Clocks.
; Returns zero if the rdtsc instruction is non available. Sets *pui64Clocks = 0.
;-------------------------------------------------------------------------------------------
;align 8
read_timestamp_counter proc frame

  ;.allocstack 0
  .endprolog

  rdtsc              ; query the clock stamp counter
  mov dword ptr [rcx], eax    ; *pui64Clocks(low dword) = eax
  mov dword ptr [rcx + 4], edx  ; *pui64Clocks(high dword) = edx

  ret

;align 8
read_timestamp_counter endp

;-------------------------------------------------------------------------------------------
; BOOL cpu_ident (DWORD aCpuInfo [4] (rcx), size_t InstructionRequestLevel (rdx));
;
; Returns zero if the InstructionRequestLevel is not supported.
; Returns non-zero if the InstructionRequestLevel is supported. 
;-------------------------------------------------------------------------------------------
;align 8
cpu_ident proc frame

  sub rsp, 8
  ;.allocstack 8
  mov [rsp], rbx      ; Save rbx.
  .Savereg rbx, 0
  .endprolog 

  mov r8, rdx        ; r8 = Instruction level request
  mov r9, rcx        ; r9 -> dword array

  xor rax, rax        ; rax = 0
  cpuid

  cmp r8, rax        ; Is the instruction level supported?
  jg short NotSupported  ; No. Our requested instruction level
              ; is greater than the supported level.

  mov rax, r8        ; Yes, r8 <= the supported level.
  cpuid
  mov [r9], eax
  mov [r9 + 4], ebx
  mov [r9 + 8], ecx
  mov [r9 + 12], edx

  mov rax, -1        ; return non-zero
  jmp short Done

;align 8
NotSupported:
  xor rax, rax      ; return zero
  mov [r9], rax      ; and set all array values to zero
  mov [r9 + 4], rax
  mov [r9 + 8], rax
  mov [r9 + 12], rax

;align 8
Done:    
  mov rbx, [rsp]
  add rsp, 8
  ret

;align 8
cpu_ident endp

end
