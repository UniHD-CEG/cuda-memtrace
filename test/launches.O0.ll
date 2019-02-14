; RUN: opt -load LLVMMemtrace.so -memtrace-locate-kcalls -analyze < %s | FileCheck %s
; ModuleID = 'launches-host-x86_64-unknown-linux-gnu.bc'
source_filename = "launches.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@0 = private unnamed_addr constant [9 x i8] c"_Z2k1Pfi\00", align 1
@1 = private constant [3329 x i8] c"P\EDU\BA\01\00\10\00\F0\0C\00\00\00\00\00\00\02\00\01\01@\00\00\00(\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00\1E\00\00\00\00\00\00\00\00\00\00\00\15\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00P\00\00\00\00\00\00\00\00\00\00\00\80\09\00\00\00\00\00\00@\07\00\00\00\00\00\00\1E\05\1E\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text._Z2k1Pfi\00.nv.info._Z2k1Pfi\00.nv.shared._Z2k1Pfi\00.nv.constant0._Z2k1Pfi\00.nv.global\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00_Z2k1Pfi\00.text._Z2k1Pfi\00.nv.info._Z2k1Pfi\00.nv.shared._Z2k1Pfi\00.nv.constant0._Z2k1Pfi\00_param\00.nv.global\00threadIdx\00blockIdx\00blockDim\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00;\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\8E\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\99\00\00\00\01\00\08\00\01\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\A3\00\00\00\01\00\08\00\02\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\AC\00\00\00\01\00\08\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\002\00\00\00\12\10\07\00\00\00\00\00\00\00\00\00@\03\00\00\00\00\00\00\04#\08\00\07\00\00\00\00\00\00\00\04\12\08\00\07\00\00\00\10\00\00\00\04\11\08\00\07\00\00\00\10\00\00\00\04\0A\08\00\02\00\00\00@\01\0C\00\03\19\0C\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0\11\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7B\80\C2\02\E0\82\22\E4]\00\10\01@\00(\03]\10@\00\C0\00H\04\1C\00\DC\00\00\00,#\DC\11\00\00\00\0E\1B\E7\01\00 \00\00\00@\07\C0\00\04\00\00\00\D0\03\1C\10\FC\00\00\00HG\80B\00B\80\02\22\E4\9D\00\00\00\00\00(\E4\DD\00\FC\00\00\00(\E4\9D\00\08\00\00\00(\E4\DD\00\0C\00\00\00(\E4\1D\01\90\00@\00(\E4]\01\FC\00\00\00(C\9C \10\00\00\00hG\80\E2\82\82B\80\22C\DC0\14\00\00\00h\E2\1D\00 \05\00\00\18\86\1C\01\00\00\00\00\14\E2\1D\00\00\05\00\00\18\A6\9C\01\00\00\00\00\14\E4\9D\01\18\00\00\00(\E4\DD\01\1C\00\00\00(G\00\C2B\80\82\E2\22\E4\1D\02\18\00\00\00(\E4]\02\1C\00\00\00(\03\9C!\FC\00\00\01HC\DC1\FC\00\00\00H\E4\9D\01\18\00\00\00(\E4\DD\01\1C\00\00\00(\A5\1Cb\00\00\00\00\94\C7B\80\82\02\82B\22\03\9C! \00\C0\01HC\DC1\FC\00\00\00H\E4\9D\01\18\00\00\00(\E4\DD\01\1C\00\00\00(\85\1Ca\00\00\00\00\94\04\1C\00\84\00\00\00,\E4\1D\00\00\00\00\00(\87B\80\82\02\C2\82\22\04\1C\01\94\00\00\00,\E4\1D\01\10\00\00\00(\E4]\01\A0\00@\00(\A3\1CA\14\00\00\00P\03\1C\00\10\00\00\00H\03\1C!0\00\C0\01HC\5C1\FC\00\00\00HG\80\E2\C2B\80\82\22\E4\1D\01\10\00\00\00(\E4]\01\14\00\00\00(\85\1C@\00\00\00\00\94\03\1C! \00\C0\01HC\5C1\FC\00\00\00H\E4\1D\01\10\00\00\00(\E4]\01\14\00\00\00(w\03\C2B\80\82\02\22\85\1C@\00\00\00\00\84\04\9E!\01\00\00\00\18\03\1C!\FC\00\00\01HC\5C1\FC\00\00\00H\E4\1D\02\10\00\00\00(\E4]\02\14\00\00\00(\A5\1C\81\00\00\00\00\84\C7B\80\82rC\00\22\03\9C 0\00\C0\01HC\DC0\FC\00\00\00H\E4\9D\00\08\00\00\00(\E4\DD\00\0C\00\00\00(\85\1C \00\00\00\00\84\E4\9D\00\00\00\00\00(#\DC\00\FC\00\00\8E\10\87B0B\82B\80\22\E2\DD\01\08\00\00\00\18\03\DCq\80\00\C0\8E\19\03\1Ep\80\00\C0\00H\03\DC0\1C\00\00\00`\03\1C\22\00\00\00\00X\03\DC0 \00\00\00H\03 p\80\FF\FF\00H7\82B\80\C2B\80\22\03\E0 \00\00\00\00`\03\1C \1C\00\00\00`\E4\9D\00\00\00\00\00(\E4\DD\00\0C\00\00\00(\03\9C@\08\00\00\01HC\DCP\0C\00\00\00H\E4\9D\00\08\00\00\00(\87r\03B\80B\00 \E4\DD\00\0C\00\00\00(\85\1C \00\00\00\00\84\00\1C\00\18\00\00\00P\E4\9D\00\08\00\00\00(\E4\DD\00\0C\00\00\00(\85\1C \00\00\00\00\94\E7\1D\00\00\00\00\00@\E7\E2\02\00\00\00\00 \E7\1D\00\00\00\00\00\80\E7\1D\00\00\00\00\00\80\E7\1D\00\E0\FF\FF\03@\E4\1D\00\00\00\00\00@\E4\1D\00\00\00\00\00@\E4\1D\00\00\00\00\00@\E4\1D\00\00\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\89\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C9\00\00\00\00\00\00\00\B5\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\01\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00A\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\07\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00g\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\02\00\00\00\00\00\00L\01\00\00\00\00\00\00\00\00\00\00\07\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00@\03\00\00\00\00\00\00\03\00\00\00\07\00\00\0A@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00~\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\07\00\00\00\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\80\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\94\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\8C\04\00\00\00\00\00\00\8C\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00@\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01H\00\00\00@\02\00\00\00\00\00\00:\02\00\00@\00\00\00\02\00\04\00\1E\00\00\00\00\00\00\00\00\00\00\00\15 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0 \0A\0A\0A\0A.version 4.2\0A.target sm_30\0A.address_size 64.\00\FF\12global .align 1 .b8 threadIdx[1];#\00\03_block\22\00\0F1Dim\22\00\F5\17\0A.visible .entry _Z2k1Pfi(\0A.param .u64\16\00\11_\14\006_0,\1E\00,32\1E\00\A61\0A)\0A{\0A.loc{\00\118{\00!__\15\00\A0_depot0[16\A4\00\CBreg .b64 %SP\0F\00\14L\10\00\95f32 %f<4>!\00\00\11\00Ir<8>2\00!rd\12\00c\0Amov.uC\00\1B,v\00b;\0Acvta\9E\00\04%\00\13,m\00\22ld\D7\00\01\D6\00l%r1, [\DC\00\18]&\00\02t\00\0F'\00\00#0]g\00#to\84\01\04-\00!2,3\00\03\1F\00\0A\1C\00\113\1C\00Q2;\0Ast\13\00q[%SP+0]\16\00\123\16\00\2232\16\00\118\16\00\221;\EB\00\01\AE\00\00Z\00Xtid.x\15\00\00S\00<cta\17\00T4, %n-\00qul.lo.s\19\00#5,5\00sr4;\0Aadd\17\00#6,a\00*r5\88\00!12\89\00\116\16\01\02\5C\00$7,\A3\00\01\E5\000.rn\BF\01\02I\00 f1.\00\127.\00\02\F5\00$4,\E8\00\01\16\00\02p\005d5,`\00T;\0Ashl\E0\01#6,\1E\00\132\9F\00\02\17\00#7,K\00\00#\00\01H\00\022\02\102G\00Brd7].\00#rn\18\00\223,\1D\001%f1\D0\00\00\16\00\02)\00\00\14\00\C03;\0Aret;\0A\0A}\0A\0A\00\00\00\00\00\00\00", section ".nv_fatbin", align 8
@__cuda_fatbin_wrapper = internal constant { i32, i32, i8*, i8* } { i32 1180844977, i32 1, i8* getelementptr inbounds ([3329 x i8], [3329 x i8]* @1, i64 0, i64 0), i8* null }, section ".nvFatBinSegment", align 8
@__cuda_gpubin_handle = internal global i8** null, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* bitcast (void (i8*)* @__cuda_module_ctor to void ()*), i8* null }]

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2k1Pfi(float* %a, i32 %b) #0 {
entry:
  %a.addr = alloca float*, align 8
  %b.addr = alloca i32, align 4
  store float* %a, float** %a.addr, align 8
  store i32 %b, i32* %b.addr, align 4
  %0 = bitcast float** %a.addr to i8*
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %setup.next, label %setup.end

setup.next:                                       ; preds = %entry
  %3 = bitcast i32* %b.addr to i8*
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 4, i64 8)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %setup.next1, label %setup.end

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*))
  br label %setup.end

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z11launches_k1Pf(float* %a) #0 {
entry:
  %a.addr = alloca float*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  %agg.tmp9 = alloca %struct.dim3, align 4
  %agg.tmp11 = alloca %struct.dim3, align 4
  %agg.tmp9.coerce = alloca { i64, i32 }, align 4
  %agg.tmp11.coerce = alloca { i64, i32 }, align 4
  %agg.tmp17 = alloca %struct.dim3, align 4
  %agg.tmp19 = alloca %struct.dim3, align 4
  %agg.tmp17.coerce = alloca { i64, i32 }, align 4
  %agg.tmp19.coerce = alloca { i64, i32 }, align 4
  %agg.tmp27 = alloca %struct.dim3, align 4
  %agg.tmp30 = alloca %struct.dim3, align 4
  %agg.tmp27.coerce = alloca { i64, i32 }, align 4
  %agg.tmp30.coerce = alloca { i64, i32 }, align 4
  store float* %a, float** %a.addr, align 8
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 8, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 8, i32 1, i32 1)
  %0 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*
  %1 = bitcast %struct.dim3* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 12, i1 false)
  %2 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0
  %3 = load i64, i64* %2, align 4
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1
  %5 = load i32, i32* %4, align 4
  %6 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*
  %7 = bitcast %struct.dim3* %agg.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 %7, i64 12, i1 false)
  %8 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0
  %9 = load i64, i64* %8, align 4
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1
  %11 = load i32, i32* %10, align 4
  %call = call i32 @cudaConfigureCall(i64 %3, i32 %5, i64 %9, i32 %11, i64 0, %struct.CUstream_st* null)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %kcall.end, label %kcall.configok

kcall.configok:                                   ; preds = %entry
  %12 = load float*, float** %a.addr, align 8
  call void @_Z2k1Pfi(float* %12, i32 4)
  br label %kcall.end

kcall.end:                                        ; preds = %kcall.configok, %entry
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp2, i32 8, i32 1, i32 1)
  %call4 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %call4, i32 1, i32 1)
  %13 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*
  %14 = bitcast %struct.dim3* %agg.tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %13, i8* align 4 %14, i64 12, i1 false)
  %15 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 0
  %16 = load i64, i64* %15, align 4
  %17 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 1
  %18 = load i32, i32* %17, align 4
  %19 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*
  %20 = bitcast %struct.dim3* %agg.tmp3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %19, i8* align 4 %20, i64 12, i1 false)
  %21 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0
  %22 = load i64, i64* %21, align 4
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1
  %24 = load i32, i32* %23, align 4
  %call5 = call i32 @cudaConfigureCall(i64 %16, i32 %18, i64 %22, i32 %24, i64 0, %struct.CUstream_st* null)
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7

kcall.configok7:                                  ; preds = %kcall.end
  %25 = load float*, float** %a.addr, align 8
  call void @_Z2k1Pfi(float* %25, i32 4)
  br label %kcall.end8

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call10 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp9, i32 %call10, i32 1, i32 1)
  %call12 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %call12, i32 1, i32 1)
  %26 = bitcast { i64, i32 }* %agg.tmp9.coerce to i8*
  %27 = bitcast %struct.dim3* %agg.tmp9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %26, i8* align 4 %27, i64 12, i1 false)
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 0
  %29 = load i64, i64* %28, align 4
  %30 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp9.coerce, i32 0, i32 1
  %31 = load i32, i32* %30, align 4
  %32 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*
  %33 = bitcast %struct.dim3* %agg.tmp11 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %32, i8* align 4 %33, i64 12, i1 false)
  %34 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0
  %35 = load i64, i64* %34, align 4
  %36 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1
  %37 = load i32, i32* %36, align 4
  %call13 = call i32 @cudaConfigureCall(i64 %29, i32 %31, i64 %35, i32 %37, i64 0, %struct.CUstream_st* null)
  %tobool14 = icmp ne i32 %call13, 0
  br i1 %tobool14, label %kcall.end16, label %kcall.configok15

kcall.configok15:                                 ; preds = %kcall.end8
  %38 = load float*, float** %a.addr, align 8
  call void @_Z2k1Pfi(float* %38, i32 4)
  br label %kcall.end16

kcall.end16:                                      ; preds = %kcall.configok15, %kcall.end8
  %call18 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp17, i32 %call18, i32 1, i32 1)
  %call20 = call i32 @_Z1fv()
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp19, i32 %call20, i32 1, i32 1)
  %39 = bitcast { i64, i32 }* %agg.tmp17.coerce to i8*
  %40 = bitcast %struct.dim3* %agg.tmp17 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %39, i8* align 4 %40, i64 12, i1 false)
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 0
  %42 = load i64, i64* %41, align 4
  %43 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 1
  %44 = load i32, i32* %43, align 4
  %45 = bitcast { i64, i32 }* %agg.tmp19.coerce to i8*
  %46 = bitcast %struct.dim3* %agg.tmp19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %45, i8* align 4 %46, i64 12, i1 false)
  %47 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp19.coerce, i32 0, i32 0
  %48 = load i64, i64* %47, align 4
  %49 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp19.coerce, i32 0, i32 1
  %50 = load i32, i32* %49, align 4
  %call21 = call i32 @cudaConfigureCall(i64 %42, i32 %44, i64 %48, i32 %50, i64 0, %struct.CUstream_st* null)
  %tobool22 = icmp ne i32 %call21, 0
  br i1 %tobool22, label %kcall.end26, label %kcall.configok23

kcall.configok23:                                 ; preds = %kcall.end16
  %call24 = call float* @_Z1gv()
  %call25 = call i32 @_Z1fv()
  call void @_Z2k1Pfi(float* %call24, i32 %call25)
  br label %kcall.end26

kcall.end26:                                      ; preds = %kcall.configok23, %kcall.end16
  %call28 = call i32 @_Z1fv()
  %call29 = call i32 @_Z1fv()
  %add = add nsw i32 %call28, %call29
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp27, i32 %add, i32 1, i32 1)
  %call31 = call i32 @_Z1fv()
  %mul = mul nsw i32 %call31, 2
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp30, i32 %mul, i32 1, i32 1)
  %51 = bitcast { i64, i32 }* %agg.tmp27.coerce to i8*
  %52 = bitcast %struct.dim3* %agg.tmp27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %51, i8* align 4 %52, i64 12, i1 false)
  %53 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 0
  %54 = load i64, i64* %53, align 4
  %55 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp27.coerce, i32 0, i32 1
  %56 = load i32, i32* %55, align 4
  %57 = bitcast { i64, i32 }* %agg.tmp30.coerce to i8*
  %58 = bitcast %struct.dim3* %agg.tmp30 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %57, i8* align 4 %58, i64 12, i1 false)
  %59 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp30.coerce, i32 0, i32 0
  %60 = load i64, i64* %59, align 4
  %61 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp30.coerce, i32 0, i32 1
  %62 = load i32, i32* %61, align 4
  %call32 = call i32 @cudaConfigureCall(i64 %54, i32 %56, i64 %60, i32 %62, i64 0, %struct.CUstream_st* null)
  %tobool33 = icmp ne i32 %call32, 0
  br i1 %tobool33, label %kcall.end39, label %kcall.configok34

kcall.configok34:                                 ; preds = %kcall.end26
  %call35 = call float* @_Z1gv()
  %call36 = call i32 @_Z1fv()
  %call37 = call i32 @_Z1fv()
  %add38 = add nsw i32 %call36, %call37
  call void @_Z2k1Pfi(float* %call35, i32 %add38)
  br label %kcall.end39

; CHECK-LABEL: name:   _Z2k1Pfi
; CHECK:       config:   %call32 = call i32 @cudaConfigureCall
; CHECK:       launch:   call void @_Z2k1Pfi

kcall.end39:                                      ; preds = %kcall.configok34, %kcall.end26
  ret void
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  store i32 %vx, i32* %vx.addr, align 4
  store i32 %vy, i32* %vy.addr, align 4
  store i32 %vz, i32* %vz.addr, align 4
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0
  %0 = load i32, i32* %vx.addr, align 4
  store i32 %0, i32* %x, align 4
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1
  %1 = load i32, i32* %vy.addr, align 4
  store i32 %1, i32* %y, align 4
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2
  %2 = load i32, i32* %vz.addr, align 4
  store i32 %2, i32* %z, align 4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #3

declare dso_local i32 @_Z1fv() #1

declare dso_local float* @_Z1gv() #1

define internal void @__cuda_register_globals(i8**) {
entry:
  %1 = call i32 @__cudaRegisterFunction(i8** %0, i8* bitcast (void (float*, i32)* @_Z2k1Pfi to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @0, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @0, i64 0, i64 0), i32 -1, i8* null, i8* null, i8* null, i8* null, i32* null)
  ret void
}

declare dso_local i32 @__cudaRegisterFunction(i8**, i8*, i8*, i8*, i32, i8*, i8*, i8*, i8*, i32*)

declare dso_local i32 @__cudaRegisterVar(i8**, i8*, i8*, i8*, i32, i32, i32, i32)

declare dso_local i8** @__cudaRegisterFatBinary(i8*)

define internal void @__cuda_module_ctor(i8*) {
entry:
  %1 = call i8** @__cudaRegisterFatBinary(i8* bitcast ({ i32, i32, i8*, i8* }* @__cuda_fatbin_wrapper to i8*))
  store i8** %1, i8*** @__cuda_gpubin_handle, align 8
  call void @__cuda_register_globals(i8** %1)
  %2 = call i32 @atexit(void (i8*)* @__cuda_module_dtor)
  ret void
}

declare dso_local void @__cudaUnregisterFatBinary(i8**)

define internal void @__cuda_module_dtor(i8*) {
entry:
  %1 = load i8**, i8*** @__cuda_gpubin_handle, align 8
  call void @__cudaUnregisterFatBinary(i8** %1)
  ret void
}

declare dso_local i32 @atexit(void (i8*)*)

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 (https://github.com/llvm-mirror/clang f3e3e0682c76f3e027ce0e21c05049ff48ccdd6f) (https://github.com/llvm-mirror/llvm a5b9a59eac7814d1276784cec74bbcfd6449d318)"}
